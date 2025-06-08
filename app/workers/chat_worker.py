# ai/app/workers/chat_worker.py

import asyncio
import json
import sys
from pathlib import Path
import aiohttp  # aiohttp 임포트
from typing import Dict, Optional

_is_logging_setup_successful = False
_fallback_logger_instance = None  # 타입 명시를 위해 Optional[logging.Logger] 사용 가능

try:
    from app.utils.logger import setup_logging, get_logger

    setup_logging()
    _is_logging_setup_successful = True
except ImportError as e_logging_import:
    print(f"CRITICAL: Failed to import or setup logging from app.utils.logger: {e_logging_import}")
    import logging  # 폴백 로깅

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    _fallback_logger_instance = logging.getLogger("FallbackLogger")
    _fallback_logger_instance.critical("Logging setup failed, using basic print/logging.")


# get_logger를 정의 (폴백 포함)
# 이 함수는 위 try-except 블록 이후에 정의되어야 _is_logging_setup_successful 값을 정확히 사용 가능
def get_logger_internal(name_param):  # type: ignore
    if _is_logging_setup_successful:
        # app.utils.logger.get_logger를 사용하기 위해선, 위에서 임포트 성공 시 해당 함수를 변수에 할당해야 함.
        # 여기서는 utils_get_logger를 호출한다고 가정 (setup_logging과 함께 임포트 되었다면)
        # from app.utils.logger import get_logger as utils_get_logger # 이런식으로 별도 이름으로 가져오거나,
        # 또는 setup_logging() 호출 후 app.utils.logger.get_logger 직접 사용
        # 현재 구조에서는 utils_get_logger가 없으므로, logging.getLogger 사용 또는 utils의 get_logger를 다시 임포트해야 함.
        # 더 간단하게는, 위에서 from app.utils.logger import get_logger를 했으므로 바로 사용.
        return get_logger(name_param)  # utils에서 임포트된 get_logger 사용
    elif _fallback_logger_instance:
        return _fallback_logger_instance  # 폴백 로거 반환
    else:  # _fallback_logger_instance도 None인 극단적 상황 (basicConfig도 실패한 경우)
        class PrintLogger:
            def __init__(self, name): self.name = name

            def info(self, msg, *args, **kwargs): print(f"INFO: {self.name}: {msg}")

            def warning(self, msg, *args, **kwargs): print(f"WARN: {self.name}: {msg}")

            def error(self, msg, *args, **kwargs): print(f"ERROR: {self.name}: {msg}")

            def critical(self, msg, *args, **kwargs): print(f"CRIT: {self.name}: {msg}")

            def debug(self, msg, *args, **kwargs): print(f"DEBUG: {self.name}: {msg}")

        return PrintLogger(name_param)


# --- 나머지 모듈 임포트 ---
_core_imports_successful = False
try:
    from app.services.database_client import DatabaseClient
    from app.services.llm_service import LLMService
    from app.services.postgresql_service import PostgreSQLService
    from app.services.backend_client import BackendApiClient
    from app.config.settings import Settings
    from .handlers.chat_handler import handle_chat_processing
    from .handlers.summary_handler import handle_summary_update

    _core_imports_successful = True
except ImportError as e_core_import:
    err_msg = f"CRITICAL Error importing core service modules in chat_worker.py: {e_core_import}"
    print(err_msg)
    logger_critical = get_logger_internal("ChatWorkerCriticalImport")
    logger_critical.critical(err_msg, exc_info=True)

if not _core_imports_successful:
    sys.exit(1)

TOKENIZER = None
logger_init = get_logger_internal("TokenizerInitWorker")
try:
    from transformers import AutoTokenizer

    current_script_dir = Path(__file__).resolve().parent

    settings_for_tokenizer = Settings()
    tokenizer_path_from_settings = getattr(settings_for_tokenizer, 'TOKENIZER_PATH', None)

    paths_to_check_tokenizer = [
        current_script_dir / "handlers" / "qwen_tokenizer",
        current_script_dir / "handlers"
    ]
    QWEN_TOKENIZER_PATH = None

    if tokenizer_path_from_settings and Path(tokenizer_path_from_settings).is_dir():
        QWEN_TOKENIZER_PATH = Path(tokenizer_path_from_settings)
    else:
        for p in paths_to_check_tokenizer:
            if p.is_dir() and any(p.glob('tokenizer*.json')):
                QWEN_TOKENIZER_PATH = p
                break

    if QWEN_TOKENIZER_PATH:
        TOKENIZER = AutoTokenizer.from_pretrained(str(QWEN_TOKENIZER_PATH), trust_remote_code=True)
        logger_init.info(f"Tokenizer loaded successfully from {QWEN_TOKENIZER_PATH}.")
    else:
        searched_paths_str = ", ".join(str(p) for p in paths_to_check_tokenizer)
        logger_init.error(
            f"Tokenizer directory NOT found. Settings path: '{tokenizer_path_from_settings if tokenizer_path_from_settings else 'N/A'}', Searched paths: {searched_paths_str}. Please check configuration.")
except Exception as e_tokenizer:
    logger_init.error(f"Failed to load Qwen Tokenizer: {e_tokenizer}", exc_info=True)

logger = get_logger_internal(__name__)

settings_instance = Settings()
QUEUE_NAME = getattr(settings_instance, 'CHAT_TASK_QUEUE_NAME', "chat_task_queue")

TASK_HANDLERS = {
    "process_chat": handle_chat_processing,
    "update_summary": handle_summary_update,
}


async def worker_loop():
    logger.info("Initializing services for Chat Worker Dispatcher...")

    # 서비스 변수들을 루프 외부에서 None으로 초기화 (finally 블록에서의 참조 오류 방지)
    redis_client: Optional[DatabaseClient] = None
    pg_service: Optional[PostgreSQLService] = None
    llm_service: Optional[LLMService] = None
    http_session: Optional[aiohttp.ClientSession] = None  # http_session도 외부에서 초기화

    try:
        redis_client = DatabaseClient()
        pg_service = PostgreSQLService()
        if TOKENIZER is None:
            logger.error("CRITICAL: Tokenizer is not loaded. Chat Worker cannot start properly.")
            return
        llm_service = LLMService(tokenizer=TOKENIZER)

        http_session = aiohttp.ClientSession()  # async with 대신 try-finally로 세션 관리
        backend_client = BackendApiClient(http_session)

        await pg_service.connect()
        logger.info("PostgreSQL connection established for ChatWorker.")
        logger.info(f"Chat Worker Dispatcher started... Waiting for tasks in queue: '{QUEUE_NAME}'")

        while True:
            raw_pop_result: Optional[tuple] = None
            work_id_for_log = "N/A_POP"

            try:
                # redis_client가 None이 아님을 확신하고 호출
                if redis_client:
                    raw_pop_result = await redis_client.brpop(QUEUE_NAME, timeout=60)
                else:
                    logger.error("Redis client is not initialized. Cannot pop from queue.")
                    await asyncio.sleep(5)  # 잠시 대기 후 재시도 또는 종료
                    continue

                if raw_pop_result:
                    _queue_name, task_json = raw_pop_result

                    task_data_parsed: Optional[Dict] = None
                    task_type = "unknown_task_type"
                    payload = {}

                    try:
                        task_data_parsed = json.loads(task_json)
                        task_type = task_data_parsed.get("type", "unknown_type_in_payload")
                        payload = task_data_parsed.get("payload", {})
                        work_id_for_log = payload.get('work_id', 'N/A_JSON_WORK_ID')

                        log_extra = {"trace_id": work_id_for_log}

                        logger.debug(f"Received raw task: {task_json[:300]}...", extra=log_extra)

                        handler_func = TASK_HANDLERS.get(task_type)

                        if handler_func:
                            logger.info(
                                f"Dispatching task '{task_type}'.", extra=log_extra)
                            # 핸들러 호출 시 서비스 객체들이 None이 아닌지 확인 (이론상 이 시점에는 항상 초기화됨)
                            if llm_service and pg_service and redis_client and backend_client:
                                await handler_func(
                                    payload, llm_service, pg_service,
                                    redis_client, TOKENIZER, backend_client
                                )
                            else:
                                logger.error("One or more services are not initialized. Cannot dispatch task.",
                                             extra=log_extra)
                        else:
                            logger.error(
                                f"No handler for task type: '{task_type}'. Task Data: {task_data_parsed}",
                                extra=log_extra)

                    except json.JSONDecodeError as e_json:
                        logger.error(f"Failed to decode JSON task data: '{task_json}'. Error: {e_json}",
                                     extra={"trace_id": "N/A_JSON_DECODE"})
                    except Exception as e_task:
                        logger.error(
                            f"Error processing task (type: {task_type}, payload snippet: {str(payload)[:200]}...): {e_task}",
                            exc_info=True, extra={"trace_id": work_id_for_log})
                else:
                    logger.debug("No task in queue (timeout), continuing to wait...")

            except asyncio.CancelledError:
                logger.info("Worker loop's current iteration cancelled.")  # 루프 자체의 취소는 아님
                # 만약 루프 전체를 중단해야 한다면 여기서 break 또는 raise
            except Exception as e_inner_loop:  # brpop 또는 작업 처리 중 예외 (핸들러 내부 예외는 핸들러가 처리)
                logger.error(f"Error in inner worker loop iteration: {e_inner_loop}. Retrying iteration or continuing.",
                             exc_info=True, extra={"trace_id": work_id_for_log})
                await asyncio.sleep(5)  # 오류 발생 시 짧은 대기


    except asyncio.CancelledError:  # worker_loop 태스크 자체가 외부에서 취소될 때
        logger.info("Worker_loop task has been cancelled.")
        # 여기서 리소스 정리가 필요하면 수행 (finally로 이동 권장)
    except Exception as e_outer_loop:  # 서비스 초기화, DB 연결 등 외부 루프의 심각한 오류
        logger.critical(
            f"Critical error in outer worker_loop structure (e.g., service init, DB connect): {e_outer_loop}. Worker shutting down.",
            exc_info=True)
    finally:
        # 이 finally 블록은 worker_loop 함수의 가장 바깥쪽 try에 대한 것임.
        logger.info("Worker_loop attempting to shut down resources...")
        if pg_service and hasattr(pg_service, 'pool') and pg_service.pool:  # 연결되었는지 확인 후 닫기
            logger.info("Closing PostgreSQL connection...")
            await pg_service.close()

        # http_session은 worker_loop 함수가 종료되기 전에 닫아야 함
        if http_session and not http_session.closed:
            logger.info("Closing aiohttp.ClientSession...")
            await http_session.close()

        # redis_client와 llm_service는 main의 finally에서 닫히도록 이전 설계였으나,
        # worker_loop 내에서 생성되었으므로 여기서 닫는 것이 더 적절할 수 있음.
        if redis_client and hasattr(redis_client, 'close'):
            logger.info("Closing Redis client...")
            await redis_client.close()
        if llm_service and hasattr(llm_service, 'close'):
            logger.info("Closing LLM service...")
            await llm_service.close()

        logger.info("Worker_loop resources cleanup attempted.")


if __name__ == "__main__":
    event_loop_instance: Optional[asyncio.AbstractEventLoop] = None
    main_task_instance: Optional[asyncio.Task] = None
    try:
        # Python 3.7+ 에서는 asyncio.run(worker_loop()) 사용 권장.
        # 아래는 기존 구조를 유지하면서 변수 undefined 가능성을 줄인 형태.
        event_loop_instance = asyncio.get_event_loop()
        if event_loop_instance:  # 루프 객체가 성공적으로 얻어졌는지 확인
            main_task_instance = event_loop_instance.create_task(worker_loop())
            event_loop_instance.run_until_complete(main_task_instance)
        else:
            logger.critical("Failed to get asyncio event loop.")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received by main. Attempting graceful shutdown...")
        if main_task_instance and not main_task_instance.done():
            main_task_instance.cancel()
            if event_loop_instance and not event_loop_instance.is_closed():
                try:
                    event_loop_instance.run_until_complete(main_task_instance)
                except asyncio.CancelledError:
                    logger.info("Main task was successfully cancelled by KeyboardInterrupt in main.")
                except Exception as e_kb_shutdown:
                    logger.error(f"Exception during KeyboardInterrupt shutdown in main: {e_kb_shutdown}", exc_info=True)
    except asyncio.CancelledError:
        logger.info("Main task was cancelled externally in main.")
    except Exception as e_unhandled_main:
        logger.critical(f"Unhandled exception in main execution block: {e_unhandled_main}", exc_info=True)
    finally:
        logger.info("Main block: ChatWorker process attempting to finish.")
        # worker_loop의 finally에서 대부분의 async 리소스가 정리되지만,
        # event_loop 자체는 여기서 닫아주는 것이 좋음.
        if event_loop_instance and not event_loop_instance.is_closed():
            # 남아있는 모든 작업들을 정리하는 로직 (선택적)
            # tasks = [t for t in asyncio.all_tasks(loop=event_loop_instance) if t is not main_task_instance and not t.done()]
            # if tasks:
            #    logger.info(f"Cancelling {len(tasks)} outstanding tasks...")
            #    for task in tasks: task.cancel()
            #    event_loop_instance.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            event_loop_instance.close()
            logger.info("Asyncio event loop closed in main.")
        logger.info("Main block: ChatWorker process finished.")