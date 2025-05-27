# ai/app/workers/chat_worker.py

import asyncio
import json
import sys
from pathlib import Path
import aiohttp

# --- 로깅 설정 먼저 호출 ---
try:
    from app.utils.logger import setup_logging

    setup_logging()
except ImportError as e:
    print(f"CRITICAL: Failed to import or setup logging: {e}")
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")
    logging.critical("Logging setup failed, using basic print/logging.")

# --- 필요한 서비스 및 모듈 임포트 ---
try:
    from app.services.database_client import DatabaseClient
    from app.services.llm_service import LLMService
    from app.services.postgresql_service import PostgreSQLService
    from app.utils.logger import get_logger
    from app.services.backend_client import BackendApiClient
    from app.config.settings import Settings  # Settings 임포트

    # --- 핸들러 함수 임포트 ---
    from .handlers.chat_handler import handle_chat_processing
    from .handlers.summary_handler import handle_summary_update

    # --- 토크나이저 로딩 ---
    TOKENIZER = None
    logger_init = get_logger("TokenizerInitWorker")
    try:
        import os
        from transformers import AutoTokenizer

        # __file__은 현재 실행 중인 스크립트의 경로를 나타냄
        # chat_worker.py가 있는 디렉토리의 'handlers' 하위 'qwen_tokenizer'를 찾음
        current_script_dir = Path(__file__).resolve().parent
        QWEN_TOKENIZER_PATH = current_script_dir / "handlers"
        if QWEN_TOKENIZER_PATH.is_dir():
            TOKENIZER = AutoTokenizer.from_pretrained(str(QWEN_TOKENIZER_PATH), trust_remote_code=True)
            logger_init.info(f"ChatWorker: Tokenizer loaded successfully from {QWEN_TOKENIZER_PATH}.")
        else:
            logger_init.error(
                f"ChatWorker: Tokenizer directory NOT found at {QWEN_TOKENIZER_PATH}. Please check the path.")
    except Exception as e:
        logger_init.error(f"ChatWorker: Failed to load Qwen Tokenizer: {e}", exc_info=True)

except ImportError as e:
    # 초기 임포트 실패 시 print로라도 남김
    print(f"CRITICAL Error importing modules in chat_worker.py: {e}")
    sys.exit(1)

# --- 로거 초기화 (명칭 변경) ---
logger = get_logger("ChatWorkerDispatcher")

# --- 상수 정의 (큐 이름 등 설정에서 가져오도록 고려) ---
settings_instance = Settings()  # settings 객체 생성
QUEUE_NAME = getattr(settings_instance, 'CHAT_TASK_QUEUE_NAME', "chat_task_queue")

# --- 작업 유형과 핸들러 함수 매핑 ---
TASK_HANDLERS = {
    "process_chat": handle_chat_processing,
    "update_summary": handle_summary_update,
}


# --- 메인 워커 루프 ---
async def worker_loop():
    logger.info("Initializing services for Chat Worker Dispatcher...")
    redis_client = DatabaseClient()  # settings에서 Redis 정보 사용 (DatabaseClient 내부 구현에 따라)

    if TOKENIZER is None:
        logger.error("CRITICAL: Tokenizer is not loaded. Chat Worker cannot start properly.")
        return  # 토크나이저 없으면 워커 시작 불가

    llm_service = LLMService(tokenizer=TOKENIZER)  # LLMService 초기화 시 토크나이저 전달
    pg_service = PostgreSQLService()  # settings에서 PG 정보 사용 (PostgreSQLService 내부 구현에 따라)

    async with aiohttp.ClientSession() as session:
        backend_client = BackendApiClient(session)  # settings에서 API URL 사용 (BackendApiClient 내부 구현에 따라)

        try:
            await pg_service.connect()
            logger.info("PostgreSQL connection established for ChatWorker.")
            logger.info(f"Chat Worker Dispatcher started... Waiting for tasks in queue: '{QUEUE_NAME}'")

            while True:
                task_data_raw = None
                try:
                    task_data_raw = await redis_client.brpop(QUEUE_NAME, timeout=60)

                    if task_data_raw:
                        _queue_name, task_json = task_data_raw
                        logger.debug(f"Dispatcher: Received raw task: {task_json[:250]}...")

                        task_data = None
                        task_type = "unknown_task_type"  # 기본값 설정

                        try:
                            task_data = json.loads(task_json)
                            task_type = task_data.get("type", "unknown_type_in_payload")
                            payload = task_data.get("payload", {})

                            handler_func = TASK_HANDLERS.get(task_type)

                            if handler_func:
                                logger.info(
                                    f"Dispatcher: Dispatching task '{task_type}' with request ID '{payload.get('request_id', 'N/A')}'.")
                                await handler_func(
                                    payload,
                                    llm_service,
                                    pg_service,
                                    redis_client,
                                    TOKENIZER,  # 핸들러에 토크나이저 전달
                                    backend_client  # 핸들러에 백엔드 클라이언트 전달
                                )
                            else:
                                logger.error(
                                    f"Dispatcher: No handler for task type: '{task_type}'. Task Data: {task_data}")

                        except json.JSONDecodeError as e:
                            logger.error(f"Dispatcher: Failed to decode JSON task data: '{task_json}'. Error: {e}")
                        except Exception as e:
                            logger.error(
                                f"Dispatcher: Error processing task (type: {task_type}, payload: {str(payload)[:200]}...): {e}",
                                exc_info=True)
                    else:
                        logger.debug("Dispatcher: No task in queue (timeout), continuing to wait...")

                except asyncio.CancelledError:
                    logger.info("Dispatcher: Worker loop has been cancelled.")
                    break
                except Exception as e:  # Redis 연결 오류 등 루프 자체의 심각한 문제
                    logger.critical(f"Dispatcher: Critical error in worker loop: {e}. Retrying in 15 seconds...",
                                    exc_info=True)
                    await asyncio.sleep(15)

        finally:
            logger.info("Dispatcher: Shutting down. Closing worker resources...")
            await pg_service.close()
            await redis_client.close()
            await llm_service.close()
            logger.info("Dispatcher: ChatWorker gracefully shut down.")


# --- 스크립트 실행 엔트리 포인트 ---
if __name__ == "__main__":
    main_task = None
    loop = asyncio.get_event_loop()
    try:
        main_task = loop.create_task(worker_loop())
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.info("Dispatcher: KeyboardInterrupt received. Attempting graceful shutdown...")
        if main_task:
            main_task.cancel()
            try:
                loop.run_until_complete(main_task)
            except asyncio.CancelledError:
                pass  # Main task가 취소될 때 발생하는 예외는 정상 종료로 간주
    except asyncio.CancelledError:
        logger.info("Dispatcher: Main task was cancelled externally.")
    finally:
        # 이벤트 루프가 이미 닫혔을 수 있으므로, 여기서 추가적인 async 작업은 주의
        logger.info("Dispatcher: ChatWorker process finished.")