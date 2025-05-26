# ai/app/workers/llm_worker.py

import asyncio
import json
import sys

# --- 필요한 서비스 및 모듈 임포트 ---
try:
    from app.services.database_client import DatabaseClient
    from app.services.llm_service import LLMService
    from app.services.postgresql_service import PostgreSQLService
    from app.utils.logger import get_logger

    # --- 핸들러 함수 임포트 ---
    from .handlers.chat_handler import handle_chat_processing
    from .handlers.summary_handler import handle_summary_update
except ImportError as e:
    print(f"Error importing modules or handlers in llm_worker.py: {e}")
    print("Ensure all service files and handler files exist and paths are correct.")
    sys.exit(1)

# --- 로거 초기화 ---
logger = get_logger("LLMWorkerDispatcher")

# --- 상수 정의 ---
QUEUE_NAME = "llm_task_queue"  # 작업이 들어올 단일 Redis 큐

# --- 작업 유형과 핸들러 함수 매핑 ---
# 각 핸들러는 (payload: dict, llm: LLMService, pg: PostgreSQLService, redis: DatabaseClient)
# 시그니처를 따르거나, 필요한 서비스만 받도록 선택적으로 구성 가능합니다.
TASK_HANDLERS = {
    "process_chat": handle_chat_processing,
    "update_summary": handle_summary_update,
    # "another_task_type": handle_another_task, # 새로운 기능 추가 시 여기에 등록
}


# --- 메인 워커 루프 ---
async def worker_loop():
    logger.info("Initializing services for LLM Worker Dispatcher...")
    redis_client = DatabaseClient()
    llm_service = LLMService()
    pg_service = PostgreSQLService()  # SSH 사용 여부는 PostgreSQLService 내부 로직 따름

    try:
        await pg_service.connect()
        logger.info("PostgreSQL connection established for worker.")
        logger.info(f"LLM Worker Dispatcher started... Waiting for tasks in queue: '{QUEUE_NAME}'")

        while True:
            task_data_raw = None
            try:
                task_data_raw = await redis_client.brpop(QUEUE_NAME, timeout=60)

                # (디버깅용 print문은 필요시 유지 또는 제거)
                # print(f"DEBUG PRINT: task_data_raw IS: {task_data_raw!r}")
                # print(f"DEBUG PRINT: type of task_data_raw IS: {type(task_data_raw)}")

                if task_data_raw:
                    # print(f"DEBUG PRINT: INSIDE IF - task_data_raw IS: {task_data_raw!r}")

                    _queue_name_from_redis, task_json_string = task_data_raw
                    logger.debug(
                        f"Dispatcher: Received raw task from '{_queue_name_from_redis}': {task_json_string[:250]}...")

                    # --- 수정: task_data 및 task_type 초기화 ---
                    task_data = None  # try 블록 진입 전 None으로 초기화
                    task_type_for_log = "unknown_before_parse"  # 로깅용 타입 초기화
                    # --- 수정 끝 ---

                    try:
                        task_data = json.loads(task_json_string)
                        # task_data가 성공적으로 할당된 후 task_type_for_log 업데이트
                        task_type_for_log = task_data.get("type", "unknown_after_parse_no_type_field")
                        payload = task_data.get("payload", {})

                        handler_func = TASK_HANDLERS.get(task_type_for_log)

                        if handler_func:
                            logger.info(f"Dispatcher: Dispatching task type '{task_type_for_log}' to its handler.")
                            await handler_func(payload, llm_service, pg_service, redis_client)
                        else:
                            logger.error(
                                f"Dispatcher: No handler registered for task type: '{task_type_for_log}'. Task Data: {task_data}")

                    except json.JSONDecodeError as e:
                        # JSON 파싱 실패 시 task_data는 여전히 None이거나 이전 값일 수 있으므로 task_json_string 사용
                        logger.error(f"Dispatcher: Failed to decode JSON task data: '{task_json_string}'. Error: {e}")
                    except Exception as e:
                        # 이 블록에서는 task_type_for_log를 사용 (최소한 "unknown_before_parse" 또는 파싱 후 값)
                        logger.error(
                            f"Dispatcher: Error while processing task (type: {task_type_for_log}, raw_json: '{task_json_string[:100]}...'): {e}",
                            exc_info=True)
                else:
                    logger.debug("Dispatcher: No task in queue (timeout), continuing to wait...")

            except asyncio.CancelledError:  # 외부에서 취소 요청 시
                logger.info("Dispatcher: Worker loop has been cancelled.")
                break  # 루프 종료
            except Exception as e:
                # Redis 연결 오류 등 루프 자체의 심각한 문제 발생 시
                logger.critical(f"Dispatcher: Critical error in worker loop: {e}. Retrying in 15 seconds...",
                                exc_info=True)
                await asyncio.sleep(15)  # 잠시 대기 후 재시도

    finally:
        # 워커 종료 시 모든 리소스 정리 (Graceful Shutdown)
        logger.info("Dispatcher: Shutting down. Closing worker resources...")
        if hasattr(llm_service, 'close') and callable(llm_service.close):
            await llm_service.close()
        if hasattr(redis_client, 'close') and callable(redis_client.close):
            await redis_client.close()
        if hasattr(pg_service, 'close') and callable(pg_service.close):
            await pg_service.close()
        logger.info("Dispatcher: Worker gracefully shut down.")


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
            # 취소가 완료되고 finally 블록이 실행될 시간을 줌
            loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        logger.info("Dispatcher: Main task was cancelled.")
    finally:
        # 이미 worker_loop의 finally에서 리소스 정리를 하지만,
        # 루프 외부에서 발생할 수 있는 예외에 대비해 추가적인 정리 로직이 필요하다면 여기에.
        logger.info("Dispatcher: Worker process finished.")