# app/api/v1/background_tasks.py
import logging
import uuid
from fastapi import BackgroundTasks

# 필요한 import 경로 확인 필요
from app.workflows.main_workflow import build_main_workflow # 또는 compiled_workflow 사용 고려
from app.workflows.state import ComicState
from app.services.database_client import DatabaseClient # DB 클라이언트 구현 필요

logger = logging.getLogger(__name__)
db_client = DatabaseClient() # DB 클라이언트 인스턴스화 (위치 고려 필요)

async def trigger_workflow_task(query: str, background_tasks: BackgroundTasks) -> str:
    """
    백그라운드로 LangGraph 워크플로우 실행을 트리거하고 DB 상태를 업데이트하는 함수.
    """
    comic_id = str(uuid.uuid4())
    logger.info(f"Triggering background workflow for query: '{query}' with comic_id: {comic_id}")

    async def workflow_runner(comic_id: str, initial_query: str):
        graph = build_main_workflow() # 또는 compiled_workflow 사용
        initial_state_obj = ComicState(initial_query=initial_query) # Query 포함하여 초기화
        initial_state_dict = initial_state_obj.model_dump() # .dict() 는 구버전

        try:
            # 초기 상태 DB 기록 (상태: PENDING 또는 STARTED)
            await db_client.set(comic_id, {"status": "STARTED", "message": "Workflow started.", "query": initial_query})
            logger.info(f"Workflow {comic_id} started.")

            # 워크플로우 실행 (ainvoke 또는 astart 사용)
            # config = {"configurable": {"thread_id": comic_id}} # 체크포인트 사용 시 ID 전달
            # final_state_dict = await graph.ainvoke(initial_state, config=config)
            # 여기서는 간단히 invoke만 가정 (실제로는 상태 업데이트 콜백 필요)
            final_state_dict = await graph.ainvoke(initial_state_dict)

            # 최종 상태 DB 기록
            final_status = "DONE" if not final_state_dict.get("error_message") else "FAILED"
            final_message = final_state_dict.get("error_message", "Workflow completed successfully.")
            # 결과 저장 (예: final_comic_url 등)
            result_data = {"final_comic_url": final_state_dict.get("final_comic_url")} # 예시

            await db_client.set(comic_id, {"status": final_status, "message": final_message, "result": result_data})
            logger.info(f"Workflow {comic_id} finished with status: {final_status}")

        except Exception as e:
            logger.exception(f"Workflow {comic_id} failed during execution: {e}")
            # 오류 상태 DB 기록
            await db_client.set(comic_id, {"status": "FAILED", "message": f"Workflow execution error: {str(e)}"})

    # 백그라운드 작업 스케줄링
    background_tasks.add_task(workflow_runner, comic_id, query)

    # 초기 상태 (PENDING) DB 기록 (선택 사항, runner 시작 시 STARTED로 덮어쓰기 가능)
    await db_client.set(comic_id, {"status": "PENDING", "message": "Workflow task accepted."})

    return comic_id