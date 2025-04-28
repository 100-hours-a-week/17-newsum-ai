# app/api/v1/background_tasks.py

# ComicState에 request.query 기반 초기 데이터 추가 필요

import uuid
from fastapi import BackgroundTasks
from app.workflows.main_workflow import build_main_workflow
from app.workflows.state import ComicState
from app.services.database_client import DatabaseClient

db_client = DatabaseClient()

# async def trigger_workflow_task(request, background_tasks: BackgroundTasks) -> str:
#     """
#     백그라운드로 LangGraph 워크플로우 실행을 트리거하는 함수.
#     """
#     comic_id = str(uuid.uuid4())
#
#     async def workflow_runner():
#         graph = build_main_workflow()
#         initial_state = ComicState(news_urls=[], articles=[], summaries=[], humor_texts=[], scenarios=[])
#         await graph.astart(initial_state)
#         await db_client.set(comic_id, "completed")
#
#     background_tasks.add_task(workflow_runner)
#
#     return comic_id
# trigger_workflow_task 수정 제안
async def trigger_workflow_task(query: str, background_tasks: BackgroundTasks) -> str: # request 대신 query 직접 받기
    """
    백그라운드로 LangGraph 워크플로우 실행을 트리거하는 함수.
    """
    comic_id = str(uuid.uuid4())

    async def workflow_runner():
        graph = build_main_workflow()
        # *** 수정된 부분: query를 사용하여 initial_state 설정 ***
        initial_state = ComicState(
            initial_query=query, # 사용자 쿼리 전달
            news_urls=[],
            articles=[],
            summaries=[],
            humor_texts=[],
            scenarios=[]
            # 다른 필드들도 필요시 초기화
        )
        # *** 초기 상태 데이터베이스 기록 (선택 사항) ***
        # await db_client.set(comic_id, {"status": "started", "state": initial_state})

        try:
            # *** 워크플로우 실행 ***
            # config={"configurable": {"thread_id": comic_id}} 와 같이 실행 ID 전달 가능 (체크포인트 사용 시)
            await graph.ainvoke(initial_state) # astart 대신 ainvoke 사용 고려 (체크포인트 불필요 시) 또는 astart(config=...)
            # *** 최종 상태 데이터베이스 기록 ***
            # final_state = ??? (ainvoke 사용 시 반환값, astart 사용 시 별도 조회 필요)
            await db_client.set(comic_id, {"status": "completed"}) # 완료 상태 및 최종 결과 저장
        except Exception as e:
            logger.error(f"Workflow {comic_id} failed: {e}")
            # *** 오류 상태 데이터베이스 기록 ***
            await db_client.set(comic_id, {"status": "failed", "error": str(e)})

    background_tasks.add_task(workflow_runner)
    # 초기 상태 저장 (선택 사항)
    # await db_client.set(comic_id, {"status": "pending"})
    return comic_id
