# app/api/v1/background_tasks.py

# ComicState에 request.query 기반 초기 데이터 추가 필요

import uuid
from fastapi import BackgroundTasks
from app.workflows.main_workflow import build_main_workflow
from app.workflows.state import ComicState
from app.services.database_client import DatabaseClient

db_client = DatabaseClient()

async def trigger_workflow_task(request, background_tasks: BackgroundTasks) -> str:
    """
    백그라운드로 LangGraph 워크플로우 실행을 트리거하는 함수.
    """
    comic_id = str(uuid.uuid4())

    async def workflow_runner():
        graph = build_main_workflow()
        initial_state = ComicState(news_urls=[], articles=[], summaries=[], humor_texts=[], scenarios=[])
        await graph.astart(initial_state)
        await db_client.set(comic_id, "completed")

    background_tasks.add_task(workflow_runner)

    return comic_id
