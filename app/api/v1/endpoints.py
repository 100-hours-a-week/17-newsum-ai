# app/api/v1/endpoints.py (수정 예시)
import logging
import json
import asyncio
from fastapi import APIRouter, HTTPException, Body, Request, BackgroundTasks, Path, Depends # Depends 추가
from fastapi.responses import StreamingResponse
# from sse_starlette.sse import EventSourceResponse

from .schemas import AsyncComicRequest, AsyncComicResponse
from .background_tasks import trigger_workflow_task
from app.services.database_con_client_v2 import DatabaseClientV2
# --- 의존성 주입 함수 및 타입 임포트 ---
from app.dependencies import get_compiled_workflow_app, get_db_client
# CompiledGraph 타입 필요 시 (langgraph 버전에 따라 경로 다를 수 있음)
# from langgraph.graph.graph import CompiledGraph
from langgraph.graph import StateGraph # 또는 CompiledGraph

logger = logging.getLogger(__name__)
# db_client = DatabaseClientV2() # --- 제거 ---

router = APIRouter(
    prefix="/v1",
    tags=["Comics V1"]
)

@router.post(
    "/comics",
    response_model=AsyncComicResponse,
    summary="Request Comic Generation (Async)",
    description="Accepts a query, starts the comic generation workflow in the background, and returns a comic ID.",
    status_code=202
)
async def request_comic_generation(
    request_data: AsyncComicRequest = Body(...), # 스키마 사용
    background_tasks: BackgroundTasks = BackgroundTasks(), # BackgroundTasks 주입
    # --- Depends 사용하여 의존성 주입 ---
    compiled_app: StateGraph = Depends(get_compiled_workflow_app), # 타입은 StateGraph 또는 CompiledGraph
    db_client: DatabaseClientV2 = Depends(get_db_client)
):
    logger.info(f"[/v1/comics] 비동기 생성 요청 수신: query='{request_data.query}'")
    try:
        # background_tasks.py의 trigger_workflow_task가 compiled_app, db_client를 받도록 수정 필요
        comic_id = await trigger_workflow_task(
            request_data.query,
            background_tasks,
            compiled_app, # 주입받은 객체 전달
            db_client     # 주입받은 객체 전달
        )
        logger.info(f"Workflow task started in background with comic_id: {comic_id}")

        return AsyncComicResponse(
            comic_id=comic_id,
            status="PENDING",
            message="Comic generation task accepted and started in the background."
        )
    except Exception as e:
        logger.exception(f"[/v1/comics] 요청 처리 중 예외 발생: {e}")
        return AsyncComicResponse( # 오류 응답 생성
             comic_id=locals().get('comic_id'), # comic_id가 할당되었다면 포함
             status="ERROR",
             message=f"Failed to start workflow: {str(e)}"
         )

@router.get("/comics/status/{comic_id}", tags=["Comics V1"])
async def get_comic_status_endpoint(
    comic_id: str = Path(..., description="조회할 작업의 고유 ID."), # 경로 파라미터 명시
    db_client: DatabaseClientV2 = Depends(get_db_client) # Depends 사용
):
    logger.info(f"API Request: Get status for comic_id='{comic_id}'")
    status_data = await db_client.get(f"comic_status::{comic_id}") # 예시 키

    if status_data:
        try:
            return json.loads(status_data) if isinstance(status_data, str) else status_data
        except json.JSONDecodeError:
             return {"status": "unknown", "detail": "Failed to parse status data."}
    else:
        raise HTTPException(status_code=404, detail="Comic generation task status not found.")