# app/api/v1/endpoints.py
import logging
import json
import asyncio
from fastapi import APIRouter, HTTPException, Body, Request, BackgroundTasks, Path # Path 추가
from fastapi.responses import StreamingResponse
# from sse_starlette.sse import EventSourceResponse # SSE 라이브러리 사용 고려

# 워크플로우 관련 (background_tasks에서 호출되므로 직접 import 필요 없을 수 있음)
# from app.workflows.state import ComicState
# from app.workflows.main_workflow import compiled_workflow

# API 스키마 및 백그라운드 작업 함수 import
from .schemas import AsyncComicRequest, AsyncComicResponse
from .background_tasks import trigger_workflow_task
# DB 클라이언트 (스트리밍 시 상태 조회용)
from app.services.database_con_client_v2 import DatabaseClientV2

logger = logging.getLogger(__name__)
db_client = DatabaseClientV2() # DB 클라이언트 인스턴스화 (위치 고려 필요)

router = APIRouter(
    prefix="/v1",
    tags=["Comics V1"]
)

@router.post(
    "/comics",
    response_model=AsyncComicResponse, # 수정된 응답 모델
    summary="Request Comic Generation (Async)",
    description="Accepts a query, starts the comic generation workflow in the background, and returns a comic ID.",
    status_code=202 # 202 Accepted 상태 코드 사용 권장
)
async def request_comic_generation(
    request: AsyncComicRequest = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks() # BackgroundTasks 주입
):
    """
    만화 생성 워크플로우를 백그라운드에서 시작하고 즉시 comic_id를 반환합니다.
    """
    logger.info(f"[/v1/comics] 비동기 생성 요청 수신: query='{request.query}'")
    try:
        # 백그라운드 작업 트리거 함수 호출
        comic_id = await trigger_workflow_task(request.query, background_tasks)
        logger.info(f"Workflow task started in background with comic_id: {comic_id}")

        return AsyncComicResponse(
            comic_id=comic_id,
            status="PENDING", # 초기 상태 PENDING 또는 STARTED
            message="Comic generation task accepted and started in the background."
        )
    except Exception as e:
        logger.exception(f"[/v1/comics] 요청 처리 중 예외 발생: {e}")
        # 오류 발생 시 응답 형식 준수
        return AsyncComicResponse(
             comic_id=None, # ID 생성 전 실패 시 None
             status="ERROR",
             message=f"Failed to start workflow: {str(e)}"
         ) # 500 대신 오류 응답 객체 반환 고려
