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
from .schemas import AsyncComicRequest, AsyncComicResponse, StreamStatusUpdate
from .background_tasks import trigger_workflow_task
# DB 클라이언트 (스트리밍 시 상태 조회용)
from app.services.database_client import DatabaseClient

logger = logging.getLogger(__name__)
db_client = DatabaseClient() # DB 클라이언트 인스턴스화 (위치 고려 필요)

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


@router.post( # API 명세는 POST 지만, GET이 더 적합할 수 있음
    "/comics/{comic_id}/stream",
    summary="Subscribe to Comic Generation Status Updates",
    description="Subscribes to Server-Sent Events (SSE) for status updates of an existing comic generation task."
)
async def stream_comic_status(
    comic_id: str = Path(..., description="The ID of the comic generation task to track."),
    # request_body: Dict = Body(None) # 명세상 body가 있지만, 여기서는 사용 안 함
):
    """
    지정된 comic_id 작업의 상태를 SSE로 스트리밍합니다.
    (주의: 이 예시는 DB 폴링 방식이며, 실제 환경에서는 Pub/Sub 등 사용 권장)
    """
    logger.info(f"[/v1/comics/{comic_id}/stream] 상태 스트리밍 구독 요청")

    async def status_event_generator(task_id: str):
        last_status = None
        error_count = 0
        max_errors = 3 # DB 조회 실패 시 최대 재시도 횟수

        while True:
            try:
                # DB에서 현재 상태 조회
                current_data = await db_client.get(task_id) # DB get 메서드 구현 필요
                error_count = 0 # 성공 시 에러 카운트 초기화

                if not current_data:
                    logger.warning(f"Stream: Task ID {task_id} not found in DB.")
                    status_update = StreamStatusUpdate(
                        comic_id=task_id, status="NOT_FOUND", message="Task ID not found."
                    )
                    yield f"data: {status_update.model_dump_json()}\n\n"
                    break # 작업 없음 종료

                current_status = current_data.get("status", "UNKNOWN")
                current_message = current_data.get("message")

                # 상태가 변경되었을 때만 클라이언트에 전송 (선택 사항)
                if current_status != last_status:
                    logger.info(f"Stream: Task {task_id} status update: {current_status}")
                    status_update = StreamStatusUpdate(
                        comic_id=task_id,
                        status=current_status,
                        message=current_message
                        # 필요한 다른 데이터 추가 (예: current_data.get('result'))
                    )
                    yield f"data: {status_update.model_dump_json()}\n\n"
                    last_status = current_status

                # 종료 상태(DONE, FAILED, NOT_FOUND 등)이면 스트림 종료
                if current_status in ["DONE", "FAILED", "NOT_FOUND", "ERROR"]:
                    logger.info(f"Stream: Task {task_id} reached terminal state: {current_status}. Closing stream.")
                    break

            except Exception as e:
                error_count += 1
                logger.exception(f"Stream: Error polling status for task {task_id}: {e} (Attempt {error_count})")
                if error_count >= max_errors:
                    status_update = StreamStatusUpdate(
                        comic_id=task_id, status="ERROR", message=f"Failed to poll status after multiple attempts: {str(e)}"
                    )
                    yield f"data: {status_update.model_dump_json()}\n\n"
                    break # 폴링 실패 시 종료
                await asyncio.sleep(2) # 오류 발생 시 잠시 대기 후 재시도
                continue # 다음 폴링 시도

            # 폴링 간격 (예: 1초)
            await asyncio.sleep(1)

    # EventSourceResponse 사용 권장
    # return EventSourceResponse(status_event_generator(comic_id))
    return StreamingResponse(status_event_generator(comic_id), media_type="text/event-stream")