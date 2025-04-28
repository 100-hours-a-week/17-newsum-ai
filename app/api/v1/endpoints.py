# app/api/v1/endpoints.py

from fastapi import APIRouter, BackgroundTasks
from app.api.v1.schemas import ComicRequest, ComicResponse
from app.api.v1.background_tasks import trigger_workflow_task
# from app.api.v1.stream_events import comic_stream_generator
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/comics", response_model=ComicResponse)
async def create_comic(request: ComicRequest, background_tasks: BackgroundTasks):
    """
    새로운 만화 생성을 요청합니다 (비동기 워크플로우 트리거).
    """
    comic_id = await trigger_workflow_task(request, background_tasks)
    return ComicResponse(comic_id=comic_id)

# @router.post("/comics/stream")
# async def create_comic_stream(request: ComicRequest):
#     """
#     스트리밍 방식으로 만화 생성을 요청합니다 (SSE 이벤트 전송).
#     """
#     return StreamingResponse(
#         comic_stream_generator(request),
#         media_type="text/event-stream"
#     )
