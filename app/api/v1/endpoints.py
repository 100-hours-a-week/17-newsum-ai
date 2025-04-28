import logging
import json
import asyncio
from fastapi import APIRouter, HTTPException, Body, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse # SSE 사용 시 더 편리한 라이브러리 (선택 사항)

# 워크플로우 상태 및 컴파일된 그래프 import
from app.workflows.state import AppState
from app.workflows.main_workflow import compiled_workflow # 컴파일된 그래프 객체 (가정)

# API 스키마 import
from .schemas import ComicRequest, ComicResponse, StreamChunk

logger = logging.getLogger(__name__)

# /api/v1 경로 아래에 엔드포인트들을 정의할 라우터 생성
router = APIRouter(
    prefix="/v1",
    tags=["Comics V1"] # Swagger UI 태그
)

@router.post(
    "/comics",
    response_model=ComicResponse, # 응답 형식 지정
    summary="Generate Comic Synchronously",
    description="Receives a query, runs the full LangGraph workflow, and returns the final result.",
)
async def generate_comic(
    request: ComicRequest = Body(...) # 요청 본문을 Pydantic 모델로 받음
):
    """
    만화 생성 요청을 받아 워크플로우를 동기식으로 실행하고 최종 결과를 반환합니다.
    (주의: 워크플로우가 길면 타임아웃 발생 가능성이 있으므로 실제 서비스에서는 비동기 처리 또는 스트리밍 권장)
    """
    logger.info(f"[/v1/comics] 요청 수신: query='{request.query}'")

    # 초기 상태 설정
    initial_state = AppState(initial_query=request.query)

    try:
        # LangGraph 워크플로우 비동기 실행 (전체 완료까지 대기)
        # config={"recursion_limit": 100} 등 실행 옵션 추가 가능
        final_state = await compiled_workflow.ainvoke(initial_state)

        # 오류 발생 여부 확인 (상태 내 error_message 필드 활용)
        if final_state.get("error_message"):
            logger.error(f"워크플로우 실행 중 오류 발생: {final_state['error_message']}")
            return ComicResponse(
                success=False,
                message=final_state["error_message"],
                result=final_state # 오류 상태 포함 반환
            )

        logger.info(f"[/v1/comics] 워크플로우 실행 완료.")
        return ComicResponse(
            success=True,
            message="Comic generation workflow completed successfully.",
            result=final_state # 최종 상태 반환
        )

    except Exception as e:
        logger.exception(f"[/v1/comics] 엔드포인트 처리 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post(
    "/comics/stream",
    summary="Generate Comic with Streaming Updates",
    description="Runs the LangGraph workflow and streams updates using Server-Sent Events (SSE)."
)
async def generate_comic_stream(
    request_body: ComicRequest = Body(...)
):
    """
    만화 생성 워크플로우를 실행하고 각 단계의 진행 상황 또는 결과를 SSE로 스트리밍합니다.
    """
    logger.info(f"[/v1/comics/stream] 스트리밍 요청 수신: query='{request_body.query}'")
    initial_state = AppState(initial_query=request_body.query)

    async def event_generator():
        try:
            # LangGraph의 astream_events 사용 (더 상세한 이벤트 제공)
            # version="v1" 또는 "v2" (v2가 더 많은 메타데이터 제공)
            async for event in compiled_workflow.astream_events(initial_state, version="v2"):
                event_type = event["event"]
                event_name = event.get("name", "") # 어떤 노드/그래프 관련 이벤트인지
                event_data = event.get("data", {})

                logger.debug(f"SSE Event: type={event_type}, name={event_name}, data_keys={list(event_data.keys())}")

                chunk_data: Dict[str, Any] | None = None
                message: str | None = None

                # 관심 있는 이벤트 타입만 클라이언트에 전송
                if event_type == "on_chain_start":
                    message = f"Workflow started for node: {event_name}"
                    chunk_data = event_data.get("input") # 입력 데이터 포함 가능
                elif event_type == "on_chain_stream":
                     # astream() 사용 시 중간 결과 (청크)를 받을 수 있음
                     chunk_data = event_data.get("chunk")
                elif event_type == "on_chain_end":
                    message = f"Node finished: {event_name}"
                    chunk_data = event_data.get("output") # 노드의 최종 출력 (상태)
                elif event_type == "on_chat_model_stream":
                     # LLM 스트리밍 시 토큰별 데이터 처리
                     chunk_data = {"token": event_data.get("chunk").content}
                     message = "LLM token stream"
                elif event_type == "on_chain_error":
                    message = f"Error in node {event_name}: {event_data.get('error')}"
                    logger.error(message)
                    # 오류 발생 시 스트림 종료 또는 오류 메시지 전송 후 계속 진행 결정 필요

                # 클라이언트에 전송할 데이터 구성
                if message or chunk_data:
                    stream_chunk = StreamChunk(
                        event_type=event_type,
                        data=chunk_data,
                        message=message
                    )
                    # 직렬화 가능한 형태로 변환 (Pydantic 모델 -> dict) 후 JSON 문자열로 변환
                    yield f"data: {stream_chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.01) # 너무 빠른 전송 방지 (선택 사항)

            # 워크플로우 최종 완료 시 (선택 사항)
            # yield f"data: {json.dumps({'event_type': 'workflow_end', 'message': 'Workflow finished.'})}\n\n"

        except Exception as e:
            logger.exception(f"[/v1/comics/stream] 스트리밍 중 예외 발생: {e}")
            error_chunk = StreamChunk(
                event_type="error",
                message=f"Streaming failed: {str(e)}"
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"

    # EventSourceResponse 사용 (더 편리)
    # return EventSourceResponse(event_generator())

    # 또는 기본 StreamingResponse 사용
    return StreamingResponse(event_generator(), media_type="text/event-stream")