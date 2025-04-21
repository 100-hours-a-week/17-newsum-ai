# api_main.py
import asyncio
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse # StreamingResponse 대신 SSE 사용
from sse_starlette.sse import EventSourceResponse # SSE 사용
import uvicorn
import logging

# 기존 LangGraph 워크플로우 임포트
from src.workflow import workflow # 컴파일된 워크플로우 객체
from src.core.utils import logger # 기존 로거 사용

# FastAPI 앱 생성
app = FastAPI(title="News Humor Workflow API")

# 워크플로우 실행 상태 추적 (간단한 예시, 실제로는 더 견고한 관리 필요)
# 여기서는 동시 실행을 제한하지 않음
# workflow_tasks = {} # 필요시 Task ID 기반 상태 추적용

async def workflow_event_generator(request: Request):
    """워크플로우를 실행하고 이벤트를 SSE 형식으로 생성하는 비동기 제너레이터"""
    logger.info("SSE connection established. Starting workflow stream...")
    initial_input = {} # 초기 입력
    run_id = None # LangGraph 실행 ID (로깅용)

    try:
        # workflow.astream()을 사용하여 이벤트 스트림 받기
        async for event in workflow.astream(initial_input, stream_mode="values"):
            # stream_mode="values"는 각 단계의 최종 상태를 받음
            # 다른 모드(예: "updates")는 변경분만 받음

            # 클라이언트 연결 끊김 확인
            if await request.is_disconnected():
                logger.warning(f"Client disconnected. Aborting workflow stream (run_id: {run_id}).")
                # TODO: 실행 중인 Pregel 태스크 취소 로직 추가 (어려울 수 있음)
                break

            # 이벤트 데이터 준비 (JSON 직렬화 가능한 형태로)
            # 상태 객체 전체 또는 필요한 부분만 직렬화
            event_data = {}
            if isinstance(event, dict):
                 # 마지막 노드 이름 찾기 (상태 업데이트를 일으킨 노드)
                 last_node = list(event.keys())[-1] if event else "unknown"
                 event_data = {
                     "node": last_node,
                     # 필요한 상태 값만 선택적으로 포함 가능
                     # "current_news_id": event.get("current_news_item", {}).get("id") if event.get("current_news_item") else None,
                     # "humor_generated": bool(event.get("humor_result")),
                     # "image_rendered": bool(event.get("image_render_result")),
                     "keys_in_state": list(event.keys()) # 현재 상태에 있는 키 목록
                 }
                 # run_id는 Config 등을 통해 얻어야 할 수 있음, 여기서는 생략

            # SSE 형식으로 데이터 전송: "data: <json_string>\n\n"
            yield json.dumps(event_data) # sse-starlette가 data: 접두사 등을 처리

        logger.info(f"Workflow stream finished successfully (run_id: {run_id}).")
        yield json.dumps({"node": "__end__", "status": "completed"}) # 종료 이벤트 전송

    except Exception as e:
        logger.error(f"Error during workflow stream (run_id: {run_id}): {e}", exc_info=True)
        # 오류 발생 시 클라이언트에게 오류 이벤트 전송
        yield json.dumps({"node": "__error__", "error": str(e)})
    finally:
        logger.info(f"Closing SSE connection (run_id: {run_id}).")


@app.get("/stream-workflow-sse")
async def stream_workflow_sse(request: Request):
    """LangGraph 워크플로우 실행 이벤트를 SSE로 스트리밍"""
    generator = workflow_event_generator(request)
    return EventSourceResponse(generator, media_type="text/event-stream")

@app.get("/")
async def read_root():
    return {"message": "News Humor Workflow API is running. Use /stream-workflow-sse to start."}

# Uvicorn으로 서버 실행 (개발용)
if __name__ == "__main__":
    logger.info("Starting API server with Uvicorn...")
    # host="0.0.0.0"으로 설정해야 외부(Docker 등)에서 접근 가능
    # reload=True는 개발 중 코드 변경 시 자동 재시작 (운영 환경에서는 False)
    uvicorn.run(app, host="0.0.0.0", port=8000)