# ai/app/api/v2/background_tasks.py (수정 예시)
import uuid
from typing import Dict, Any, Optional
from app.utils.logger import get_logger
from app.config.settings import Settings  # Settings 임포트
from app.api.v2.schemas import ChatResponse

# 서비스 및 워크플로우 구성 요소 타입 임포트
from app.services.postgresql_service import PostgreSQLService
from app.services.llm_service import LLMService
from app.services.database_client import DatabaseClient
from app.tools.search.Google_Search_tool import GoogleSearchTool  # 임포트 추가
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver  # 체크포인터 예시
# WorkflowControllerV3 임포트 경로 수정
from app.workflows.workflow_controller import WorkflowControllerV3
# ChatRequestPayload 임포트
from app.services.backend_client import BackendApiClient
from app.api.v2.schemas import ChatRequestPayload

logger = get_logger(__name__)


async def run_workflow_step_in_background(
        user_id: str,
        initial_work_id_str: Optional[str],
        chat_message: Optional[str],
        controller_payload: Dict[str, Any],
        pg_service: PostgreSQLService,
        llm_service: LLMService,
        redis_client: DatabaseClient,
        backend_client: BackendApiClient,
        google_search_tool: GoogleSearchTool,
        settings_obj: Settings,
        compiled_graph_from_lifespan: StateGraph,
        checkpointer_from_lifespan: MemorySaver
):
    target_status = controller_payload.get("target_status")
    request_id_from_payload = controller_payload.get("request_id", "N/A_BG_TASK")
    room_id_from_payload = controller_payload.get("room_id")

    log_work_id = initial_work_id_str or "NEW_WORKFLOW_PENDING"
    extra_log = {
        "work_id_initial": log_work_id, "target_status": target_status,
        "bg_task_id": request_id_from_payload, "user_id": user_id,
    }
    logger.info(f"백그라운드 작업 시작: 사용자 '{user_id}', 목표 상태 '{target_status}'.", extra=extra_log)

    controller = WorkflowControllerV3(
        pg_service=pg_service,
        backend_client=backend_client,
        compiled_graph=compiled_graph_from_lifespan,
        checkpointer=checkpointer_from_lifespan
    )

    payload_for_controller = ChatRequestPayload(
        request_id=request_id_from_payload,
        room_id=str(room_id_from_payload),
        user_id=user_id,
        message=chat_message,
        target_status=target_status,
        work_id=initial_work_id_str
    )

    try:
        # 1. 컨트롤러 실행 결과를 chat_response 변수에 저장
        chat_response: Optional[ChatResponse] = await controller.process_chat_interaction(
            payload=payload_for_controller)

        # 2. chat_response에 보낼 메시지가 있는지 확인 후 콜백 전송
        if chat_response and chat_response.message:
            callback_id_to_send = chat_response.request_id
            try:
                # request_id를 UUID 객체로 변환
                callback_id_uuid = uuid.UUID(chat_response.request_id)
                callback_id_to_send = callback_id_uuid
            except (ValueError, TypeError):
                logger.warning(f"콜백 request_id '{chat_response.request_id}'가 유효한 UUID가 아닙니다.", extra=extra_log)

            try:
                logger.info(f"백그라운드 작업 결과 콜백 전송 시도. ID: {callback_id_to_send}, Msg: '{chat_response.message[:100]}...'",
                            extra=extra_log)

                # backend_client를 사용하여 직접 응답 메시지 전송
                callback_success = await backend_client.streamlit_send_ai_response(
                    request_id=callback_id_to_send,
                    content=chat_response.message
                )

                if callback_success:
                    logger.info(f"백그라운드 작업 콜백 전송 성공 (ReqID: {request_id_from_payload}).", extra=extra_log)
                else:
                    # PG 저장은 컨트롤러의 fallback 로직이 아닌, 여기서 직접 처리하거나 별도 정책 수립 가능
                    logger.error(f"백그라운드 작업 콜백 전송 실패 (False 반환) (ReqID: {request_id_from_payload}).", extra=extra_log)

            except Exception as e_callback:
                logger.error(f"백그라운드 작업 콜백 전송 중 예외 발생 (ReqID: {request_id_from_payload}): {e_callback}", exc_info=True,
                             extra=extra_log)

        logger.info(f"백그라운드 작업 성공적으로 완료됨 (ReqID: {request_id_from_payload}).", extra=extra_log)
        # ▲▲▲ 핵심 변경 부분 ▲▲▲

    except Exception as e:
        logger.critical(
            f"백그라운드 작업 실행 중 심각한 예외 발생 (ReqID: {request_id_from_payload}): {e}",
            exc_info=True, extra=extra_log,
        )
