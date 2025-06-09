# ai/app/api/v2/endpoints.py
from cryptography.hazmat.backends.openssl import backend
from fastapi import (
    APIRouter,
    HTTPException,
    status,
    Body,
    BackgroundTasks,
    Depends  # Depends 추가
)
import json, aiohttp

# 스키마, 로거, 의존성 임포트
from .schemas import ChatRequestPayload, ChatResponse, VALID_TARGET_STATUSES  # ChatResponse, VALID_TARGET_STATUSES 추가
from app.utils.logger import get_logger
from app.dependencies import (
    PostgreSQLServiceDep,
    CompiledWorkflowDep,
    LLMServiceDep,
    DatabaseClientDep,
    GoogleSearchToolDep,  # GoogleSearchTool 의존성 추가
    SettingsDep,  # Settings 의존성 추가
    CheckpointSaverDep  # Checkpointer 의존성 추가
)
# background_tasks.py에서 사용할 실제 서비스 타입 (FastAPI 의존성과는 별개로, 타입 힌트용)
from app.services.postgresql_service import PostgreSQLService
from app.services.llm_service import LLMService
from app.services.database_client import DatabaseClient
from app.services.backend_client import BackendApiClient
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.config.settings import Settings
from langgraph.graph import StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

# 백그라운드 태스크 함수 임포트
from .background_tasks import run_workflow_step_in_background

router = APIRouter()
logger = get_logger(__name__)

# QUEUE_NAME 및 DEFAULT_CHAT_PARAMS_FOR_API는 background_tasks.py 또는 워커에서 사용될 수 있으므로,
# 여기서는 직접 사용하지 않는다면 제거해도 무방합니다. 만약 여기서 Redis 큐에 직접 넣는 로직이 있다면 유지합니다.
# 현재 로직은 target_status가 "CHAT"일 때 Redis 큐에 넣으므로 db_client는 필요합니다.
QUEUE_NAME = "chat_task_queue"
DEFAULT_CHAT_PARAMS_FOR_API = {"max_tokens": 1024, "temperature": 0.4, "use_cot": False}


@router.post(
    "/chat/workflow",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ChatResponse,  # 응답 스키마 지정
    summary="채팅 메시지 수신 및 워크플로우 단계 실행 트리거 (비동기)",
    description="""사용자 메시지를 받아, payload의 target_status에 따라 워크플로우 단계를 백그라운드에서 실행하거나,
                 target_status가 'CHAT'인 경우 메시지를 Redis 큐에 추가합니다."""
)
async def handle_chat_and_workflow(
        background_tasks: BackgroundTasks, # FastAPI 자체 제공
        # --- 의존성 주입 (Annotated 타입만 명시, = Depends() 제거) ---
        pg_service: PostgreSQLServiceDep,
        compiled_app: CompiledWorkflowDep,
        llm_service: LLMServiceDep,
        redis_client: DatabaseClientDep,
        google_search_tool: GoogleSearchToolDep,
        settings_obj: SettingsDep,
        checkpointer: CheckpointSaverDep,
        payload: ChatRequestPayload = Body(...),  # FastAPI 자체 제공
):
    logger.info(payload)
    request_id = payload.request_id
    room_id_str = payload.room_id
    user_id = payload.user_id
    message_content = payload.message
    target_status_key = payload.target_status
    work_id_from_payload = payload.work_id

    extra_log = {
        "request_id": request_id, "room_id": room_id_str, "user_id": user_id,
        "target_status": target_status_key, "work_id_input": work_id_from_payload
    }
    logger.info(f"API /chat/workflow 수신: Target='{target_status_key}', Msg='{message_content or '[메시지 없음]'}'.",
                extra=extra_log)
    api_response_message = "요청이 접수되었습니다."

    try:
        # target_status 유효성 검사 (선택적이지만 권장)
        if target_status_key not in VALID_TARGET_STATUSES:
            logger.warning(f"유효하지 않은 target_status 값 수신: {target_status_key}", extra=extra_log)
            # raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid target_status: {target_status_key}")
            # 또는, 컨트롤러에서 이 값을 처리하도록 그대로 전달할 수도 있음. 여기서는 경고만 로깅.

        if target_status_key.upper() != "CHAT":
            logger.info(f"워크플로우 실행 요청 감지: Target='{target_status_key}'. 백그라운드 작업 스케줄링.",
                        extra=extra_log)

            # INITIAL_PROCESSING_UPTO_N03와 같은 초기화 상태 값은 main_workflow.py의 V3_TARGET_MAP에도 정의되어 있어야 함
            # 또는 schemas.py의 VALID_TARGET_STATUSES를 유일한 기준으로 사용
            is_initial_step_target = target_status_key in [
                "INITIAL_PROCESSING_UPTO_N03",
                # V3_TARGET_MAP.get("TOPIC_CLARIFICATION_N01") # 만약 V3_TARGET_MAP 키를 직접 사용한다면
            ]
            if not work_id_from_payload and not is_initial_step_target:
                logger.error(f"잘못된 요청: 초기 단계가 아닌데 work_id가 없습니다 (target_status: {target_status_key}).",
                             extra=extra_log)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"기존 워크플로우를 진행하려면 work_id가 필요합니다 (target_status: {target_status_key})."
                )

            http_session = aiohttp.ClientSession()  # async with 대신 try-finally로 세션 관리
            backend_client = BackendApiClient(http_session)
            # background_tasks.py의 run_workflow_step_in_background 함수 시그니처에 맞춰 모든 의존성 전달
            background_tasks.add_task(
                run_workflow_step_in_background,
                user_id=user_id,
                initial_work_id_str=work_id_from_payload,
                chat_message=message_content,
                controller_payload={  # ChatRequestPayload의 주요 정보 전달
                    "target_status": target_status_key,
                    "room_id": room_id_str,
                    "request_id": request_id,
                    # work_id와 message는 run_workflow_step_in_background의 별도 인자로 전달되므로 여기서는 생략 가능
                },
                # 서비스 및 주요 객체 전달
                pg_service=pg_service,
                llm_service=llm_service,
                redis_client=redis_client,  # DatabaseClientDep이 실제로는 Redis 클라이언트
                google_search_tool=google_search_tool,
                settings_obj=settings_obj,
                compiled_graph_from_lifespan=compiled_app,  # CompiledWorkflowDep
                checkpointer_from_lifespan=checkpointer,
                backend_client=backend_client,
            )
            logger.info(f"워크플로우 단계 '{target_status_key}' 실행을 위한 백그라운드 작업 예약됨 (ReqID: {request_id}).",
                        extra=extra_log)

            if message_content:
                api_response_message += f" '{target_status_key}' 단계가 백그라운드에서 처리됩니다."
            else:
                api_response_message = f"워크플로우 단계 '{target_status_key}' 실행 요청이 백그라운드에서 처리됩니다."

        # target_status가 "CHAT"인 경우 (단순 채팅 메시지 처리)
        else:
            if message_content:
                task_data_for_redis = {
                    "type": "process_chat",
                    "payload": {
                        "room_id": room_id_str,
                        "user_id_str": user_id,
                        "user_message": message_content,
                        "request_id": request_id,
                        "target_status": target_status_key,
                        "work_id": work_id_from_payload,
                        "llm_params": DEFAULT_CHAT_PARAMS_FOR_API.copy()
                    }
                }
                task_json_to_redis = json.dumps(task_data_for_redis)
                # DatabaseClientDep (redis_client) 사용
                await redis_client.lpush(QUEUE_NAME, task_json_to_redis)
                logger.info(f"사용자 메시지를 Redis 큐 '{QUEUE_NAME}'에 추가했습니다.", extra=extra_log)
                api_response_message = "채팅 메시지가 처리 대기열에 추가되었습니다."
            else:  # CHAT인데 메시지가 없는 경우
                logger.info("Target_status가 CHAT이지만 메시지 내용이 없어 Redis 큐에 추가하지 않습니다.", extra=extra_log)
                api_response_message = "메시지 내용이 없어 처리할 수 없습니다 (target_status: CHAT)."
                # 이 경우 202 대신 400을 반환할 수도 있음
                # raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message content is required for CHAT target_status.")

        return ChatResponse(
            message=api_response_message,
            request_id=request_id,
            work_id=work_id_from_payload,
            current_overall_status="ACCEPTED"  # 작업이 수락되었음을 명시
        )

    except HTTPException as http_exc:
        # 이미 HTTPException인 경우 그대로 다시 발생시켜 FastAPI가 처리하도록 함
        raise http_exc
    except Exception as e:
        logger.error(f"API /chat/workflow 처리 중 예상치 못한 오류 발생 (ReqID: {request_id}): {e}", exc_info=True, extra=extra_log)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"요청 처리 중 내부 서버 오류가 발생했습니다."  # 상세 오류는 로그에만 남김
        )

