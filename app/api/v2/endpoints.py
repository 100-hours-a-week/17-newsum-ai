# ai/app/api/v2/endpoints.py

from fastapi import APIRouter, Depends, HTTPException, status
import uuid
import json  # Redis에 저장하기 위해 json 임포트

# 스키마 및 서비스 임포트
from .schemas import ChatRequestPayload
from app.services.database_client import DatabaseClient
from app.dependencies import get_db_client  # Redis 클라이언트 의존성 주입
from app.utils.logger import get_logger

# 라우터 및 로거 초기화
router = APIRouter()
logger = get_logger(__name__)

# --- 상수 및 설정값 ---
QUEUE_NAME = "llm_task_queue"  # 사용할 단일 큐 이름
# 채팅 작업에 대한 기본 LLM 파라미터 (워커가 사용)
DEFAULT_CHAT_PARAMS_FOR_API = {
    "max_tokens": 256,
    "temperature": 0.7,
    "use_cot": True  # 채팅 응답 시 CoT(사고 기능) 기본 사용
}


@router.post(
    "/chat",
    status_code=status.HTTP_202_ACCEPTED,
    summary="사용자 채팅 메시지 수신 (비동기 처리)",
    description="""사용자 채팅 메시지를 수신하여 SLM 비동기 처리를 위해 Redis 큐에 작업을 추가합니다.
                 **참고:** 이 API는 DB 스키마에 따라 사용자 메시지를 직접 저장하지 않습니다.
                 'chat_handler' 워커가 사용자 메시지 및 AI 응답을 DB에 저장합니다.""" # 설명 수정
)
async def receive_chat_message_for_queue(
        payload: ChatRequestPayload,
        redis_client: DatabaseClient = Depends(get_db_client),
):
    """
    사용자 채팅 메시지를 수신하고 Redis 큐에 작업을 추가합니다.
    """
    logger.info(
        f"API /api/v2/chat received (ReqID: {payload.request_id}): "
        f"Room={payload.room_id}, User={payload.user_id}. Pushing to Redis queue."
    )

    try:
        # Redis 큐에 추가할 작업 데이터 구성
        task_data = {
            "type": "process_chat",  # 워커가 이 타입으로 작업을 식별
            "payload": {
                "room_id": payload.room_id,
                "user_id_str": payload.user_id, # 명확하게 _str 추가 (DB의 INT ID와 구분)
                "user_message": payload.message,
                "request_id": payload.request_id,
                "llm_params": DEFAULT_CHAT_PARAMS_FOR_API.copy()
            }
        }

        task_json = json.dumps(task_data)  # Redis에 저장하기 위해 JSON 문자열로 변환
        queue_len = await redis_client.lpush(QUEUE_NAME, task_json)

        if queue_len is None:
            logger.error(f"Failed to add task (ReqID: {payload.request_id}) to Redis queue '{QUEUE_NAME}'.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="메시지를 처리 큐에 추가하는 데 실패했습니다."
            )

        logger.info(f"Task for ReqID {payload.request_id} added to '{QUEUE_NAME}'. Queue length: {queue_len}.")

        # 202 Accepted 응답 반환
        return {
            "message": "메시지가 성공적으로 접수되어 처리 대기 중입니다.",
            "request_id": payload.request_id
        }

    except Exception as e:
        logger.error(f"Error in API /api/v2/chat (ReqID: {payload.request_id}): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="메시지 처리 요청 중 내부 오류가 발생했습니다."
        )