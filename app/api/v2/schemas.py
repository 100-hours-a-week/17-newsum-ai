# ai/app/api/v2/schemas.py

from pydantic import BaseModel, Field
import uuid

class ChatRequestPayload(BaseModel):
    """
    POST /api/v2/chat 엔드포인트의 요청 페이로드 모델입니다.
    클라이언트(다른 백엔드 서버)로부터 수신하는 사용자 채팅 정보를 정의합니다.
    """
    message: str = Field(
        ...,
        description="사용자가 입력한 채팅 메시지 본문",
        example="오늘 날씨 어때?"
    )
    user_id: str = Field( # DB 스키마에 따르면 이 ID를 기반으로 ai_test_users에서 실제 user_id(INT)를 찾아야 함
        ...,
        description="채팅을 입력한 사용자의 고유 ID (예: 닉네임 또는 외부 서비스 ID)",
        example="user_nickname_01"
    )
    room_id: str = Field( # DB 스키마에 따르면 INT 타입이지만, API에서는 문자열로 받아 PG 서비스에서 변환
        ...,
        description="대화가 이루어지는 채팅방의 고유 ID (DB의 ai_test_chatrooms.room_id)",
        example="123"
    )
    request_id: str = Field(
        ...,
        description="이 특정 요청을 식별하기 위한 클라이언트 측의 고유 ID (UUID 권장)",
        example=str(uuid.uuid4())
    )

    class Config:
        # FastAPI 문서용 예시 데이터
        json_schema_extra = {
            "example": {
                "message": "오늘 주요 뉴스 좀 요약해 줄래?",
                "user_id": "user_backend_service_01",
                "room_id": "1", # 예시를 숫자로 변경
                "request_id": str(uuid.uuid4())
            }
        }