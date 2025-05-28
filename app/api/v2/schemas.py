# ai/app/api/v2/schemas.py

from pydantic import BaseModel, Field, HttpUrl # HttpUrl 추가
import uuid
from typing import Optional # Optional 추가

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
    user_id: str = Field(
        ...,
        description="채팅을 입력한 사용자의 고유 ID (예: 닉네임 또는 외부 서비스 ID). DB의 ai_test_users.user_id (INT)와는 다름.",
        example="user_streamlit_app_01" # 예시 수정: Streamlit 앱에서 user_id (INT)를 문자열로 보냄
    )
    room_id: str = Field(
        ...,
        description="대화가 이루어지는 채팅방의 고유 ID. DB의 ai_test_chatrooms.room_id (INT)를 문자열로 받음.",
        example="1" # 예시 일관성
    )
    request_id: str = Field(
        ...,
        description="이 특정 요청을 식별하기 위한 클라이언트 측의 고유 ID (UUID 권장)",
        example=str(uuid.uuid4())
    )
    callback_url: Optional[HttpUrl] = Field(  # <--- 추가된 필드
        None, # 필수는 아니지만, 비동기 응답을 위해 강력히 권장
        description="AI 처리 완료 후 응답을 POST할 콜백 URL",
        example="http://localhost:9090/receive_response"
    )
    target_status: Optional[str] = Field( # <--- 추가된 필드
        None,
        description="사용자가 요청한 워크플로우 목표 상태 (예: 'IDEA_DONE')",
        example="IDEA_DONE"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "다음 아이디어를 생성해줘.",
                "user_id": "1", # Streamlit의 user_id (int)를 문자열화
                "room_id": "1",
                "request_id": str(uuid.uuid4()),
                "callback_url": "http://localhost:9090/receive_response",
                "target_status": "IDEA_DONE"
            }
        }

class AsyncComicResponse(BaseModel):
    comic_id: str
    status: str
    message: str
