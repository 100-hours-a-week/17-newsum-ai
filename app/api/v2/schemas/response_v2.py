# ai/app/api/v2/schemas/response_v2.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class ProcessTurnResponse(BaseModel):
    """HEMA v2 턴 처리 응답 스키마"""
    session_id: str = Field(..., description="현재 대화 세션 ID")
    response_to_user: str = Field(..., description="사용자에게 전달될 최종 응답 메시지")
    hema_update_status: Optional[str] = Field(
        default="success", 
        description="HEMA 데이터 업데이트 결과"
    )
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")
    context_summary: Optional[Dict[str, Any]] = Field(
        None, 
        description="사용된 HEMA 컨텍스트 요약"
    )
    debug_info: Optional[Dict[str, Any]] = Field(
        None, 
        description="디버깅용 추가 정보 (개발 환경에서만)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="응답 생성 시간"
    )
    
    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()},
        "json_schema_extra": {
            "example": {
                "session_id": "session_abc123",
                "response_to_user": "AI 개발자가 주인공인 웹툰 아이디어를 제안드려요...",
                "hema_update_status": "success",
                "processing_time": 2.5,
                "context_summary": {
                    "used_snippets": 3,
                    "used_ideas": 2,
                    "context_relevance": 0.85
                },
                "timestamp": "2024-03-15T10:30:00"
            }
        }
    }
