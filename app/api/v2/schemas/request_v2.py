# ai/app/api/v2/schemas/request_v2.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class ProcessTurnRequest(BaseModel):
    """HEMA v2 턴 처리 요청 스키마"""
    user_id: str = Field(..., description="사용자 고유 ID", min_length=1)
    session_id: str = Field(..., description="현재 대화 세션 ID", min_length=1)
    user_message: str = Field(..., description="사용자의 직전 입력 메시지", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="추가 메타데이터"
    )
    
    # 메타데이터 예시:
    # {
    #   "task_type": "아이디어_생성",
    #   "target_char_id": "char123",
    #   "client_info": {"version": "1.0", "platform": "web"},
    #   "context_preferences": {
    #     "max_snippets": 3,
    #     "focus_areas": ["character", "plot"]
    #   }
    # }
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_12345",
                "session_id": "session_abc123",
                "user_message": "웹툰 주인공이 AI 개발자인 스토리를 만들어줘",
                "metadata": {
                    "task_type": "아이디어_생성",
                    "context_preferences": {
                        "max_snippets": 5,
                        "focus_areas": ["character", "plot"]
                    }
                }
            }
        }
    }
