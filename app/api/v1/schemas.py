# app/api/v1/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ComicRequest(BaseModel):
    """만화 생성 요청 스키마"""
    query: str = Field(..., description="만화 생성을 위한 초기 쿼리 (뉴스 주제 또는 URL)")
    # 필요시 다른 옵션 추가 (예: 스타일, 언어 등)
    # user_id: Optional[str] = None

class ComicResponse(BaseModel):
    """만화 생성 결과 응답 스키마 (동기식 또는 최종 결과용)"""
    success: bool = Field(..., description="성공 여부")
    message: Optional[str] = Field(default=None, description="메시지 또는 오류 내용")
    # 최종 결과 데이터 포함 (예시, AppState 전체 또는 일부)
    result: Optional[Dict[str, Any]] = Field(default=None, description="워크플로우 최종 상태 또는 결과 데이터")
    # 또는 comic_id: Optional[str] = Field(default=None, description="생성된 만화 ID (백그라운드 처리 시)")

class StreamChunk(BaseModel):
    """스트리밍 응답의 각 조각 스키마"""
    event_type: str = Field(..., description="이벤트 타입 (예: node_start, node_end, error, final_result)")
    data: Optional[Dict[str, Any]] = Field(default=None, description="이벤트 관련 데이터 (예: 현재 상태)")
    message: Optional[str] = Field(default=None, description="메시지")