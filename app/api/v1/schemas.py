# app/api/v1/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict # Dict, Any 추가

class AsyncComicRequest(BaseModel):
    """비동기 만화 생성 요청 스키마"""
    query: str = Field(..., description="만화 생성을 위한 초기 쿼리 (뉴스 주제 또는 URL)")
    # comic_id: Optional[str] = Field(default=None, description="클라이언트 제안 ID (일반적이지 않음)") # 명세에는 있지만 서버 생성 가정

class AsyncComicResponse(BaseModel):
    """비동기 만화 생성 요청 응답 스키마"""
    comic_id: Optional[str] = Field(..., description="생성된 또는 요청된 만화 ID")
    status: str = Field(..., description="현재 작업 상태 (e.g., pending, started, error)")
    message: str = Field(..., description="응답 메시지")
