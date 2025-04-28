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

# 스트리밍 응답 형식에 대한 모델 추가
class StreamStatusUpdate(BaseModel):
    """스트리밍 상태 업데이트 스키마"""
    comic_id: str = Field(..., description="관련 만화 ID")
    status: str = Field(..., description="워크플로우 상태 (e.g., processing, collecting, scraping, done, failed)")
    message: Optional[str] = Field(default=None, description="상태 관련 메시지 또는 오류 내용")
    # 필요시 추가 데이터 필드 (예: 현재 단계 이름, 결과 URL 등)
    # current_step: Optional[str] = None
    # result_url: Optional[str] = None
    # progress: Optional[float] = None # 진행률 (구현 어려움)

# 이전 StreamChunk는 LangGraph 내부 이벤트를 위한 것이었으므로,
# API 명세에 맞는 StreamStatusUpdate를 사용하거나 이름을 변경합니다.
# 여기서는 StreamStatusUpdate를 사용하겠습니다.