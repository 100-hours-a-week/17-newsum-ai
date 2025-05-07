# ai/app/api/v1/schemas.py (또는 관련 스키마 파일)

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# --- 새로운 요청 스키마 정의 ---

class SitePreferencesPayload(BaseModel):
    """요청 데이터 내 'site' 객체 스키마"""
    code_related: Optional[List[str]] = None
    research_paper: Optional[List[str]] = None
    deep_dive_tech: Optional[List[str]] = None
    # 필요에 따라 다른 카테고리 추가 가능
    community: Optional[List[str]] = None
    news: Optional[List[str]] = None

class RequestDataPayload(BaseModel):
    """요청 데이터 내 'data' 객체 스키마"""
    query: str = Field(..., description="사용자의 원본 검색 또는 생성 쿼리")
    # <<< 추가: site 필드 추가 >>>
    site: Optional[SitePreferencesPayload] = Field(None, description="사용자 지정 검색 대상 사이트 (선택 사항)")
    # 필요 시 data 객체 내 다른 필드 추가 가능
    # target_audience: Optional[str] = Field(None, description="대상 독자층 (선택 사항)")

class AsyncComicRequest(BaseModel):
    """POST /comics 엔드포인트 요청 본문 스키마 (업데이트됨)"""
    writer_id: Optional[str] = Field(None, description="사용할 AI 작가 ID (선택 사항)")
    data: RequestDataPayload = Field(..., description="쿼리 및 선택적 사이트 설정을 포함하는 객체")

# --- 응답 스키마 (변경 없음) ---

class AsyncComicResponse(BaseModel):
    comic_id: str
    status: str
    message: str

class ComicStatusResponse(BaseModel):
    # (이전 최종 버전과 동일하게 유지)
    comic_id: str
    status: str
    message: str
    query: Optional[str] = None
    writer_id: Optional[str] = None
    user_site_preferences_provided: Optional[bool] = None # DB 저장 필드 반영
    timestamp_accepted: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    duration_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None