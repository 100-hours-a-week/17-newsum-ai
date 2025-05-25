from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal


# ───── 모델 정의 ─────
class FrameSearchPlan(BaseModel):
    index: int = Field(..., description="원본 프레임의 인덱스")
    title: str = Field(..., description="프레임 제목")
    purpose: Literal["explanation", "conflict", "punchline"] = Field(..., description="프레임 목적")
    queries: List[str] = Field(..., min_items=1, description="검색 키워드 확장 결과")
    domains: List[str] = Field(..., min_items=1, description="검색 대상 도메인 목록 (preferred_sources)")
    tool: Literal["news", "web", "site", "community", "youtube"] = Field(..., description="검색에 사용할 도구 유형")
    lang: Literal["ko", "en", "auto"] = Field("auto", description="검색 키워드 주요 언어")
    max_results: int = Field(5, ge=1, le=50, description="프레임당 최대 검색 결과 수")


# ───── 검증 함수 ─────
def validate_search_plan(data: dict) -> FrameSearchPlan:
    """
    주어진 dict를 FrameSearchPlan으로 검증 후 리턴.
    실패 시 ValidationError가 raise됨.
    """
    return FrameSearchPlan(**data)

__all__ = [
    "FrameSearchPlan",
    "validate_search_plan",
    "ValidationError",
]
