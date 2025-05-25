from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal

# endpoints
class QueryIntentRequest(BaseModel):
    query: str


# nodes
class FrameSchema(BaseModel):
    title: str = Field(..., description="프레임 제목")
    purpose: Literal["explanation", "conflict", "punchline"] = Field(..., description="프레임 목적")
    search_terms: List[str] = Field(..., min_items=1, description="검색어 리스트")
    preferred_sources: List[str] = Field(
        ..., min_items=1,
        description="신뢰 도메인 중 하나 이상"
    )

class QueryAnalysisSchema(BaseModel):
    category: Literal["IT", "Politics", "Economy"]
    refined_intent: str
    frames: List[FrameSchema] = Field(..., min_items=1)


def validate_query_analysis(data: dict) -> QueryAnalysisSchema:
    """
    주어진 dict를 QueryAnalysisSchema로 검증 후 리턴.
    ValidationError 발생 시 호출부에서 처리하도록 던집니다.
    """
    return QueryAnalysisSchema(**data)
