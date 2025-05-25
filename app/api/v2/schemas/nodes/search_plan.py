from pydantic import BaseModel, Field
from typing import List, Literal

class FrameSearchPlan(BaseModel): # 있긴 한데 안쓰는 중
    index: int
    title: str
    purpose: Literal["explanation", "conflict", "punchline"]
    queries_ko: List[str] = Field(..., min_items=1)
    queries_en: List[str] = Field(..., min_items=1)
    tool: Literal["news", "site", "community", "youtube", "blog", "web"]
    domains: List[str] = Field(..., min_items=1)
    max_results: int = 5

class SearchPlanBatch(BaseModel):
    plans: List[FrameSearchPlan] = Field(..., min_items=1)
