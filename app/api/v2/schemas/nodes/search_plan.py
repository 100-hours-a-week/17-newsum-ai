from pydantic import BaseModel, Field
from typing import List, Literal

class FrameSearchPlan(BaseModel):
    index: int
    title: str
    purpose: Literal["explanation", "conflict", "punchline"]
    queries: List[str] = Field(..., min_items=1)
    tool: Literal["news", "site", "youtube", "blog", "web"] #, "community"]
    domains: List[str] = Field(..., min_items=1)
    max_results: int = 5

class SearchPlanBatch(BaseModel):
    plans: List[FrameSearchPlan] = Field(..., min_items=1)
