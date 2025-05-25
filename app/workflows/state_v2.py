from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List

from app.api.v2.schemas.nodes.query_intent import FrameSchema
from app.api.v2.schemas.nodes.search_frame import FrameSearchPlan
from app.api.v2.schemas.nodes.multi_search import SearchResultSchema


class QuerySection(BaseModel):
    original_query: Optional[str] = None
    category: Optional[str] = None
    refined_intent: Optional[str] = None
    frames: List[FrameSchema] = Field(default_factory=list)
    search_plan: List[FrameSearchPlan] = Field(default_factory=list)
    search_results: List[SearchResultSchema] = Field(default_factory=list)


class WorkflowState(BaseModel):
    query: QuerySection = Field(default_factory=QuerySection)

    model_config = {
        "validate_assignment": False,
        "arbitrary_types_allowed": True,
    }
