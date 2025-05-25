from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class QuerySection(BaseModel):
    original_query: Optional[str] = None
    category: Optional[str] = None
    refined_intent: Optional[str] = None
    frames: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

class SearchSection(BaseModel):
    search_plan: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    search_results: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    user_feedback: Optional[str] = None   # 🔹 사용자 수정·추가 요청

class WorkflowState(BaseModel):
    query: QuerySection = Field(default_factory=QuerySection)
    search: SearchSection = Field(default_factory=SearchSection)

    model_config = {
        "validate_assignment": False,
        "arbitrary_types_allowed": True,
    }
