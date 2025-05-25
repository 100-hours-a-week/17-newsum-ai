from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional, Dict

class QuerySection(BaseModel):
    original_query: Optional[str] = None
    category: Optional[str] = None
    refined_intent: Optional[str] = None
    frames: Optional[Dict[str, list]] = Field(default_factory=dict)

class WorkflowState(BaseModel):
    # Only QueryIntent for now
    query: QuerySection = Field(default_factory=QuerySection)

    model_config = {
        "validate_assignment": False,
        "arbitrary_types_allowed": True,
    }