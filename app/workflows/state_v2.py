# ai/app/workflows/state.py

from __future__ import annotations

from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime, timezone
from pydantic import BaseModel, Field


#  Raw / large TypedDict types
class RawArticle(TypedDict):
    url: str
    title: str
    snippet: str
    rank: int


#  Section models (BaseModel)
class MetaSection(BaseModel):
    trace_id: Optional[str] = None
    comic_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    current_stage: Optional[str] = None
    retry_count: int = 0
    error_log: List[Dict[str, Any]] = Field(default_factory=list)
    workflow_status: Optional[str] = None
    error_message: Optional[str] = None


class QuerySection(BaseModel):
    original_query: Optional[str] = None
    query_context: Dict[str, Any] = Field(default_factory=dict)
    initial_context_results: List[Dict[str, Any]] = Field(default_factory=list)


class SearchSection(BaseModel):
    search_strategy: Optional[Dict[str, Any]] = None
    raw_search_results: Optional[List[RawArticle]] = None  # TypedDict list – 검증 최소화


class ReportSection(BaseModel):
    report_content: Optional[str] = None
    report_full: Optional[str] = None
    referenced_urls_for_report: List[Dict[str, str]] = Field(default_factory=list)
    contextual_summary: Optional[str] = None
    # HITL
    hitl_status: Optional[str] = None
    hitl_last_updated: Optional[str] = None
    hitl_feedback: Optional[str] = None
    hitl_revision_history: List[Dict[str, Any]] = Field(default_factory=list)
    # 저장
    saved_report_path: Optional[str] = None
    translated_report_path: Optional[str] = None


class IdeaSection(BaseModel):
    comic_ideas: List[Dict[str, Any]] = Field(default_factory=list)


class ScenarioSection(BaseModel):
    selected_comic_idea_for_scenario: Optional[Dict[str, Any]] = None
    comic_scenarios: List[Dict[str, Any]] = Field(default_factory=list)
    thumbnail_image_prompt: Optional[str] = None
    # test
    simple_scenario: Optional[Dict[str, Any]] = None
    thumbnail_details: Optional[Dict[str, Any]] = None
    panel_details: Optional[List[Dict[str, Any]]] = None


class ImageSection(BaseModel):
    refined_prompts: List[Dict[str, Any]] = Field(default_factory=list) # 썸네일, 패널 프롬프트 모두 저장
    generated_comic_images: List[Dict[str, Any]] = Field(default_factory=list)
    # test


class UploadSection(BaseModel):
    uploaded_image_urls: List[Dict[str, Optional[str]]] = Field(default_factory=list)
    uploaded_report_s3_uri: Optional[str] = None
    uploaded_translated_report_s3_uri: Optional[str] = None
    external_api_response: Optional[Dict[str, Any]] = None


class ConfigSection(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


#  Root WorkflowState
class WorkflowState(BaseModel):
    meta: MetaSection = Field(default_factory=MetaSection)
    query: QuerySection = Field(default_factory=QuerySection)
    search: SearchSection = Field(default_factory=SearchSection)
    report: ReportSection = Field(default_factory=ReportSection)
    idea: IdeaSection = Field(default_factory=IdeaSection)
    scenario: ScenarioSection = Field(default_factory=ScenarioSection)
    image: ImageSection = Field(default_factory=ImageSection)
    upload: UploadSection = Field(default_factory=UploadSection)
    config: ConfigSection = Field(default_factory=ConfigSection)

    # 자유 사용 임시 영역 – 검증 off
    scratchpad: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "validate_assignment": False,
        "arbitrary_types_allowed": True,
    }