# ai/app/workflows/state_v2.py

from __future__ import annotations

from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime, timezone
from pydantic import BaseModel, Field


# --- Raw / large TypedDict types ---
class RawArticle(TypedDict, total=False):
    """
    검색 결과 항목. 기존 필드 외 추가 필드를 허용하기 위해 total=False로 설정.
    필드:
      - url: str
      - title: str
      - snippet: str
      - rank: int
      - 추가적으로 query_source, tool_used, retrieved_at, source_domain, transcript, quality_score 등
    """
    url: str
    title: str
    snippet: str
    rank: int


# --- Section models (BaseModel) ---
class MetaSection(BaseModel):
    # 워크플로우 실행 및 생성되는 만화 콘텐츠의 고유 통합 ID
    work_id: Optional[str] = None

    # 워크플로우 상태가 처음 생성되거나 주요 변경이 발생한 시간 (UTC)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # 각 노드별 진행 상태: {node_order: "PROCESSING"/"COMPLETED"/"ERROR"/"WAITING"}
    workflow_status: Dict[int, str] = Field(default_factory=dict)

    # LLM의 <think> 태그 내용을 저장하기 위한 리스트
    # 각 항목: {"node_name": str, "request_id": str, "timestamp": str, "log_content": str}
    llm_think_traces: List[Dict[str, Any]] = Field(default_factory=list)

    # 컨트롤러 등이 사용할 다음 액션 정보 (optional)
    next_action: Optional[str] = None


class QuerySection(BaseModel):
    # 사용자가 입력한 원본 요청 문자열
    original_query: Optional[str] = None

    # N03에서 채운, 검색 타깃으로 사용할 시드 도메인 목록
    search_target_site_domain: Optional[List[str]] = Field(default_factory=list)

    # N02에서 초기 컨텍스트 확보를 위해 수행한 검색 결과 스니펫 리스트
    # 각 dict에 title, snippet, url, source_domain, rank, etc. 포함
    initial_context_results: List[Dict[str, Any]] = Field(default_factory=list)

    # N02~N03에서 분석/정제된 사용자 쿼리 관련 구조화 정보
    # e.g. extracted_keywords, query_type, query_category,
    # issue_summary_for_comic, satire_target, tone_suggestion, refined_intent, key_aspects_to_search, etc.
    query_context: Dict[str, Any] = Field(default_factory=dict)

    # N02 LLM 분석 과정 및 판단 근거 (한글)
    llm_analysis_details_korean: Optional[str] = None

    # N03 LLM 검색 계획 수립 관련 판단 근거 (한글)
    llm_analysis_details_searchplan_korean: Optional[str] = None


class SearchSection(BaseModel):
    # N03에서 수립된 검색 전략 (writer_concept, selected_tools, queries, target_seed_domains, tool_parameters 등)
    search_strategy: Optional[Dict[str, Any]] = None

    # N04에서 수집된 원본 검색 결과. RawArticle에 추가 필드를 허용한 Dictionary 형태.
    raw_search_results: Optional[List[Dict[str, Any]]] = None


class ReportSection(BaseModel):
    # N05에서 생성된 보고서의 HTML 콘텐츠
    report_content: Optional[str] = None

    # 보고서 생성 시 참고한 핵심 이슈 요약 (issue_summary_for_comic)
    contextual_summary: Optional[str] = None

    # 보고서 HTML 저장 경로
    saved_report_path: Optional[str] = None

    # 보고서 작성에 인용된 출처 정보: {"section_name": [RawArticle dict, ...], ...}
    # RawArticle이 total=False로 추가 필드를 허용하므로, Dict[str, Any] 형태로 저장 가능
    referenced_urls_for_report: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)

    # N05 LLM이 보고서 구조를 세운 한글 근거 (optional)
    llm_report_structuring_details_korean: Optional[str] = None


class FinalSatiricalIdea(TypedDict):
    title_concept: str
    detailed_content: str  # 만평 시각 묘사 및 대사 등
    applied_satire_techniques: List[str]
    expected_appeal_points: str
    ethical_review_notes: Optional[str]


class IdeaSection(BaseModel):
    # N07에서 생성된 만화 아이디어 리스트
    comic_ideas: List[Dict[str, Any]] = Field(default_factory=list)

    # N06에서 분석된 구조화된 이슈 분석 결과
    structured_issue_analysis: Optional[Dict[str, Any]] = None

    # 커뮤니티 반응 (예: { "reddit": [{"type": "comment", "content": "..."}], ... })
    community_reactions: Optional[Dict[str, List[Dict[str, str]]]] = None

    # 스크랩된 커뮤니티 텍스트 반응 (예: 플랫폼별 문자열 목록)
    scraped_community_reactions: Optional[Dict[str, List[str]]] = None

    # 커뮤니티 반응 분석 결과
    community_reaction_analysis: Optional[Dict[str, Dict[str, Any]]] = None

    # 생성된 풍자 반응 예시 (N06C)
    generated_satirical_reactions: Optional[Dict[str, List[Dict[str, str]]]] = None

    # N07에서 생성된 최종 만화 아이디어 목록 (FinalSatiricalIdea 구조)
    final_comic_ideas: List[FinalSatiricalIdea] = Field(default_factory=list)


class ScenarioSection(BaseModel):
    # 사용자가 선택한 아이디어 인덱스 (0부터 시작)
    selected_comic_idea_for_scenario: Optional[int] = None

    # N08에서 생성된 4컷 만화 패널별 시나리오 리스트
    comic_scenarios: List[Dict[str, Any]] = Field(default_factory=list)

    # N08 이미지 프롬프트용 원본 설명(썸네일 등)
    comic_scenario_thumbnail: Optional[str] = None


class ImageSection(BaseModel):
    # N08a에서 정제된 이미지 생성용 프롬프트 리스트
    refined_prompts: List[Dict[str, Any]] = Field(default_factory=list)

    # N09에서 생성된 만화 이미지 정보 리스트
    generated_comic_images: List[Dict[str, Any]] = Field(default_factory=list)


class UploadSection(BaseModel):
    # 외부 저장소(S3 등)에 업로드된 이미지 URL 리스트
    uploaded_image_urls: List[Dict[str, Optional[str]]] = Field(default_factory=list)

    # 외부 저장소에 업로드된 보고서 파일 URI
    uploaded_report_s3_uri: Optional[str] = None


class ConfigSection(BaseModel):
    # 워크플로우 전반 설정값: writer_id, target_audience, youtube_transcript_languages, 등
    config: Dict[str, Any] = Field(default_factory=dict)


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

    # scratchpad: 노드 실행 중 데이터 전달·임시 저장 용 임의 필드
    scratchpad: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "validate_assignment": False,
        "arbitrary_types_allowed": True,
    }
