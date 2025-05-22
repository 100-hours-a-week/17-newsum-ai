# ai/app/workflows/state.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class WorkflowState(BaseModel):
    """뉴스 기반 만화 생성 워크플로우의 전체 상태를 관리하는 모델"""

    # --- 공통 메타데이터 ---
    trace_id: Optional[str] = Field(None, description="워크플로우 실행 추적 ID")
    comic_id: Optional[str] = Field(None, description="생성된 콘텐츠 고유 ID")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(),
                           description="워크플로우 시작 또는 갱신 시각 (ISO 형식)")
    current_stage: Optional[str] = Field(None, description="현재 실행 중인 노드 이름")
    retry_count: int = Field(0, description="현재 노드 재시도 횟수")
    error_log: List[Dict[str, Any]] = Field(default_factory=list, description="오류 발생 기록")
    config: Dict[str, Any] = Field(default_factory=dict, description="설정값 (ex: writer_id, 스타일)")
    original_query: Optional[str] = Field(None, description="사용자 최초 입력 쿼리")
    error_message: Optional[str] = Field(None, description="워크플로우 최상위 오류 메시지")

    # --- N02: 쿼리 분석 ---
    query_context: Dict[str, Any] = Field(default_factory=dict, description="쿼리 분석 결과")
    initial_context_results: List[Dict[str, Any]] = Field(default_factory=list, description="초기 문맥 검색 결과")

    # --- N03: 검색 전략 수립 ---
    search_strategy: Optional[Dict[str, Any]] = Field(None, description="검색 전략")

    # --- N04: 검색 실행 ---
    raw_search_results: Optional[List[Dict[str, Any]]] = Field(None, description="실제 검색 결과 (외부 URL 포함 가능)")

    # --- N05: 보고서 생성 및 HITL ---
    report_content: Optional[str] = Field(None, description="보고서 본문 (HTML)")
    report_full: Optional[str] = Field(None, description="보고서 전체 (HTML)")
    hitl_status: Optional[str] = Field(None, description="HITL 리뷰 상태")
    hitl_last_updated: Optional[str] = Field(None, description="HITL 마지막 업데이트 시각")
    workflow_status: Optional[str] = Field(None, description="전체 워크플로우 상태")
    hitl_feedback: Optional[str] = Field(None, description="HITL 사용자 피드백")
    hitl_revision_history: List[Dict[str, Any]] = Field(default_factory=list, description="HITL 피드백 수정 이력")
    referenced_urls_for_report: Optional[List[Dict[str, str]]] = Field(default_factory=list,
        description="보고서 작성 시 참조한 외부 기사 목록 (예: {'title': ..., 'url': ...})")

    # --- N06: 보고서 저장 ---
    saved_report_path: Optional[str] = Field(None, description="로컬 저장된 보고서 경로")
    translated_report_path: Optional[str] = Field(None, description="로컬 저장된 번역본 경로")

    # --- N06A: 보고서 문맥 기반 요약 ---
    contextual_summary: Optional[str] = Field(None, description="refined intent 기반 요약된 보고서 요지")

    # --- N07: 아이디어 생성 ---
    comic_ideas: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="생성된 만화 아이디어 목록")

    # --- N08: 시나리오 생성 ---
    selected_comic_idea_for_scenario: Optional[Dict[str, Any]] = Field(None, description="선택된 아이디어")
    comic_scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="생성된 시나리오 목록")
    thumbnail_image_prompt: Optional[str] = Field(None, description="썸네일 이미지용 프롬프트")

    # --- N09: 이미지 생성 ---
    generated_comic_images: Optional[List[Dict[str, Any]]] = Field(default_factory=list,
        description="생성된 만화 이미지 정보 (로컬 경로 포함)")

    # --- N10: 업로드 및 알림 ---
    uploaded_image_urls: Optional[List[Dict[str, Optional[str]]]] = Field(default_factory=list,
        description="S3에 업로드된 이미지 URL 목록")
    uploaded_report_s3_uri: Optional[str] = Field(None, description="업로드된 보고서 URL")
    uploaded_translated_report_s3_uri: Optional[str] = Field(None, description="업로드된 번역본 URL")
    external_api_response: Optional[Dict[str, Any]] = Field(None, description="외부 API 응답 결과 (예: 슬랙 알림 등)")

    # --- 모델 설정 (Pydantic v2 방식) ---
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }
