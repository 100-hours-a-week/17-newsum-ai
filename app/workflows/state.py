# ai/app/workflows/state.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class WorkflowState(BaseModel):
    """워크플로우 상태 정의"""

    # --- 메타데이터 ---
    trace_id: Optional[str] = Field(None, description="실행 추적 ID")
    comic_id: Optional[str] = Field(None, description="콘텐츠 고유 ID")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(),
                           description="워크플로우 시작/업데이트 시간 (ISO)")
    current_stage: Optional[str] = Field(None, description="현재 실행 중인 노드 이름")
    retry_count: int = Field(0, description="현재 노드 재시도 횟수")
    error_log: List[Dict[str, Any]] = Field(default_factory=list, description="오류 발생 기록")
    config: Dict[str, Any] = Field(default_factory=dict, description="워크플로우 설정")
    original_query: Optional[str] = Field(None, description="사용자의 원본 입력 쿼리")
    error_message: Optional[str] = Field(None, description="최상위 워크플로우 오류 메시지")

    # --- Node 2 (Analyze Query) ---
    query_context: Dict[str, Any] = Field(default_factory=dict, description="쿼리 분석 결과")
    initial_context_results: List[Dict[str, Any]] = Field(default_factory=list, description="초기 컨텍스트 검색 결과")

    # --- Node 3 (Understand & Plan) ---
    search_strategy: Optional[Dict[str, Any]] = Field(None, description="수립된 최종 검색 전략")

    # --- Node 4 (Execute Search) ---
    raw_search_results: Optional[List[Dict[str, Any]]] = Field(None,
                                                               description="실제 검색 실행 결과 (원시)")  # 여기에는 외부 URL 정보 포함 가능

    # --- Node 5 (Report Generation) ---
    report_content: Optional[str] = Field(None, description="생성된 원본 보고서 내용 (HTML)")
    # Node 5에서 생성된, 보고서 작성에 참조된 외부 URL 목록 (N10의 sourceNews와는 별개로 관리)
    referenced_urls_for_report: Optional[List[Dict[str, str]]] = Field(default_factory=list,
                                                                       description="보고서 생성에 참조된 외부 소스 정보 목록 (예: {'title': '기사 제목', 'url': 'http://...'})")

    # --- Node 6 (Save Report) ---
    saved_report_path: Optional[str] = Field(None, description="로컬에 저장된 원본 보고서 파일 경로")
    translated_report_path: Optional[str] = Field(None, description="로컬에 저장된 번역본 보고서 파일 경로 (N06 생성)")

    # --- Node 7 (Ideation) ---
    comic_ideas: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="생성된 만화 아이디어 목록")

    # --- Node 8 (Scenario Generation) ---
    selected_comic_idea_for_scenario: Optional[Dict[str, Any]] = Field(None, description="시나리오 작성을 위해 선택된 만화 아이디어")
    comic_scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="생성된 만화 시나리오 목록")
    thumbnail_image_prompt: Optional[str] = Field(None, description="썸네일 이미지 생성을 위한 LLM 생성 프롬프트")

    # --- Node 9 (Image Generation) ---
    generated_comic_images: Optional[List[Dict[str, Any]]] = Field(default_factory=list,
                                                                   description="생성된 만화 장면 이미지 정보 목록 (로컬 경로 포함)")

    # --- Node 10 (Finalize & Notify) ---
    # N10에서 생성/업데이트하는 필드들
    uploaded_image_urls: Optional[List[Dict[str, Optional[str]]]] = Field(default_factory=list,
                                                                          description="S3에 업로드된 이미지 URL(HTTPS) 목록")
    uploaded_report_s3_uri: Optional[str] = Field(None, description="S3에 업로드된 원본 보고서 URL (HTTPS)")
    uploaded_translated_report_s3_uri: Optional[str] = Field(None, description="S3에 업로드된 번역본 보고서 URL (HTTPS)")

    # N10에서 폴백으로 번역을 수행했을 경우, 그 내용을 상태로 남길 필요가 있다면 이 필드 사용.
    # N06에서 이미 translated_report_path로 경로를 전달하므로, 내용 자체는 필수가 아닐 수 있음.
    # translated_report_html_content_by_n10: Optional[str] = Field(None, description="N10에서 (폴백) 생성한 번역 보고서 HTML 내용")

    external_api_response: Optional[Dict[str, Any]] = Field(None, description="외부 API 호출 결과")

    # api_payload_source_news_sent: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="외부 API로 실제 전송된 sourceNews 페이로드 (보고서 링크만 포함, 디버깅/로깅용)")

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True