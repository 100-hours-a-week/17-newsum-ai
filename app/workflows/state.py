# ai/app/workflows/state.py (수정)
from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel, Field

class WorkflowState(BaseModel):
    """(업그레이드됨) 워크플로우 상태 정의 (N06 필드 추가)"""

    # --- 메타데이터 (기존과 동일) ---
    trace_id: Optional[str] = Field(None, description="실행 추적 ID")
    comic_id: Optional[str] = Field(None, description="콘텐츠 고유 ID")
    timestamp: Optional[str] = Field(None, description="워크플로우 시작 시간 (ISO)")
    current_stage: Optional[str] = Field(None, description="현재 실행 중인 노드 이름")
    retry_count: int = Field(0, description="현재 노드 재시도 횟수")
    error_log: List[Dict[str, Any]] = Field(default_factory=list, description="오류 발생 기록")
    config: Dict[str, Any] = Field(default_factory=dict, description="워크플로우 설정")
    original_query: Optional[str] = Field(None, description="사용자의 원본 입력 쿼리")
    error_message: Optional[str] = Field(None, description="최상위 워크플로우 오류 메시지") # 오류 처리용 필드

    # --- Node 2 (Analyze Query) (기존과 동일) ---
    query_context: Dict[str, Any] = Field(default_factory=dict, description="쿼리 분석 결과")
    initial_context_results: List[Dict[str, Any]] = Field(default_factory=list, description="초기 컨텍스트 검색 결과")

    # --- Node 3 (Understand & Plan) (기존과 동일) ---
    search_strategy: Optional[Dict[str, Any]] = Field(None, description="수립된 최종 검색 전략")

    # --- Node 4 (Execute Search) (기존과 동일) ---
    raw_search_results: Optional[List[Dict[str, Any]]] = Field(None, description="실제 검색 실행 결과 (원시)")

    # --- Node 5 (Report Generation) (기존과 동일) ---
    report_content: Optional[str] = Field(None, description="생성된 보고서 내용 (HTML)")

    # --- Node 6 (Save Report) 추가 ---
    saved_report_path: Optional[str] = Field(None, description="로컬 파일 시스템에 저장된 보고서 파일 경로")

    # --- Node 7 (Ideation) 추가 ---
    comic_ideas: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="생성된 만화 아이디어 목록")

    # --- Node 8 (Scenario Generation) 추가 ---
    comic_scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="생성된 만화 시나리오 목록 (선택된 아이디어 기반)")
    selected_comic_idea_for_scenario: Optional[Dict[str, Any]] = Field(None, description="시나리오 작성을 위해 선택된 만화 아이디어")  # 선택적: 만약 하나의 아이디어만 시나리오로 만든다면

    # --- Node 9 (Image Generation) 추가 ---
    generated_comic_images: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="생성된 만화 장면 이미지 정보 목록 (경로/URL 등)")

    # --- Node 10 (Finalize & Notify) 추가 ---
    uploaded_image_urls: Optional[List[Dict[str, Optional[str]]]] = Field(default_factory=list, description="S3에 업로드된 이미지 URL 또는 URI 목록")  # 예: [{"scene_identifier": "Scene 1", "s3_url": "...", "error": null}]
    translated_report_content: Optional[str] = Field(None, description="번역된 보고서 내용 (HTML)")
    referenced_urls: Optional[List[str]] = Field(default_factory=list, description="보고서 생성 시 참조된 외부 URL 목록")
    external_api_response: Optional[Dict[str, Any]] = Field(None, description="외부 API 호출 결과")  # 성공/실패 정보 등

    class Config:
        arbitrary_types_allowed = True