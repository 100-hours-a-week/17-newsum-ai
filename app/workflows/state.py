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

    # --- 기타 필드 (필요시 추가) ---
    # analyzed_data: Optional[Any] = Field(None, description="검색 결과 분석/종합 데이터")
    # evaluation_metrics: Optional[Dict[str, float]] = Field(None, description="중간 결과 평가 지표")

    class Config:
        arbitrary_types_allowed = True