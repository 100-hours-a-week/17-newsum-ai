# app/workflows/state.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Annotated
import operator

# --- 리듀서 함수 정의 (예: 문자열 연결) ---
def append_error_reducer(left: Optional[str], right: Optional[str]) -> Optional[str]:
    """오류 메시지를 안전하게 연결하는 리듀서"""
    if right: # 새로운 오류 메시지가 있는 경우에만 처리
        if left:
            return f"{left}; {right}" # 세미콜론으로 구분하여 연결
        return right # 기존 메시지가 없으면 새 메시지 사용
    return left # 새로운 메시지가 없으면 기존 메시지 유지
# -----------------------------------------

class ComicState(BaseModel):
    """
    LangGraph 워크플로우의 노드 간에 전달되는 상태를 정의합니다.
    사양 문서 및 README.md를 기반으로 합니다.
    """
    # --- 핵심 ID 및 메타데이터 ---
    comic_id: Optional[str] = Field(default=None, description="만화 생성 작업의 고유 ID.")
    trace_id: Optional[str] = Field(default=None, description="관찰 가능성(예: LangSmith)을 위한 추적 ID.")
    timestamp: Optional[str] = Field(default=None, description="워크플로우 시작 시점의 ISO 8601 타임스탬프.")
    initial_query: Optional[str] = Field(default=None, description="사용자의 초기 쿼리.")

    # --- 설정 ---
    config: Dict[str, Any] = Field(default_factory=dict, description="워크플로우 공통 설정 (예: 모델명, 기능 플래그).")

    # --- 토픽 분석 (노드 02 출력) ---
    topic_analysis: Dict[str, Any] = Field(default_factory=dict, description="초기 쿼리에 대한 구조화된 분석 (주요 토픽, 엔티티, 키워드).")
    search_keywords: List[str] = Field(default_factory=list, description="정보 수집을 위한 키워드 목록.")

    # --- 수집된 URL (노드 03 & 04 출력) ---
    fact_urls: List[Dict[str, str]] = Field(default_factory=list, description="수집된 뉴스 기사 URL 목록 [{'url': str, 'source': str, 'search_keyword': str}].")
    opinion_urls: List[Dict[str, str]] = Field(default_factory=list, description="수집된 의견 URL 목록 [{'url': str, 'source': str, 'search_keyword': str}].")

    # --- 스크랩된 콘텐츠 (노드 05 & 06 출력) ---
    articles: List[Dict[str, Any]] = Field(default_factory=list, description="스크랩된 뉴스 기사 콘텐츠 목록 [{'url': str, 'title': str, 'text': str, ...}].")
    opinions_raw: List[Dict[str, Any]] = Field(default_factory=list, description="스크랩된 원본 의견 콘텐츠 목록 [{'url': str, 'text': str, 'author': str, ...}].")

    # --- 필터링 및 클러스터링된 의견 (노드 07 출력) ---
    opinions_clean: List[Dict[str, Any]] = Field(default_factory=list, description="필터링, 중복 제거, 클러스터링된 의견 ('cluster_id', 'is_representative' 추가됨).")

    # --- 요약 (노드 08, 09, 10 출력) ---
    news_summaries: List[Dict[str, Any]] = Field(default_factory=list, description="FEQA 점수가 포함된 뉴스 요약 목록 [{'original_url': str, 'summary_text': str, 'feqa_score': float}].")
    opinion_summaries: Dict[str, Any] = Field(default_factory=dict, description="의견 요약 (스탠스 클러스터, 감성 분포 포함) {'summary_text': str, ...}.")
    final_summary: Optional[str] = Field(default=None, description="뉴스와 의견을 종합한 최종 요약.")

    # --- 평가 및 트렌드 (노드 11 & 12 출력) ---
    evaluation_metrics: Dict[str, float] = Field(default_factory=dict, description="최종 요약에 대한 평가 지표 {'rouge_l': float, 'bert_score': float, 'topic_coverage': float}.")
    decision: Optional[str] = Field(default=None, description="평가 기반 결정 ('proceed', 'research_again', 'refine_topic').")
    trend_scores: List[Dict[str, Any]] = Field(default_factory=list, description="키워드 트렌드 점수 목록 [{'keyword': str, 'score': float, ...}].")

    # --- 창작물 생성 (노드 14, 15, 17 출력) ---
    comic_ideas: List[Dict[str, Any]] = Field(default_factory=list, description="생성된 만화 아이디어 목록 [{'idea_title': str, 'concept': str, 'creative_score': float}].")
    # 참고: 'chosen_idea'는 노드 14 실행 후, 노드 15 실행 전에 채워져야 함
    #      (Human-in-the-loop 또는 자동 선택 노드 필요 가능성)
    chosen_idea: Optional[Dict[str, Any]] = Field(default=None, description="진행하기로 선택된 만화 아이디어.")
    scenarios: List[Dict[str, Any]] = Field(default_factory=list, description="4컷 패널 시나리오 목록 [{'scene': int, 'panel_description': str, 'dialogue': str, 'seed_tags': List[str]}].")
    scenario_prompt: Optional[str] = Field(default=None, description="시나리오 생성에 사용된 프롬프트 (노드 16 보고서용).")
    image_urls: List[str] = Field(default_factory=list, description="4개 패널에 대해 생성된 이미지 URL 목록.")

    # --- 번역 (선택적 - 노드 18 출력) ---
    translated_text: Optional[List[Dict[str, str]]] = Field(default=None, description="번역된 대화 목록 [{'scene': int, 'original_dialogue': str, 'translated_dialogue': str}].")

    # --- 최종 결과물 (노드 19 출력) ---
    final_comic: Dict[str, Optional[str]] = Field(default_factory=dict, description="최종 만화 결과물 상세 정보 (예: {'png_url': ..., 'webp_url': ..., 'alt_text': ...})")

    # # --- 추적 및 통계 ---
    # used_links: List[Dict[str, str]] = Field(default_factory=list, description="프로세스 중 사용된 URL 목록 [{'url': str, 'purpose': str, 'status': Optional[str]}].")
    # operator.add 리듀서를 사용하여 여러 노드에서 반환된 리스트를 이어붙임
    used_links: Annotated[List[Dict[str, str]], operator.add] = Field(default_factory=list)
    # ---------------------------

    #processing_stats: Dict[str, float] = Field(default_factory=dict, description="노드별 처리 시간 저장 딕셔너리 {'node_name_time': float, ...}.")
    node1_processing_stats: Optional[float] = Field(default=None, description="노드1 처리 시간 저장")
    node2_processing_stats: Optional[float] = Field(default=None, description="노드2 처리 시간 저장")
    node3_processing_stats: Optional[float] = Field(default=None, description="노드3 처리 시간 저장")
    node4_processing_stats: Optional[float] = Field(default=None, description="노드4 처리 시간 저장")
    node5_processing_stats: Optional[float] = Field(default=None, description="노드5 처리 시간 저장")
    node6_processing_stats: Optional[float] = Field(default=None, description="노드6 처리 시간 저장")
    node7_processing_stats: Optional[float] = Field(default=None, description="노드7 처리 시간 저장")
    node8_processing_stats: Optional[float] = Field(default=None, description="노드8 처리 시간 저장")
    node9_processing_stats: Optional[float] = Field(default=None, description="노드9 처리 시간 저장")
    node10_processing_stats: Optional[float] = Field(default=None, description="노드10 처리 시간 저장")
    node11_processing_stats: Optional[float] = Field(default=None, description="노드11 처리 시간 저장")
    node12_processing_stats: Optional[float] = Field(default=None, description="노드12 처리 시간 저장")
    node13_processing_stats: Optional[float] = Field(default=None, description="노드13 처리 시간 저장")
    node14_processing_stats: Optional[float] = Field(default=None, description="노드14 처리 시간 저장")
    node15_processing_stats: Optional[float] = Field(default=None, description="노드15 처리 시간 저장")
    node16_processing_stats: Optional[float] = Field(default=None, description="노드16 처리 시간 저장")
    node17_processing_stats: Optional[float] = Field(default=None, description="노드17 처리 시간 저장")
    node18_processing_stats: Optional[float] = Field(default=None, description="노드18 처리 시간 저장")
    node19_processing_stats: Optional[float] = Field(default=None, description="노드19 처리 시간 저장")

    # --- 오류 처리 ---
    #error_message: Optional[str] = Field(default=None, description="노드 실행 중 오류 발생 시 메시지 저장.")
    # --- error_message 필드 수정 ---
    # Annotated를 사용하여 리듀서 함수 지정
    error_message: Annotated[Optional[str], append_error_reducer] = None
    # --- 또는 오류 메시지를 리스트로 관리하는 방법 ---
    # error_messages: Annotated[List[str], operator.add] # operator.add는 리스트를 이어붙임

    class Config:
        # Pydantic 설정: 임의 타입 허용 (사용자 정의 객체 저장 시 유용)
        arbitrary_types_allowed = True
