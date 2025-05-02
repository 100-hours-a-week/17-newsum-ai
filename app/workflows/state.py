# app/workflows/state.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ComicState(BaseModel):
    """
    Defines the state passed between nodes in the LangGraph workflow.
    Based on the specification document and README.md.
    """
    comic_id: Optional[str] = Field(default=None, description="만화 생성 작업 고유 ID")
    initial_query: Optional[str] = Field(default=None, description="사용자의 초기 입력 쿼리")
    search_results: Optional[List[Dict[str, str]]] = Field(default=None, description="검색 엔진 결과")
    news_urls: Optional[List[str]] = Field(default_factory=list, description="선별된 뉴스 URL 목록")
    selected_url: Optional[str] = Field(default=None, description="대표 URL (사용 여부 재검토 필요)")
    articles: Optional[List[str]] = Field(default_factory=list, description="스크랩된 기사 본문 리스트")
    summaries: Optional[List[str]] = Field(default_factory=list, description="개별 기사 요약문 리스트") # <--- 개별 요약 저장
    final_summary: Optional[str] = Field(default=None, description="개별 요약들을 종합한 최종 요약문") # <--- 최종 요약 저장 필드 추가
    additional_context: Optional[Dict[str, Any]] = Field(default=None, description="YouTube 영상 및 댓글에서 추출한 추가 컨텍스트 정보")
    public_sentiment: Optional[Dict[str, Dict[str, float]]] = Field(default=None, description="감정 분석 결과")
    humor_texts: Optional[List[str]] = Field(default_factory=list, description="유머 포인트 목록")
    scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="4컷 만화 시나리오")
    image_urls: Optional[List[str]] = Field(default_factory=list, description="생성된 이미지 URL 목록")
    final_comic_url: Optional[str] = Field(default=None, description="최종 조합된 만화 URL")
    translated_texts: Optional[List[str]] = Field(default_factory=list, description="번역된 텍스트(대사 등)")
    error_message: Optional[str] = Field(default=None, description="에러 발생 시 메시지")
    processing_stats: Optional[Dict[str, Any]] = Field(default_factory=dict, description="처리 시간 등 성능 통계")

    # Tracking & Stats
    used_links: List[Dict[str, str]] = Field(default_factory=list, description="List of URLs used during the process [{'url': str, 'purpose': str, 'status': Optional[str]}].")
    processing_stats: Dict[str, float] = Field(default_factory=dict, description="Dictionary to store processing time per node {'node_name_time': float, ...}.")

    # Error Handling
    error_message: Optional[str] = Field(default=None, description="Stores error messages if a node fails.")

    class Config:
        # Allows for arbitrary types, useful if custom objects are stored
        arbitrary_types_allowed = True