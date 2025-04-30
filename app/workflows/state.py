# app/workflows/state.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any # Dict, Any 추가

class ComicState(BaseModel):
    """
    워크플로우 실행 중 전달되는 상태 정보 정의
    """
    initial_query: Optional[str] = Field(default=None, description="사용자의 초기 입력 쿼리")
    search_results: Optional[List[Dict[str, str]]] = Field(default=None, description="검색 엔진 결과")
    news_urls: Optional[List[str]] = Field(default_factory=list, description="선별된 뉴스 URL 목록")
    selected_url: Optional[str] = Field(default=None, description="대표 URL (사용 여부 재검토 필요)")
    articles: Optional[List[str]] = Field(default_factory=list, description="스크랩된 기사 본문 리스트")
    summaries: Optional[List[str]] = Field(default_factory=list, description="개별 기사 요약문 리스트") # <--- 개별 요약 저장
    final_summary: Optional[str] = Field(default=None, description="개별 요약들을 종합한 최종 요약문") # <--- 최종 요약 저장 필드 추가
    additional_context: Optional[Dict[str, Any]] = Field(default=None, description="YouTube 영상 및 댓글에서 추출한 추가 컨텍스트 정보")
    humor_texts: Optional[List[str]] = Field(default_factory=list) # (이후 단계)
    scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list) # (이후 단계)
    image_urls: Optional[List[str]] = Field(default_factory=list) # (이후 단계)
    final_comic_url: Optional[str] = None # (이후 단계)
    translated_texts: Optional[List[str]] = Field(default_factory=list) # (이후 단계)
    error_message: Optional[str] = None

    # Pydantic 모델 설정 (선택 사항)
    # class Config:
    #     extra = 'ignore' # 모델에 정의되지 않은 필드는 무시