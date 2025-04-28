# app/workflows/state.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any # Dict, Any 추가

class ComicState(BaseModel):
    """
    워크플로우 실행 중 전달되는 상태 정보 정의
    """
    # *** 추가된 필드 ***
    initial_query: Optional[str] = Field(default=None, description="사용자의 초기 입력 쿼리")

    # Collector 결과 관련 필드
    search_results: Optional[List[Dict[str, str]]] = Field(default=None, description="검색 엔진 결과 (title, link, snippet)")
    news_urls: Optional[List[str]] = Field(default_factory=list, description="LLM이 선별한 뉴스 URL 목록")
    selected_url: Optional[str] = Field(default=None, description="스크랩 대상으로 최종 선택된 URL") # 단일 처리 가정 시

    # 이후 단계 필드 (빈 리스트 또는 None으로 초기화)
    articles: Optional[List[str]] = Field(default_factory=list) # 실제로는 article: Optional[str] = None 이 나을 수 있음 (단일 처리 시)
    summaries: Optional[List[str]] = Field(default_factory=list) # summary: Optional[str] = None
    humor_texts: Optional[List[str]] = Field(default_factory=list) # humor_text: Optional[str] = None
    # 시나리오 구조는 List[Dict] 또는 List[Pydantic 모델] 권장
    scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    image_urls: Optional[List[str]] = Field(default_factory=list)
    final_comic_url: Optional[str] = None
    translated_texts: Optional[List[str]] = Field(default_factory=list)

    # 오류 관리용
    error_message: Optional[str] = None

    # Pydantic 모델 설정 (선택 사항)
    # class Config:
    #     extra = 'ignore' # 모델에 정의되지 않은 필드는 무시