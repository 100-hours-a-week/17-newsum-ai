# app/workflows/state.py

from pydantic import BaseModel, Field
from typing import List, Optional

class ComicState(BaseModel):
    """
    워크플로우 실행 중 전달되는 상태 정보 정의
    """
    news_urls: Optional[List[str]] = Field(default_factory=list)
    articles: Optional[List[str]] = Field(default_factory=list)
    summaries: Optional[List[str]] = Field(default_factory=list)
    humor_texts: Optional[List[str]] = Field(default_factory=list)
    scenarios: Optional[List[str]] = Field(default_factory=list)
    image_urls: Optional[List[str]] = Field(default_factory=list)
    final_comic_url: Optional[str] = None
    translated_texts: Optional[List[str]] = Field(default_factory=list)

    # 오류 관리용
    error_message: Optional[str] = None