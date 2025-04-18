# src/core/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class NewsItem(BaseModel):
    id: str = Field(..., description="뉴스 기사 고유 ID (예: URL 해시)")
    title: str = Field(..., description="뉴스 제목")
    url: str = Field(..., description="원본 뉴스 URL")
    content: str = Field(..., description="뉴스 본문 (또는 요약)")
    source: str = Field(..., description="뉴스 출처 (예: BBC, CNN)")
    published_time: Optional[datetime] = Field(None, description="발행 시각")

class AnalysisResult(BaseModel):
    news_id: str
    summary: Optional[str] = None # Llama가 생성한 요약
    keywords: List[str] = [] # Llama가 추출한 키워드
    key_event: Optional[str] = None # CoT 결과: 핵심 사건
    twist_point: Optional[str] = None # CoT 결과: 반전 포인트

class HumorResult(BaseModel):
    news_id: str
    humor_text: str = Field(..., description="생성된 유머 텍스트")
    humor_style: Optional[str] = Field(None, description="적용된 유머 스타일 (예: satire, pun)")
    confidence: Optional[float] = Field(None, description="모델 생성 신뢰도 또는 평가 점수")

class ImagePromptResult(BaseModel):
    news_id: str
    positive_prompt: str
    negative_prompt: str = "text, words, letters, watermark, signature, username, artist name, low quality, blurry"

class ImageRenderResult(BaseModel):
    news_id: str
    image_path: str = Field(..., description="생성된 이미지 파일 저장 경로")
    is_fallback: bool = False # 폴백 이미지를 사용했는지 여부

class FinalContent(BaseModel):
    news_id: str
    title: str
    url: str
    source: str
    published_time: Optional[datetime]
    humor_text: str
    image_path: str # 이미지 접근을 위한 상대 경로 또는 URL
    created_at: datetime = Field(default_factory=datetime.now)

# --- LangGraph Workflow State ---
from typing_extensions import TypedDict

class WorkflowState(TypedDict):
    """LangGraph 워크플로우 상태 정의"""
    news_items: Optional[List[NewsItem]] # 수집된 전체 뉴스 리스트
    current_news_item: Optional[NewsItem] # map 연산 시 현재 처리 중인 뉴스
    analysis_result: Optional[AnalysisResult]
    humor_result: Optional[HumorResult]
    image_prompt_result: Optional[ImagePromptResult]
    image_render_result: Optional[ImageRenderResult]
    final_content: Optional[FinalContent] # 단일 처리 결과
    processed_results: List[FinalContent] # map 연산 후 최종 결과 모음
    error_message: Optional[str] # 에러 발생 시 메시지