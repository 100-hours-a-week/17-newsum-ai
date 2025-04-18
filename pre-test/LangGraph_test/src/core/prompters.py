# src/core/prompters.py
from .schemas import HumorResult, ImagePromptResult, AnalysisResult
from .utils import logger

def make_prompt(humor_result: HumorResult, analysis_result: AnalysisResult) -> ImagePromptResult:
    """유머와 분석 결과 기반으로 이미지 프롬프트 생성"""
    logger.info(f"Drafting image prompt for {humor_result.news_id}...")

    # 키워드와 유머 텍스트 조합
    base_keywords = ", ".join(analysis_result.keywords[:5]) if analysis_result.keywords else "news"
    humor_essence = humor_result.humor_text[:100] # 유머 텍스트 일부 사용

    # 스타일 토큰 추가 (예시)
    style_tokens = "cartoon style, humorous, funny illustration, vibrant colors, high detail"

    positive_prompt = f"{humor_essence}, {base_keywords}, {style_tokens}"
    # 길이 제한 및 정제
    positive_prompt = positive_prompt[:250] # SDXL 프롬프트 길이 고려

    result = ImagePromptResult(
        news_id=humor_result.news_id,
        positive_prompt=positive_prompt
        # negative_prompt는 schemas에서 기본값 사용
    )
    logger.info(f"Image prompt drafted for {humor_result.news_id}: {result.positive_prompt[:60]}...")
    return result

# ----------------------------------------