# app/workflows/transitions.py
from langgraph.graph import END
import logging

logger = logging.getLogger(__name__)

def determine_next_step(state) -> str:
    """
    현재 상태를 기반으로 다음 단계 결정 (예: 조건부 분기용)
    """
    # 에러 상태 확인 및 처리
    if state.error_message:
        logger.warning(f"워크플로우 오류 감지: {state.error_message}. 워크플로우를 종료합니다.")
        return END  # 에러 발생 시 워크플로우 종료
        
    # 필수 데이터 존재 여부에 따른 다음 단계 결정
    if not state.news_urls:
        return "collect"

    if not state.articles:
        return "scrape"

    if not state.summaries:
        return "summarize_individual"
        
    if not state.final_summary:
        return "summarize_synthesis"
        
    if state.additional_context is None:  # 컨텍스트 분석 결과가 없으면
        return "analyze_content"

    if not state.humor_texts:
        return "humor"

    if not state.scenarios:
        return "scenario"

    if not state.image_urls:
        return "image"

    if not state.final_comic_url:
        return "postprocess"

    if not state.translated_texts:
        return "translate"

    return "finish"
