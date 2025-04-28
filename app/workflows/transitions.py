# app/workflows/transitions.py

# ============================================================
#               여기에 Conditional Edges를 추가해야함
# ============================================================
def determine_next_step(state) -> str:
    """
    현재 상태를 기반으로 다음 단계 결정 (예: 조건부 분기용)
    """
    if not state.news_urls:
        return "collect"

    if not state.articles:
        return "scrape"

    if not state.summaries:
        return "analyze"

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