# app/agents/collector_agent.py (일부 수정)
import logging
from typing import List, Dict

from app.workflows.state import AppState
# 검색 '서비스 클라이언트' 대신 '도구'를 import
from app.tools.search.google_search_tool import run_google_search
from app.services.llm_server_client import call_llm_api
from app.config.settings import settings

logger = logging.getLogger(__name__)

async def collect_news(state: AppState) -> AppState:
    logger.info("--- Collector Agent 실행 시작 ---")
    query = state.get("initial_query")

    if not query:
        # ... 오류 처리 ...
        return state

    try:
        # 1. 검색 '도구' 사용
        logger.info(f"'{query}' 관련 뉴스 검색 중 (Google 검색 도구 사용)...")
        # 검색 도구 호출 (결과는 [{'title': ..., 'link': ..., 'snippet': ...}] 형식 가정)
        search_results: List[Dict[str, str]] = await run_google_search(query, num_results=5)

        if not search_results:
            # ... 검색 결과 없음 처리 ...
            return state

        # PoC 문서 [cite: 30] 와 유사하게 URL만 추출하거나, snippet 등 추가 정보 활용 가능
        # 여기서는 link 와 title 을 사용한다고 가정
        state["search_results"] = search_results # 검색 결과 전체 저장 (필요시)

        # 2. LLM을 이용한 선별 (필요시)
        use_llm_filter = True
        selected_links = []
        if use_llm_filter:
            logger.info("Filtering news articles for suitability using LLM...")
            prompt_items = [f"Title: {res['title']}\nSnippet: {res['snippet']}\nURL: {res['link']}"
                            for res in search_results]
            prompt = f"""From the following list of news items, select 1 to 3 articles that would be most interesting to turn into a 4-panel comic strip. Provide only the URLs of the selected articles, each on a new line.
            News
            Items:
            {chr(10).join(prompt_items)}
            Selected
            URLs:
            """
            llm_response: str = await call_llm_api(prompt) # 영어 프롬프트 전달
            selected_links = [url.strip() for url in llm_response.split('\n') if url.strip().startswith("http")]
            logger.info(f"LLM selected URLs: {selected_links}")

            llm_response: str = await call_llm_api(prompt)
            selected_links = [url.strip() for url in llm_response.split('\n') if url.strip().startswith("http")]
            logger.info(f"LLM이 선별한 URL: {selected_links}")
        if not selected_links:
             selected_links = [res['link'] for res in search_results] # LLM 실패 시 모든 링크 사용

        if not selected_links:
             return state

        state["news_urls"] = selected_links
        state["selected_url"] = selected_links[0]
        logger.info(f"최종 선택된 URL: {state['selected_url']}")


    except Exception as e:
        logger.exception(f"Collector Agent 실행 중 오류 발생: {e}")
        state["error_message"] = f"뉴스 수집 중 오류 발생: {str(e)}"

    logger.info("--- Collector Agent 실행 종료 ---")
    return state