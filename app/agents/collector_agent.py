# app/agents/collector_agent.py
import logging
from typing import List, Dict, Optional # Optional 추가

# 상태, 도구, 서비스, 설정 import
from app.workflows.state import ComicState
from app.tools.llm.google_search_tool import run_google_search # 경로 확인!
from app.services.llm_server_client import call_llm_api
from app.config.settings import settings # settings 객체 직접 사용

# 로거 설정 (간단 예시)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def collect_news(state: ComicState) -> Dict[str, Optional[List[str]] | Optional[str] | Optional[List[Dict[str, str]]]]:
    """
    초기 쿼리를 기반으로 뉴스를 검색하고, LLM을 이용해 스크랩할 뉴스를 선별하는 에이전트.
    상태 업데이트 내용을 담은 딕셔너리를 반환합니다.
    """
    logger.info("--- Collector Agent 실행 시작 ---")
    query = state.initial_query # Pydantic 모델 직접 접근

    updates: Dict[str, Optional[List[str]] | Optional[str] | Optional[List[Dict[str, str]]]] = {}

    if not query:
        logger.error("초기 쿼리가 상태에 없습니다.")
        updates["error_message"] = "Input query is missing."
        return updates # 업데이트 딕셔너리 반환

    try:
        # 1. Google 검색 도구 사용
        logger.info(f"Searching news related to '{query}' using Google Search tool...")
        # 검색 결과: List[Dict[str, str]] (title, link, snippet 포함)
        search_results: List[Dict[str, str]] = await run_google_search(query, num_results=5)
        if not search_results:
            logger.warning("No search results found.")
            updates["error_message"] = "Could not find relevant news articles."
            return updates

        logger.info(f"Found {len(search_results)} news articles.")
        updates["search_results"] = search_results # 업데이트할 내용 추가

        # 2. LLM을 이용한 스크랩 대상 선별 (영어 프롬프트 사용)
        use_llm_filter = True # LLM 필터 사용 여부 결정 로직 필요 시 추가
        selected_links = []
        if use_llm_filter:
            logger.info("Filtering news articles for suitability using LLM...")
            # 검색 결과의 title, snippet 정보를 프롬프트에 활용
            prompt_items = [f"Title: {res.get('title', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}\nURL: {res.get('link', 'N/A')}"
                            for res in search_results if res.get('link')] # 링크가 있는 결과만 포함

            if not prompt_items:
                 logger.warning("No valid items to send to LLM for filtering.")
                 state.error_message = "Could not prepare items for LLM filtering."
                 return state

            # 영어 프롬프트
            prompt = f"""From the following list of news items, select 1 to 3 articles that would be most interesting to turn into a 4-panel comic strip. Provide only the URLs of the selected articles, each on a new line.

News Items:
---
{chr(10).join(prompt_items)}
---

Selected URLs:
"""
            # LLM API 호출
            try:
                llm_response: str = await call_llm_api(prompt, max_tokens=150, temperature=0.5) # max_tokens 조정 필요
                selected_links = [url.strip() for url in llm_response.split('\n') if url.strip().startswith("http")]
                logger.info(f"LLM selected URLs: {selected_links}")
            except Exception as llm_error:
                logger.error(f"LLM filtering failed: {llm_error}. Proceeding without LLM filter.")
                # LLM 호출 실패 시 필터링 없이 진행하도록 fallback (선택 사항)
                selected_links = [res['link'] for res in search_results if res.get('link')]

        # LLM 필터링 사용 안 하거나 실패 시 모든 링크 사용
        if not selected_links:
            logger.info("Using all found URLs as LLM filtering was skipped or yielded no results.")
            selected_links = [res['link'] for res in search_results if res.get('link')]

        if not selected_links:
            logger.warning("No suitable news URLs selected.")
            updates["error_message"] = "Could not select any suitable news URLs."
            return updates

        # # 3. 상태 업데이트 (Pydantic 모델 직접 수정)
        # state.news_urls = selected_links
        # # 첫 번째 URL을 대표 URL로 선택 (단일 처리 가정 시)
        # state.selected_url = selected_links[0]
        # logger.info(f"Final selected URL: {state.selected_url}")
        # logger.info(f"Collected URL list: {state.news_urls}")
        # # 이전 오류 메시지 초기화 (성공 시)
        # state.error_message = None

        # 상태 업데이트 내용 추가
        updates["news_urls"] = selected_links
        updates["selected_url"] = selected_links[0]
        updates["error_message"] = None # 성공 시 오류 메시지 초기화

    except Exception as e:
        logger.exception(f"Collector Agent execution failed: {e}")
        updates["error_message"] = f"Error during news collection: {str(e)}"

    logger.info("--- Collector Agent 실행 종료 ---")
    # *** 수정된 반환 방식: 변경된 필드만 담은 딕셔너리 반환 ***
    return updates