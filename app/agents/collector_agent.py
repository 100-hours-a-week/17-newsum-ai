# app/agents/collector_agent.py
import logging
import json
from typing import List, Dict, Optional, Any
# from langdetect import detect as detect_language # 영어 제한 정책으로 불필요

# 상태, 도구, 서비스, 설정 import
from app.workflows.state import ComicState
from app.tools.search.google_search import run_google_search
from app.tools.search.tavily_search import run_tavily_search
# from app.tools.search.bing_search import run_bing_search # 함수는 import 유지 (맵에 포함)
from app.tools.search.naver_search import run_naver_search
from app.tools.search.kakao_search import run_kakao_search
from app.services.llm_server_client import call_llm_api
from app.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 검색 함수 맵핑 (Bing 포함, 호출 로직에서 제외)
SEARCH_FUNCTION_MAP = {
    "google": run_google_search,
    "tavily": run_tavily_search,
    # "bing": run_bing_search, # 맵에는 유지
    "naver": run_naver_search,
    "kakao": run_kakao_search,
}

# --- 검색 규칙 관련 설정 ---
PRIMARY_SEARCH_ENGINE = settings.DEFAULT_SEARCH_ENGINE.lower()
# *** Bing을 폴백에서 제외하고 주석 처리 ***
FALLBACK_ENGINES = ["google"] # Bing 제외
# FALLBACK_ENGINES = ["google", "bing"] # 이전 설정
REFERENCE_SEARCH_THRESHOLD = 3
REFERENCE_SEARCH_ENGINE = "naver"
KOREA_RELATED_KEYWORDS = [
    "korea", "korean", "seoul", "busan", "incheon", "daegu", "gwangju", "ulsan",
    "samsung", "hyundai", "lg", "sk", "posco", "kakao", "naver",
    "north korea", "pyongyang", "kim jong", "south korea", "rok", "kpop", "kdrama",
]

async def _execute_search(
    engine_name: str,
    query: str,
    num_results: int
) -> List[Dict[str, str]]:
    """선택된 검색 엔진 함수를 호출하는 헬퍼 함수"""
    engine_name_lower = engine_name.lower()
    if engine_name_lower not in SEARCH_FUNCTION_MAP:
        logger.error(f"[Search] Unsupported search engine specified: {engine_name_lower}")
        return []

    # *** Bing 호출 로직은 유지하되, FALLBACK_ENGINES 리스트에서 제외되어 호출되지 않음 ***
    search_func = SEARCH_FUNCTION_MAP[engine_name_lower]
    logger.info(f"[Search] Attempting search using {engine_name_lower} for query: '{query}' (num_results: {num_results})")

    try:
        results: List[Dict[str, str]] = []
        if engine_name_lower == "tavily":
            results = await search_func(query, max_results=num_results, topic="news")
        # elif engine_name_lower == "bing":
        #      # Bing이 FALLBACK_ENGINES 에서 제외되어 이 부분은 현재 호출되지 않음
        #      results = await search_func(query, num_results=num_results, market=settings.SEARCH_MARKET)
        else: # Google, Naver, Kakao
             results = await search_func(query, num_results=num_results)
        logger.debug(f"[Search] Raw results from {engine_name_lower} (first 2 shown): {json.dumps(results[:2], indent=2, ensure_ascii=False)}")
        return results
    except Exception as search_error:
        logger.error(f"[Search] Search failed for engine {engine_name_lower}: {search_error}")
        return []

def _contains_korea_keywords(query: str) -> bool:
    """쿼리에 한국 관련 키워드가 있는지 확인"""
    # ... (이전과 동일) ...
    query_lower = query.lower()
    found = any(keyword.lower() in query_lower for keyword in KOREA_RELATED_KEYWORDS)
    logger.debug(f"[Rule Check] Korea keywords check for query '{query}': {found}")
    return found

async def collect_news(state: ComicState) -> Dict[str, Optional[Any]]:
    """
    [Collector Agent]
    규칙에 따라 검색 엔진을 선택/폴백(Bing 제외)/참조하여 검색하고,
    LLM 필터링 후 상태 업데이트 딕셔너리를 반환합니다.
    """
    logger.info("--- [Collector Agent] 실행 시작 ---")
    query = state.initial_query
    updates: Dict[str, Optional[Any]] = {}

    if not query:
        logger.error("[Collector Agent] 초기 쿼리가 상태에 없습니다.")
        updates["error_message"] = "Input query is missing."
        updates["news_urls"] = []  # Ensure news_urls is empty when there's an error
        updates["selected_url"] = None  # Also clear selected_url
        return updates

    logger.info(f"[Collector Agent] Received query: '{query}'")

    try:
        num_results = settings.DEFAULT_NUM_RESULTS
        search_results: List[Dict[str, str]] = []
        used_engines = []
        korea_keywords_found = _contains_korea_keywords(query)
        referenced_naver_kakao = False # Naver/Kakao 참조 여부 플래그

        # --- 1. 기본 검색 엔진 시도 ---
        logger.info(f"[Collector Agent] Starting search with primary engine: {PRIMARY_SEARCH_ENGINE}")
        search_results = await _execute_search(PRIMARY_SEARCH_ENGINE, query, num_results)
        if search_results:
            used_engines.append(PRIMARY_SEARCH_ENGINE)
            logger.info(f"[Collector Agent] Found {len(search_results)} results using primary engine: {PRIMARY_SEARCH_ENGINE}.")

        # --- 2. 조건부 Naver 참조 검색 ---
        if (len(search_results) < REFERENCE_SEARCH_THRESHOLD) and korea_keywords_found:
            logger.info(f"[Collector Agent] Primary results insufficient ({len(search_results)}) and Korea keywords found. Trying {REFERENCE_SEARCH_ENGINE} search as reference.")
            reference_results = await _execute_search(REFERENCE_SEARCH_ENGINE, query, num_results)
            if reference_results:
                referenced_naver_kakao = True # 참조 검색 사용 플래그
                used_engines.append(f"{REFERENCE_SEARCH_ENGINE}(ref)")
                logger.info(f"[Collector Agent] Found {len(reference_results)} results from {REFERENCE_SEARCH_ENGINE} as reference.")
                if not search_results:
                     search_results = reference_results
                     logger.info(f"[Collector Agent] Using reference results as primary results were empty.")
            else:
                 logger.info(f"[Collector Agent] {REFERENCE_SEARCH_ENGINE} reference search returned no results.")

        # --- 3. 최종 폴백 검색 시도 (Bing 제외) ---
        if not search_results:
            logger.warning(f"[Collector Agent] No results found yet. Trying fallbacks: {FALLBACK_ENGINES}")
            for fallback_engine in FALLBACK_ENGINES:
                # 이미 시도한 엔진 (Primary 또는 Reference)이면 건너뜀
                if fallback_engine == PRIMARY_SEARCH_ENGINE or (fallback_engine == REFERENCE_SEARCH_ENGINE and referenced_naver_kakao):
                    continue
                search_results = await _execute_search(fallback_engine, query, num_results)
                if search_results:
                    used_engines.append(f"{fallback_engine}(fallback)")
                    logger.info(f"[Collector Agent] Successfully found results using fallback engine: {fallback_engine}.")
                    break
            if not search_results:
                 logger.error(f"[Collector Agent] No search results found after trying all engines. Engines tried: {used_engines}")
                 updates["error_message"] = "Could not find relevant news articles using any configured search engine."
                 updates["news_urls"] = []  # Ensure news_urls is empty when there's an error
                 updates["selected_url"] = None  # Also clear selected_url
                 return updates

        # --- 4. 검색 결과 처리 ---
        logger.info(f"[Collector Agent] Final search results count: {len(search_results)}. Engines used: {used_engines}.")
        logger.debug(f"[Collector Agent] Final search results (URLs): {[res.get('link') for res in search_results]}")
        updates["search_results"] = search_results # 전체 검색 결과 저장

        # --- 5. LLM 필터링 (선택 사항) ---
        use_llm_filter = True
        selected_links = []
        if use_llm_filter and search_results:
            logger.info("[Collector Agent] Starting LLM filtering...")
            # ... (LLM 필터링 로직 상세 - 이전 답변과 동일) ...
            try:
                 prompt_items = [f"Title: {res.get('title', 'N/A')}\nSnippet: {res.get('snippet', 'N/A')}\nURL: {res.get('link', 'N/A')}"
                                 for res in search_results if res.get('link')]
                 if prompt_items:
                     prompt = f"""From the following list of news items, select 1 to 3 articles that would be most interesting to turn into a 4-panel comic strip. Provide only the URLs of the selected articles, each on a new line.

News Items:
---
{chr(10).join(prompt_items)}
---

Selected URLs:
"""
                     llm_response: str = await call_llm_api(prompt, max_tokens=150, temperature=0.5)
                     selected_links = [url.strip() for url in llm_response.split('\n') if url.strip().startswith("http")]
                     logger.info(f"[Collector Agent] LLM selected {len(selected_links)} URLs: {selected_links}")
                 else:
                      logger.warning("[Collector Agent] No valid items to send to LLM for filtering.")
                      selected_links = [res['link'] for res in search_results if res.get('link')]
            except Exception as llm_error:
                logger.error(f"[Collector Agent] LLM filtering failed: {llm_error}. Proceeding without LLM filter.")
                selected_links = [res['link'] for res in search_results if res.get('link')]
        else:
            logger.info("[Collector Agent] LLM filtering is disabled or no search results available. Using all found URLs.")
            selected_links = [res['link'] for res in search_results if res.get('link')]

        if not selected_links:
            selected_links = [res['link'] for res in search_results if res.get('link')]

        # --- 6. 상태 업데이트 ---
        if not selected_links:
            logger.warning("[Collector Agent] No suitable news URLs selected finally.")
            updates["news_urls"] = [] # 빈 리스트 전달
            updates["selected_url"] = None
            updates["error_message"] = "No suitable news URLs could be selected."
        else:
            logger.info(f"[Collector Agent] Final selected URLs for scraping ({len(selected_links)}): {selected_links}")
            updates["news_urls"] = selected_links # 필터링된(또는 전체) URL 리스트
            # selected_url은 단일 URL만 필요했던 이전 방식이므로, 리스트를 전달하는 지금은 필요 없을 수 있음.
            # 하지만 하위 호환성 또는 다른 로직 위해 남겨두거나, 첫번째 URL을 저장할 수 있음.
            # 여기서는 첫번째 URL 저장 유지.
            updates["selected_url"] = selected_links[0]
            updates["error_message"] = None

    except Exception as e:
        logger.exception(f"[Collector Agent] Execution failed unexpectedly: {e}")
        updates["error_message"] = f"Unexpected error during news collection: {str(e)}"
        updates["news_urls"] = []  # Ensure news_urls is empty when there's an error
        updates["selected_url"] = None  # Also clear selected_url

    logger.info("--- [Collector Agent] 실행 종료 ---")
    logger.debug(f"[Collector Agent] Returning updates: {json.dumps(updates, indent=2, ensure_ascii=False)}")
    return updates