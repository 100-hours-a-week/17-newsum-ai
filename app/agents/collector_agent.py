# app/agents/collector_agent.py
import logging
from typing import List, Dict, Optional, Any
# from langdetect import detect as detect_language # 영어 제한 정책으로 불필요

# 상태, 도구, 서비스, 설정 import
from app.workflows.state import ComicState
from app.tools.search.google_search import run_google_search
from app.tools.search.tavily_search import run_tavily_search
# from app.tools.search.bing_search import run_bing_search
from app.tools.search.naver_search import run_naver_search
from app.tools.search.kakao_search import run_kakao_search
from app.services.llm_server_client import call_llm_api
from app.config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 검색 함수 맵핑
SEARCH_FUNCTION_MAP = {
    "google": run_google_search,
    "tavily": run_tavily_search,
    #"bing": run_bing_search,
    "naver": run_naver_search,
    "kakao": run_kakao_search,
}

# --- 새로운 설정 ---
# 기본 검색 엔진 및 폴백 순서 (Tavily -> Google -> Bing)
PRIMARY_SEARCH_ENGINE = settings.DEFAULT_SEARCH_ENGINE.lower() # .env 에서 tavily 등으로 설정
FALLBACK_ENGINES = ["google"] #, "bing"] # 기본 폴백 순서
# 한국 관련 키워드 목록 (settings.py 또는 여기서 관리)
KOREA_RELATED_KEYWORDS = [
    "korea", "korean", "seoul", "busan", "incheon", "daegu", "gwangju", "ulsan",
    "samsung", "hyundai", "lg", "sk", "posco", "kakao", "naver", # 주요 기업명
    "north korea", "pyongyang", "kim jong", "south korea", "rok", # 남북 관련
    "kpop", "kdrama", # 문화 관련 (예시)
    # 필요에 따라 더 많은 키워드 추가
]
# Naver/Kakao 참조 검색 조건 (결과 수 기준)
REFERENCE_SEARCH_THRESHOLD = 3 # 기본 검색 결과가 이 숫자 미만일 때 Naver/Kakao 참조 시도

async def _execute_search(
    engine_name: str,
    query: str,
    num_results: int
) -> List[Dict[str, str]]:
    """선택된 검색 엔진 함수를 호출하는 헬퍼 함수"""
    # ... (이전 답변과 동일) ...
    if engine_name not in SEARCH_FUNCTION_MAP:
        logger.error(f"Unsupported search engine specified: {engine_name}")
        return []
    search_func = SEARCH_FUNCTION_MAP[engine_name]
    logger.info(f"Attempting search using {engine_name}...")
    try:
        if engine_name == "tavily":
            return await search_func(query, max_results=num_results, topic="news")
        # elif engine_name == "bing":
        #      return await search_func(query, num_results=num_results, market=settings.SEARCH_MARKET)
        else: # Google, Naver, Kakao
             return await search_func(query, num_results=num_results)
    except Exception as search_error:
        logger.error(f"Search failed for engine {engine_name}: {search_error}")
        return []

def _contains_korea_keywords(query: str) -> bool:
    """쿼리에 한국 관련 키워드가 있는지 확인"""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in KOREA_RELATED_KEYWORDS)

async def collect_news(state: ComicState) -> Dict[str, Optional[Any]]:
    """
    영어로 된 초기 쿼리를 받아, 정의된 규칙에 따라 검색 엔진을 선택/폴백하며
    (국내 이슈 키워드 발견 시 Naver/Kakao 참조), 뉴스를 검색하고,
    LLM 필터링 후 상태 업데이트 딕셔너리를 반환합니다.
    """
    logger.info("--- Collector Agent 실행 시작 ---")
    query = state.initial_query
    updates: Dict[str, Optional[Any]] = {}

    if not query:
        logger.error("초기 쿼리가 상태에 없습니다.")
        updates["error_message"] = "Input query is missing."
        return updates

    try:
        num_results = settings.DEFAULT_NUM_RESULTS
        search_results: List[Dict[str, str]] = []
        used_engine = None
        referenced_naver_kakao = False

        # --- 1. 기본 검색 엔진 시도 ---
        search_results = await _execute_search(PRIMARY_SEARCH_ENGINE, query, num_results)
        if search_results:
            used_engine = PRIMARY_SEARCH_ENGINE
            logger.info(f"Successfully found results using primary engine: {used_engine}.")

        # --- 2. 조건부 Naver/Kakao 참조 검색 ---
        # 기본 검색 결과가 부족하고, 쿼리에 한국 관련 키워드가 있을 경우
        if (len(search_results) < REFERENCE_SEARCH_THRESHOLD) and _contains_korea_keywords(query):
            logger.info(f"Primary search results insufficient ({len(search_results)}) and Korea keywords found. Trying Naver search as reference.")
            naver_results = await _execute_search("naver", query, num_results)
            if naver_results:
                referenced_naver_kakao = True
                logger.info(f"Found {len(naver_results)} results from Naver as reference.")
                # 결과 처리 방식 선택:
                # 옵션 1: Naver 결과를 기본 결과 대신 사용 (기본 결과가 없을 때만)
                if not search_results:
                     search_results = naver_results
                     used_engine = "naver (conditional)"
                # 옵션 2: Naver 결과를 기존 결과에 추가 (중복 제거 필요)
                # combined_urls = {res['link'] for res in search_results}
                # for res in naver_results:
                #     if res['link'] not in combined_urls:
                #         search_results.append(res)
                #         combined_urls.add(res['link'])
                # logger.info(f"Combined results. Total: {len(search_results)}")
                # 여기서는 옵션 1 (기본 결과 없을 때만 사용) 채택
            else:
                 logger.info("Naver reference search returned no results.")

        # --- 3. 최종 폴백 검색 시도 ---
        if not search_results:
            logger.warning(f"No results found from primary ({PRIMARY_SEARCH_ENGINE}) or conditional Naver search. Trying fallbacks: {FALLBACK_ENGINES}")
            for fallback_engine in FALLBACK_ENGINES:
                search_results = await _execute_search(fallback_engine, query, num_results)
                if search_results:
                    used_engine = f"{fallback_engine} (fallback)"
                    logger.info(f"Successfully found results using fallback engine: {used_engine}.")
                    break # 폴백 성공 시 중단
            if not search_results:
                 logger.error(f"No search results found after trying all engines.")
                 updates["error_message"] = "Could not find relevant news articles using any configured search engine."
                 return updates

        # --- 4. 검색 결과 처리 ---
        logger.info(f"Found {len(search_results)} news articles using {used_engine}.")
        updates["search_results"] = search_results

        # --- 5. LLM 필터링 (선택 사항) ---
        # ... (LLM 필터링 로직 - 이전과 동일) ...
        use_llm_filter = True
        selected_links = []
        # (LLM 필터링 로직 상세 생략 - 이전 코드 참고)
        if use_llm_filter:
            # ...
             pass
        else:
            selected_links = [res['link'] for res in search_results if res.get('link')]

        if not selected_links:
            selected_links = [res['link'] for res in search_results if res.get('link')]

        if not selected_links:
             logger.warning("No suitable news URLs selected.")
             updates["error_message"] = "Could not select any suitable news URLs."
             return updates

        # --- 6. 상태 업데이트 ---
        updates["news_urls"] = selected_links
        updates["selected_url"] = selected_links[0]
        updates["error_message"] = None # 성공 시 오류 초기화

    except Exception as e:
        logger.exception(f"Collector Agent execution failed unexpectedly: {e}")
        updates["error_message"] = f"Unexpected error during news collection: {str(e)}"

    logger.info("--- Collector Agent 실행 종료 ---")
    return updates