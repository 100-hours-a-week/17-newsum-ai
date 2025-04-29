# app/tools/search/tavily_search.py
import logging
import asyncio
from typing import List, Dict, Optional
from tavily import TavilyClient # tavily-python 라이브러리 설치 필요: pip install tavily-python
from app.config.settings import settings

logger = logging.getLogger(__name__)

async def run_tavily_search(
    query: str,
    search_depth: str = "basic", # "basic" 또는 "advanced"
    max_results: int = settings.DEFAULT_NUM_RESULTS,
    include_domains: Optional[List[str]] = settings.SEARCH_INCLUDE_DOMAINS,
    exclude_domains: Optional[List[str]] = settings.SEARCH_EXCLUDE_DOMAINS,
    time_range: Optional[str] = settings.SEARCH_TIME_RANGE, # 예: 'month'
    topic: str = "news" # "general", "news", "finance" 등
) -> List[Dict[str, str]]:
    """Tavily Search API를 사용하여 웹 검색을 수행합니다."""
    logger.info(f"Tavily 검색 실행: query='{query}', max_results={max_results}, topic={topic}")
    if not settings.TAVILY_API_KEY:
        logger.error("Tavily API 키가 설정되지 않았습니다.")
        return []

    try:
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        # TavilyClient는 비동기 메서드를 직접 제공하지 않으므로, asyncio.to_thread 사용
        response = await asyncio.to_thread(
            client.search,
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            # time_range=time_range, # time_range 파라미터는 현재 tavily-python SDK에서 직접 지원하지 않는 것으로 보임 (문서 재확인 필요)
            topic=topic,
            include_answer=False, # 필요시 True 설정 가능
            include_raw_content=False,
            include_images=False,
        )

        # 결과 형식 변환 (Google/Bing과 유사하게)
        search_results: List[Dict[str, str]] = []
        if response and "results" in response:
            for item in response["results"]:
                search_results.append({
                    "title": item.get("title"),
                    "link": item.get("url"), # Tavily는 'url' 키 사용
                    "snippet": item.get("content"), # Tavily는 'content' 키 사용
                })
        logger.info(f"Tavily 검색 결과 {len(search_results)}개 반환.")
        return search_results

    except Exception as e:
        logger.exception(f"Tavily 검색 중 예외 발생: {e}")
        return []