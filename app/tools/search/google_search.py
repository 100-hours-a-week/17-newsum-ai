# app/tools/search/google_search.py
import logging
import asyncio
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from app.config.settings import settings

logger = logging.getLogger(__name__)

async def run_google_search(query: str, num_results: int = settings.DEFAULT_NUM_RESULTS) -> List[Dict[str, str]]:
    """Google Custom Search API를 사용하여 웹 검색을 수행합니다."""
    logger.info(f"Google 검색 실행: query='{query}', num_results={num_results}")
    if not settings.GOOGLE_API_KEY or not settings.GOOGLE_CSE_ID:
        logger.error("Google API 키 또는 CSE ID가 설정되지 않았습니다.")
        return []
    try:
        # asyncio.to_thread 사용하여 동기 라이브러리 호출
        results = await asyncio.to_thread(
            _execute_google_search_sync,
            query,
            num_results,
            settings.GOOGLE_API_KEY,
            settings.GOOGLE_CSE_ID
        )
        return results
    except HttpError as e:
        logger.error(f"Google 검색 API 오류: {e}")
        return []
    except Exception as e:
        logger.exception(f"Google 검색 중 예외 발생: {e}")
        return []

def _execute_google_search_sync(query: str, num_results: int, api_key: str, cse_id: str) -> List[Dict[str, str]]:
    service = build("customsearch", "v1", developerKey=api_key)
    response = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
    search_results: List[Dict[str, str]] = []
    if 'items' in response:
        for item in response['items']:
            search_results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            })
    logger.info(f"Google 검색 결과 {len(search_results)}개 반환.")
    return search_results