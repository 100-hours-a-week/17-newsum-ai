# app/tools/search/kakao_search.py
import logging
import httpx
from typing import List, Dict, Optional, Any
from app.config.settings import settings

logger = logging.getLogger(__name__)

async def run_kakao_search(
    query: str,
    num_results: int = settings.DEFAULT_NUM_RESULTS,
    page: int = 1, # 페이지 번호 (1~50)
    sort: str = 'accuracy' # 정렬 옵션: accuracy (정확도순), recency (최신순)
) -> List[Dict[str, str]]:
    """카카오 웹 검색 API (Daum 검색 결과)를 사용하여 검색을 수행합니다."""
    logger.info(f"Kakao 검색 실행: query='{query}', num_results={num_results}")
    if not settings.KAKAO_REST_API_KEY:
        logger.error("Kakao REST API 키가 설정되지 않았습니다.")
        return []

    # API 엔드포인트 (웹문서 검색 - 필요시 'news' 등 다른 엔드포인트 사용 가능)
    api_url = "https://dapi.kakao.com/v2/search/web"

    headers = {
        "Authorization": f"KakaoAK {settings.KAKAO_REST_API_KEY}",
        'User-Agent': 'Mozilla/5.0 (... 생략 ...)'
    }
    params = {
        "query": query,
        "size": num_results, # 카카오는 size 파라미터 사용 (1~50)
        "page": page,
        "sort": sort
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            search_results: List[Dict[str, str]] = []
            if data and "documents" in data:
                for item in data["documents"]:
                    search_results.append({
                        "title": item.get("title"),
                        "link": item.get("url"), # 카카오는 url 사용
                        "snippet": item.get("contents"), # 카카오는 contents 사용
                    })
            logger.info(f"Kakao 검색 결과 {len(search_results)}개 반환.")
            return search_results

    except httpx.HTTPStatusError as e:
        logger.error(f"Kakao 검색 API 오류: Status {e.response.status_code}, Response: {e.response.text}")
        return []
    except Exception as e:
        logger.exception(f"Kakao 검색 중 예외 발생: {e}")
        return []