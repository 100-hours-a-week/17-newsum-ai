# app/tools/search/naver_search.py
import logging
import httpx
import re # HTML 태그 제거용
from typing import List, Dict, Optional, Any
from app.config.settings import settings

logger = logging.getLogger(__name__)

# HTML 태그 제거용 정규 표현식
TAG_RE = re.compile(r'<[^>]+>')

def remove_html_tags(text: str) -> str:
    """간단한 HTML 태그 제거 함수"""
    return TAG_RE.sub('', text) if text else text

async def run_naver_search(
    query: str,
    num_results: int = settings.DEFAULT_NUM_RESULTS,
    start: int = 1, # 검색 시작 위치
    sort: str = 'sim' # 정렬 옵션: sim (유사도순), date (날짜순)
) -> List[Dict[str, str]]:
    """네이버 검색 API (웹문서)를 사용하여 검색을 수행합니다."""
    logger.info(f"Naver 검색 실행: query='{query}', num_results={num_results}")
    if not settings.NAVER_CLIENT_ID or not settings.NAVER_CLIENT_SECRET:
        logger.error("Naver Client ID 또는 Secret이 설정되지 않았습니다.")
        return []

    # API 엔드포인트 (웹문서 검색 예시 - 필요시 'news', 'blog' 등으로 변경)
    # 웹문서: webkr.json, 뉴스: news.json, 블로그: blog.json, 책: book.json 등
    api_url = "https://openapi.naver.com/v1/search/webkr.json"

    headers = {
        "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
        'User-Agent': 'Mozilla/5.0 (... 생략 ...)' # User-Agent 추가 가능
    }
    params = {
        "query": query,
        "display": num_results, # 네이버는 display 파라미터 사용 (최대 100)
        "start": start,         # 검색 시작 위치 (최대 1000)
        "sort": sort
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            search_results: List[Dict[str, str]] = []
            if data and "items" in data:
                for item in data["items"]:
                    # 네이버는 title, description 필드에 <b> 태그가 포함될 수 있음
                    title = remove_html_tags(item.get("title"))
                    snippet = remove_html_tags(item.get("description")) # 네이버는 description 사용
                    search_results.append({
                        "title": title,
                        "link": item.get("link"),
                        "snippet": snippet,
                    })
            logger.info(f"Naver 검색 결과 {len(search_results)}개 반환.")
            return search_results

    except httpx.HTTPStatusError as e:
        logger.error(f"Naver 검색 API 오류: Status {e.response.status_code}, Response: {e.response.text}")
        return []
    except Exception as e:
        logger.exception(f"Naver 검색 중 예외 발생: {e}")
        return []