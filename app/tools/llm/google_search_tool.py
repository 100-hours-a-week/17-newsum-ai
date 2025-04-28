# app/agents/collector_agent.py (import 경로 수정)
import logging
from typing import List, Dict

from app.workflows.state import AppState
# 변경된 경로에서 검색 도구 import
from app.tools.llm.google_search_tool import run_google_search
from app.services.llm_server_client import call_llm_api
from app.config.settings import settings

logger = logging.getLogger(__name__)

# Google API 클라이언트 라이브러리는 기본적으로 동기 방식이므로,
# 비동기 환경(FastAPI, LangGraph)에서 사용하려면 별도의 처리가 필요합니다.
# 여기서는 asyncio.to_thread를 사용하여 동기 함수를 비동기적으로 실행합니다.

async def run_google_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Google Custom Search API를 사용하여 웹 검색을 수행하는 도구.

    Args:
        query (str): 검색할 쿼리 문자열.
        num_results (int): 반환할 검색 결과의 최대 개수 (최대 10).

    Returns:
        List[Dict[str, str]]: 검색 결과 목록. 각 결과는 'title', 'link', 'snippet' 키를 가짐.
                               오류 발생 시 빈 리스트 반환.
    """
    logger.info(f"Google 검색 도구 실행: query='{query}', num_results={num_results}")
    if not settings.GOOGLE_API_KEY or not settings.GOOGLE_CSE_ID:
        logger.error("Google API 키 또는 CSE ID가 설정되지 않았습니다.")
        return []

    try:
        # 동기 함수인 google search api 호출을 비동기 이벤트 루프에서 실행
        results = await asyncio.to_thread(
            _execute_google_search_sync,
            query,
            num_results
        )
        return results
    except HttpError as e:
        logger.error(f"Google 검색 API 호출 중 오류 발생: {e}")
        return []
    except Exception as e:
        logger.exception(f"Google 검색 도구 실행 중 예외 발생: {e}")
        return []

def _execute_google_search_sync(query: str, num_results: int) -> List[Dict[str, str]]:
    """Google 검색 API를 동기적으로 호출하는 내부 함수"""
    service = build("customsearch", "v1", developerKey=settings.GOOGLE_API_KEY)
    # API 호출: cse().list() 사용
    response = service.cse().list(
        q=query,
        cx=settings.GOOGLE_CSE_ID,
        num=num_results # 한 번에 가져올 결과 수 (최대 10)
        # 필요한 다른 파라미터 추가 가능 (예: siteSearch 특정 사이트 검색)
    ).execute()

    # 결과 파싱
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

# --- 로컬 테스트용 코드 ---
async def main():
    # 테스트 전에 .env 파일 및 settings.py 설정 필요
    if not settings.GOOGLE_API_KEY or not settings.GOOGLE_CSE_ID:
        print("테스트를 위해 .env 파일에 GOOGLE_API_KEY와 GOOGLE_CSE_ID를 설정해주세요.")
        return

    test_query = "LangChain 한국어"
    print(f"테스트 쿼리: {test_query}")
    results = await run_google_search(test_query, num_results=3)

    if results:
        print("\n검색 결과:")
        for i, result in enumerate(results):
            print(f"--- 결과 {i+1} ---")
            print(f"  제목: {result['title']}")
            print(f"  링크: {result['link']}")
            print(f"  요약: {result['snippet']}")
    else:
        print("\n검색 결과가 없거나 오류가 발생했습니다.")

if __name__ == '__main__':
    # .env 파일 로드를 위해 settings를 먼저 로드하도록 할 수 있음 (config.__init__ 등에서)
    # 여기서는 간단히 asyncio.run 사용
    asyncio.run(main())