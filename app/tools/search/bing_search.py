# # app/tools/search/bing_search.py
# import logging
# import asyncio
# from typing import List, Dict, Optional
# # LangChain 커뮤니티의 Bing Search Wrapper 사용 (langchain-community 설치 필요)
# # pip install langchain-community
# from langchain_community.utilities import BingSearchAPIWrapper
# from app.config.settings import settings
#
# logger = logging.getLogger(__name__)
#
# # LangChain Wrapper는 동기 방식이므로 asyncio.to_thread 필요
# async def run_bing_search(
#     query: str,
#     num_results: int = settings.DEFAULT_NUM_RESULTS,
#     market: str = settings.SEARCH_MARKET
# ) -> List[Dict[str, str]]:
#     """Bing Web Search API를 사용하여 웹 검색을 수행합니다 (LangChain Wrapper 사용)."""
#     logger.info(f"Bing 검색 실행: query='{query}', num_results={num_results}, market={market}")
#     if not settings.BING_SUBSCRIPTION_KEY:
#         logger.error("Bing 구독 키가 설정되지 않았습니다.")
#         return []
#
#     try:
#         # LangChain Wrapper 인스턴스 생성
#         # wrapper = BingSearchAPIWrapper(
#         #     bing_subscription_key=settings.BING_SUBSCRIPTION_KEY,
#         #     bing_search_url=settings.BING_SEARCH_URL,
#         #     k=num_results,
#         #     search_kwargs={'mkt': market} # market 등 추가 파라미터 전달
#         # )
#         # run 메서드는 결과 요약 문자열만 반환하므로, results 메서드 사용
#
#         # 동기 메서드인 results를 비동기로 실행
#         results_list = await asyncio.to_thread(
#             _execute_bing_search_sync,
#             query,
#             num_results,
#             market,
#             settings.BING_SUBSCRIPTION_KEY,
#             settings.BING_SEARCH_URL
#         )
#         logger.info(f"Bing 검색 결과 {len(results_list)}개 반환.")
#         return results_list
#
#     except Exception as e:
#         logger.exception(f"Bing 검색 중 예외 발생: {e}")
#         return []
#
# def _execute_bing_search_sync(query: str, num_results: int, market: str, api_key:str, api_url:str) -> List[Dict[str, str]]:
#      wrapper = BingSearchAPIWrapper(
#             bing_subscription_key=api_key,
#             bing_search_url=api_url,
#             k=num_results,
#             search_kwargs={'mkt': market}
#         )
#      # results 메서드는 [{'snippet': ..., 'title': ..., 'link': ...}] 형태의 리스트 반환
#      return wrapper.results(query, num_results)