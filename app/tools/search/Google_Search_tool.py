# ai/app/tools/search/Google Search_tool.py

import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.settings import Settings
from app.utils.logger import get_logger, summarize_for_logging

settings = Settings()
logger = get_logger("GoogleSearchTool")


class GoogleSearchTool:
    """
    Google API(YouTube 검색, Custom Search Engine)와 상호작용하는 도구입니다.
    (업그레이드됨) 일반 웹 검색, 뉴스 검색, 특정 사이트 검색 시 고급 검색 옵션(dateRestrict, fileType 등) 지원.
    """

    Youtube_URL = "https://www.googleapis.com/youtube/v3/search"
    YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
    CSE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

    # Google CSE에서 지원하는 주요 파라미터 중 N04에서 활용할 가능성이 있는 것들
    SUPPORTED_CSE_PARAMS = ["dateRestrict", "fileType", "rights", "safe", "lr", "cr", "filter"]

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.api_key = settings.GOOGLE_API_KEY
        self.cse_id = settings.GOOGLE_CSE_ID
        self.target_community_domains = settings.TARGET_COMMUNITY_DOMAINS
        self.http_timeout = settings.TOOL_HTTP_TIMEOUT

        self._session = session
        self._created_session = False
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.http_timeout))
            self._created_session = True
            logger.info("GoogleSearchTool을 위한 새로운 aiohttp ClientSession 생성됨.")

        if not self.api_key:
            logger.warning("settings에 GOOGLE_API_KEY가 설정되지 않았습니다. Google API 호출이 실패합니다.")
        if not self.cse_id:
            logger.warning("settings에 GOOGLE_CSE_ID가 설정되지 않았습니다. CSE 기반 검색이 실패합니다.")
        if not self.target_community_domains:
            logger.warning("settings에 TARGET_COMMUNITY_DOMAINS가 설정되지 않았습니다. search_communities_via_cse의 기본 동작이 제한됩니다.")

    @retry(
        stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN, max=settings.TOOL_RETRY_WAIT_MAX),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True
    )
    async def _google_api_request(self, url: str, params: Dict[str, Any], trace_id: str) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            logger.error("Google API Key가 설정되지 않아 요청을 보낼 수 없습니다.", extra={'trace_id': trace_id})
            return None
        if not self._session or self._session.closed:
            # 세션이 닫힌 경우, 내부 생성 세션이었다면 새로 만들기 시도
            if self._created_session:
                logger.warning("내부 aiohttp 세션이 닫혀있어 새로 생성합니다.", extra={'trace_id': trace_id})
                self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.http_timeout))
            else:  # 외부 주입 세션이 닫힌 경우는 에러
                logger.error("외부 주입 aiohttp 세션이 닫혔거나 초기화되지 않았습니다.", extra={'trace_id': trace_id})
                return None

        if 'key' not in params:
            params['key'] = self.api_key

        # 빈 값 파라미터 제거 (API 오류 방지)
        final_params = {k: v for k, v in params.items() if v is not None and v != ""}

        extra_log_data = {'trace_id': trace_id, 'url': url,
                          'params': {k: v for k, v in final_params.items() if k != 'key'}}
        logger.debug(f"Google API 요청 실행 중... Params: {summarize_for_logging(final_params, exclude_keys=['key'])}",
                     extra=extra_log_data)

        try:
            async with self._session.get(url, params=final_params) as response:
                response_text = await response.text()
                if response.status == 200:
                    logger.debug(f"Google API 요청 성공 (상태 코드 {response.status})", extra=extra_log_data)
                    try:
                        return await response.json()
                    except aiohttp.ContentTypeError:
                        logger.error(
                            f"Google API 응답이 유효한 JSON이 아닙니다. 상태: {response.status}. 응답: {response_text[:200]}...",
                            extra=extra_log_data)
                        return None
                elif response.status == 429:
                    logger.warning(f"Google API Rate Limit 초과 (상태 코드 429). 응답: {response_text[:200]}...",
                                   extra=extra_log_data)
                    raise aiohttp.ClientResponseError(response.request_info, response.history, status=response.status,
                                                      message="Rate Limit Exceeded", headers=response.headers)
                else:
                    logger.error(f"Google API 요청 실패 (상태 코드 {response.status}). 응답: {response_text[:200]}...",
                                 extra=extra_log_data)
                    response.raise_for_status()
                    return None
        except asyncio.TimeoutError as e:
            logger.error(f"Google API 요청 시간 초과 ({self.http_timeout}초).", extra=extra_log_data)
            raise e
        except aiohttp.ClientError as e:
            logger.error(f"Google API 요청 클라이언트 오류: {e}", exc_info=False,
                         extra=extra_log_data)  # exc_info=False로 변경 (재시도 시 너무 많은 로그 방지)
            raise e
        except Exception as e:
            logger.error(f"Google API 요청 중 예상치 못한 오류 발생: {e}", exc_info=True, extra=extra_log_data)
            raise e

    def _process_cse_results(self, data: Optional[Dict[str, Any]], source_label: str) -> List[Dict[str, str]]:
        results = []
        if data and "items" in data:
            for item in data["items"]:
                results.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": source_label
                })
        return results

    async def search_web_via_cse(
            self, keyword: str, max_results: int, trace_id: str, **kwargs: Any
    ) -> List[Dict[str, str]]:  # **kwargs 추가
        """Google Custom Search API를 사용하여 일반 웹 검색을 수행합니다 (고급 옵션 지원)."""
        if not self.cse_id:
            logger.warning("CSE ID가 없어 일반 웹 검색을 수행할 수 없습니다.", extra={'trace_id': trace_id})
            return []

        params = {
            "cx": self.cse_id,
            "q": keyword,
            "num": min(max_results, 10)
        }
        # --- 고급 검색 옵션 추가 ---
        for p_name in self.SUPPORTED_CSE_PARAMS:
            if p_name in kwargs and kwargs[p_name] is not None:
                params[p_name] = kwargs[p_name]
        # --------------------------
        data = await self._google_api_request(self.CSE_SEARCH_URL, params, trace_id)
        results = self._process_cse_results(data, "GoogleCSE_WebSearch")
        logger.info(f"CSE 웹 검색: '{keyword}' (옵션: {kwargs})에 대해 {len(results)}개의 결과 찾음.", extra={'trace_id': trace_id})
        return results

    async def search_news_via_cse(  # 이름 변경: search_news -> search_news_via_cse
            self, keyword: str, max_results: int, trace_id: str, **kwargs: Any
    ) -> List[Dict[str, str]]:  # **kwargs 추가
        """Google Custom Search API를 사용하여 뉴스 기사를 검색합니다 (날짜순 정렬, 고급 옵션 지원)."""
        if not self.cse_id:
            logger.warning("CSE ID가 없어 뉴스 검색을 수행할 수 없습니다.", extra={'trace_id': trace_id})
            return []

        params = {
            "cx": self.cse_id,
            "q": keyword,
            "num": min(max_results, 10),
            "sort": "date"  # 뉴스 검색 시 날짜순 정렬 기본값
        }
        # --- 고급 검색 옵션 추가 (dateRestrict 등) ---
        for p_name in self.SUPPORTED_CSE_PARAMS:
            if p_name in kwargs and kwargs[p_name] is not None:
                # sort는 이미 설정되어 있으므로 덮어쓰지 않도록 할 수 있으나, kwargs 우선 적용 가능
                if p_name == "sort" and "sort" in params and kwargs[p_name] != params["sort"]:
                    logger.debug(f"Overriding default sort '{params['sort']}' with '{kwargs[p_name]}'",
                                 extra={'trace_id': trace_id})
                params[p_name] = kwargs[p_name]
        # -----------------------------------------
        data = await self._google_api_request(self.CSE_SEARCH_URL, params, trace_id)
        results = self._process_cse_results(data, "GoogleCSE_NewsSearch")
        logger.info(f"CSE 뉴스 검색: '{keyword}' (옵션: {kwargs})에 대해 {len(results)}개의 결과 찾음.", extra={'trace_id': trace_id})
        return results

    async def search_specific_sites_via_cse(
            self, keyword: str, sites: List[str], max_results: int, trace_id: str, **kwargs: Any
    ) -> List[Dict[str, str]]:  # **kwargs 추가
        """Google Custom Search를 사용하여 지정된 사이트 목록 내에서 검색합니다 (고급 옵션 지원)."""
        if not self.cse_id:
            logger.warning("CSE ID가 없어 특정 사이트 검색을 수행할 수 없습니다.", extra={'trace_id': trace_id})
            return []
        if not sites:
            logger.warning("검색할 대상 사이트 목록이 비어 있습니다.", extra={'trace_id': trace_id})
            return []

        site_query_parts = [f"site:{site.strip()}" for site in sites if site.strip()]
        if not site_query_parts:
            logger.warning("유효한 검색 대상 사이트가 없습니다.", extra={'trace_id': trace_id})
            return []

        # Google 검색어 구문: keyword (site:example.com OR site:another.org)
        # siteSearch 파라미터를 사용할 수도 있지만, q에 직접 넣는 것이 더 유연할 수 있음
        # 여기서는 기존 로직 유지 (q에 site: 결합)
        site_restriction_string = " OR ".join(site_query_parts)
        full_query = f"{keyword} ({site_restriction_string})"

        # siteSearch 파라미터 사용 방식 (선택적, CSE 설정에 따라 동작 상이할 수 있음)
        # params["siteSearch"] = "|".join(sites) # siteSearchFilter 파라미터와 함께 사용
        # params["siteSearchFilter"] = "i" # include (i) or exclude (e)

        params = {
            "cx": self.cse_id,
            "q": full_query,
            "num": min(max_results, 10)
        }
        # --- 고급 검색 옵션 추가 ---
        for p_name in self.SUPPORTED_CSE_PARAMS:
            if p_name in kwargs and kwargs[p_name] is not None:
                params[p_name] = kwargs[p_name]
        # --------------------------
        data = await self._google_api_request(self.CSE_SEARCH_URL, params, trace_id)
        source_label = f"GoogleCSE_SiteSearch[{','.join(sites)}]"
        results = self._process_cse_results(data, source_label)
        logger.info(f"CSE 특정 사이트 검색: '{keyword}' (사이트: {sites}, 옵션: {kwargs})에 대해 {len(results)}개의 결과 찾음.",
                    extra={'trace_id': trace_id})
        return results

    async def search_youtube_videos(
            self, keyword: str, max_results: int, trace_id: str, **kwargs: Any  # kwargs 추가 (YouTube API도 일부 지원)
    ) -> List[Dict[str, str]]:
        """YouTube Data API를 사용하여 유튜브 비디오를 검색합니다 (일부 고급 옵션 지원 가능성)."""
        params = {
            "part": "snippet",
            "q": keyword,
            "maxResults": min(max_results, 50),
            "type": "video",
            "order": "relevance"  # 기본값
        }
        # YouTube API에서 지원하는 추가 파라미터 예시 (필요 시 N04에서 전달)
        youtube_supported_params = ["videoDuration", "publishedAfter", "publishedBefore", "regionCode",
                                    "relevanceLanguage"]
        for p_name in youtube_supported_params:
            if p_name in kwargs and kwargs[p_name] is not None:
                params[p_name] = kwargs[p_name]
        if "order" in kwargs and kwargs["order"]:  # 정렬 순서 변경 가능
            params["order"] = kwargs["order"]
        # -----------------------------------------------------------
        data = await self._google_api_request(self.Youtube_URL, params, trace_id)
        results = []
        if data and "items" in data:
            for item in data["items"]:
                video_id = item.get("id", {}).get("videoId")
                snippet = item.get("snippet", {})
                if video_id and snippet:
                    # 유튜브 URL 형식 일관성 유지 (youtu.be 대신 표준 시청 URL)
                    # video_url = f"https://www.youtube.com/watch?v={video_id}"
                    # 이전 https://www.youtube.com/watch?v={video_id} 형식 유지
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    results.append({
                        "url": video_url,
                        "title": snippet.get("title", ""),
                        "description": snippet.get("description", ""),
                        "published_at": snippet.get("publishedAt", ""),
                        "channel_title": snippet.get("channelTitle", ""),
                        "video_id": video_id,
                        "source": "YouTube"
                    })
        logger.info(f"YouTube 검색: '{keyword}' (옵션: {kwargs})에 대해 {len(results)}개의 비디오 찾음.",
                    extra={'trace_id': trace_id})
        return results

    async def search_blogs_via_cse(
            self, keyword: str, max_results: int, trace_id: str, **kwargs: Any  # **kwargs 추가
    ) -> List[Dict[str, str]]:
        """Google Custom Search를 사용하여 잠재적인 블로그 콘텐츠를 검색합니다 (고급 옵션 지원)."""
        if not self.cse_id:
            logger.warning("CSE ID가 없어 블로그 검색을 수행할 수 없습니다.", extra={'trace_id': trace_id})
            return []

        params = {
            "cx": self.cse_id,
            "q": keyword,  # N03에서 "블로그 후기" 등을 키워드에 포함시켰다고 가정
            "num": min(max_results, 10)
        }
        # --- 고급 검색 옵션 추가 ---
        for p_name in self.SUPPORTED_CSE_PARAMS:
            if p_name in kwargs and kwargs[p_name] is not None:
                params[p_name] = kwargs[p_name]
        # --------------------------
        data = await self._google_api_request(self.CSE_SEARCH_URL, params, trace_id)
        results = self._process_cse_results(data, "GoogleCSE_BlogSearch")
        logger.info(f"CSE 블로그 검색: '{keyword}' (옵션: {kwargs})에 대해 {len(results)}개의 결과 찾음.", extra={'trace_id': trace_id})
        return results

    async def search_communities_via_cse(
            self, keyword: str, max_results: int, trace_id: str, **kwargs: Any  # **kwargs 추가
    ) -> List[Dict[str, str]]:
        """Google Custom Search를 사용하여 설정에 정의된 대상 온라인 커뮤니티를 검색합니다 (고급 옵션 지원)."""
        if not self.target_community_domains:
            logger.warning("설정에 TARGET_COMMUNITY_DOMAINS가 없어 커뮤니티 검색을 수행할 수 없습니다.", extra={'trace_id': trace_id})
            # 대상 도메인이 없으면 일반 웹 검색으로 fallback 하거나 빈 결과 반환
            # 여기서는 빈 결과 반환. N04에서 이 경우 다른 도구를 선택하거나 일반 웹 검색으로 대체할 수 있음.
            return []

        # search_specific_sites_via_cse 함수 재사용 (이 함수가 **kwargs를 처리함)
        return await self.search_specific_sites_via_cse(
            keyword=keyword,
            sites=self.target_community_domains,  # 설정 파일에서 가져온 도메인 사용
            max_results=max_results,
            trace_id=trace_id,
            **kwargs  # 고급 검색 옵션 전달
        )

    async def get_youtube_details(self, video_id: str, trace_id: str) -> Optional[Dict[str, Any]]:
        # (이 함수는 N04에서 직접 사용되지 않으므로 변경 없음, 필요시 유사하게 **kwargs 추가 가능)
        if not self.api_key:
            logger.error("API 키가 없어 YouTube 상세 정보를 가져올 수 없습니다.", extra={'trace_id': trace_id, 'video_id': video_id})
            return None
        extra_log_data = {'trace_id': trace_id, 'video_id': video_id}
        logger.debug("API를 통해 YouTube 비디오 상세 정보 가져오는 중...", extra=extra_log_data)

        params = {"part": "snippet,statistics", "id": video_id}
        data = await self._google_api_request(self.YOUTUBE_VIDEOS_URL, params, trace_id)

        if data and data.get("items"):
            item = data["items"][0]
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            details = {
                "text": snippet.get("description", ""),
                "title": snippet.get("title", ""),
                "author": snippet.get("channelTitle", ""),
                "timestamp": snippet.get("publishedAt", None),
                "view_count": int(stats.get("viewCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "likes": int(stats.get("likeCount", 0)) if stats.get("likeCount") is not None else None,
                "raw_data": item
            }
            logger.debug(f"YouTube 비디오 상세 정보 가져오기 성공: {video_id}", extra=extra_log_data)
            return details
        else:
            logger.warning(f"API를 통해 YouTube 비디오 상세 정보를 찾을 수 없음: {video_id}", extra=extra_log_data)
            return None

    async def close(self):
        if self._session and self._created_session and not self._session.closed:
            await self._session.close()
            logger.info("GoogleSearchTool 내부에서 생성된 aiohttp ClientSession을 닫았습니다.")
        elif self._session and not self._created_session:
            logger.debug("외부에서 주입된 aiohttp 세션은 GoogleSearchTool에서 닫지 않습니다.")


# --- 예제 사용법 (main 함수) ---
# (main 함수는 변경 없음, 필요시 각 검색 함수 호출 시 dateRestrict, fileType 등 추가 옵션 테스트 가능)
async def main():
    if not settings.GOOGLE_API_KEY or not settings.GOOGLE_CSE_ID:
        print("경고: GOOGLE_API_KEY 또는 GOOGLE_CSE_ID 환경 변수가 설정되지 않았습니다.")
        # return

    search_tool = GoogleSearchTool()
    trace_id = "test-advanced-search-123"

    try:
        print("\n--- 일반 웹 검색 (CSE) - 고급 옵션 테스트 ---")
        web_results_advanced = await search_tool.search_web_via_cse(
            "파이썬 비동기 프로그래밍 filetype:pdf",  # 쿼리에 직접 넣는 방식 테스트
            max_results=2,
            trace_id=trace_id
        )
        for res in web_results_advanced: print(f"  - [{res['source']}] {res['title']}: {res['url']}")

        web_results_params = await search_tool.search_web_via_cse(
            "머신러닝 기초",
            max_results=2,
            trace_id=trace_id,
            fileType="pdf",  # 파라미터 방식 테스트
            dateRestrict="y1"  # 지난 1년
        )
        print("\n--- 일반 웹 검색 (CSE) - 파라미터 방식 고급 옵션 ---")
        for res in web_results_params: print(f"  - [{res['source']}] {res['title']}: {res['url']}")

        print("\n--- 뉴스 검색 (CSE) - 고급 옵션 테스트 ---")
        news_results_advanced = await search_tool.search_news_via_cse(  # 이름 변경됨
            "인공지능 윤리",
            max_results=2,
            trace_id=trace_id,
            dateRestrict="m1"  # 지난 1개월
        )
        for res in news_results_advanced: print(f"  - [{res['source']}] {res['title']}: {res['url']}")

        print("\n--- 특정 사이트 검색 (CSE) - 고급 옵션 테스트 ---")
        sites_to_search_adv = ["arxiv.org"]
        site_results_advanced = await search_tool.search_specific_sites_via_cse(
            "large language model survey",
            sites=sites_to_search_adv,
            max_results=2,
            trace_id=trace_id,
            fileType="pdf"
        )
        for res in site_results_advanced: print(f"  - [{res['source']}] {res['title']}: {res['url']}")

        # (YouTube, Blog, Community 검색 테스트는 기존과 유사하게 진행 가능, 필요시 kwargs 추가)

    except Exception as e:
        logger.error(f"메인 실행 중 오류 발생: {e}", exc_info=True, extra={'trace_id': trace_id})
    finally:
        await search_tool.close()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            loop.run_until_complete(main())
        else:
            asyncio.run(main())
    except RuntimeError as e:
        if "Cannot run the event loop while another loop is running" not in str(e) and \
                "Nesting asyncio event loops is not supported" not in str(e):  # nest_asyncio가 적용된 경우 이 오류는 무시
            raise e