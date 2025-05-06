# app/tools/search/google.py
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.settings import settings # 설정 임포트
from app.utils.logger import get_logger # 로거 임포트

logger = get_logger("GoogleSearchTool") # 로거 이름

class GoogleSearchTool:
    """Google API(YouTube 검색, Custom Search)와 상호작용하는 도구입니다."""

    # API 엔드포인트 정의
    Youtube_URL = "https://www.googleapis.com/youtube/v3/search"
    CSE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """
        GoogleSearchTool 초기화.

        Args:
            session (Optional[aiohttp.ClientSession]): 외부에서 관리되는 aiohttp 세션 (선택 사항).
                                                     제공되지 않으면 내부적으로 세션을 생성합니다.
        """
        # 설정에서 API 키, CSE ID, 타겟 도메인 등 로드
        self.api_key = settings.GOOGLE_API_KEY
        self.cse_id = settings.GOOGLE_CSE_ID
        self.target_community_domains = settings.TARGET_COMMUNITY_DOMAINS
        self.http_timeout = settings.TOOL_HTTP_TIMEOUT # HTTP 타임아웃 설정값

        # aiohttp 세션 관리
        self._session = session
        self._created_session = False # 내부 생성 여부 플래그
        if self._session is None:
            # 외부 세션 없으면 내부적으로 생성
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.http_timeout))
            self._created_session = True
            logger.info("GoogleSearchTool을 위한 새로운 aiohttp ClientSession 생성됨.")

        # API 키 누락 시 경고 로깅
        if not self.api_key:
            logger.warning("settings에 GOOGLE_API_KEY가 설정되지 않았습니다. Google 검색이 실패합니다.")
        # CSE ID는 CSE 검색에만 필요
        if not self.cse_id:
             logger.warning("settings에 GOOGLE_CSE_ID가 설정되지 않았습니다. 블로그 및 커뮤니티 검색이 실패합니다.")
        # 타겟 도메인은 커뮤니티 검색에만 필요
        if not self.target_community_domains:
             logger.warning("settings에 TARGET_COMMUNITY_DOMAINS가 설정되지 않았습니다. 커뮤니티 검색이 실패합니다.")

    @retry(
        stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS), # 설정된 재시도 횟수 사용
        wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN, max=settings.TOOL_RETRY_WAIT_MAX), # 지수적 대기
        # 네트워크 오류 또는 타임아웃 시 재시도
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True # 모든 재시도 실패 시 예외 다시 발생
    )
    async def _google_api_request(self, url: str, params: Dict[str, Any], trace_id: str) -> Optional[Dict[str, Any]]:
        """Google API에 요청을 보내는 내부 헬퍼 메서드."""
        # API 키 또는 세션 없으면 요청 불가
        if not self.api_key: return None
        if not self._session or self._session.closed:
            logger.error("aiohttp 세션이 닫혔거나 초기화되지 않았습니다.", extra={'trace_id': trace_id})
            return None

        extra_log_data = {'trace_id': trace_id, 'url': url, 'params': params} # 로깅용 추가 데이터
        logger.debug("Google API 요청 실행 중...", extra=extra_log_data)

        try:
            # GET 요청 실행
            async with self._session.get(url, params=params) as response:
                response_text = await response.text() # 오류 로깅 위해 미리 텍스트 읽기
                if response.status == 200: # 성공 시
                    logger.debug(f"Google API 요청 성공 (상태 코드 {response.status})", extra=extra_log_data)
                    try:
                        # JSON 파싱 시도
                        return await response.json()
                    except aiohttp.ContentTypeError: # JSON 아닌 응답 처리
                         logger.error(f"Google API 응답이 유효한 JSON이 아닙니다. 상태: {response.status}. 응답: {response_text[:200]}...", extra=extra_log_data)
                         return None
                elif response.status == 429: # Rate Limit 초과 시
                    logger.warning(f"Google API Rate Limit 초과 (상태 코드 429). 응답: {response_text[:200]}...", extra=extra_log_data)
                    # 재시도 위해 예외 발생시킴
                    raise aiohttp.ClientResponseError(response.request_info, response.history, status=response.status, message="Rate Limit Exceeded", headers=response.headers)
                else: # 기타 HTTP 오류 시
                    logger.error(f"Google API 요청 실패 (상태 코드 {response.status}). 응답: {response_text[:200]}...", extra=extra_log_data)
                    # 오류 상태 코드에 대한 예외 발생 (4xx, 5xx)
                    response.raise_for_status()
                    return None # raise_for_status가 예외 발생시키므로 도달하지 않음
        except asyncio.TimeoutError as e: # 타임아웃 발생 시
            logger.error(f"Google API 요청 시간 초과 ({self.http_timeout}초).", extra=extra_log_data)
            raise e # 재시도 위해 예외 발생
        except aiohttp.ClientError as e: # 클라이언트 측 오류 발생 시 (네트워크 등)
            logger.error(f"Google API 요청 클라이언트 오류: {e}", exc_info=True, extra=extra_log_data)
            raise e # 재시도 위해 예외 발생
        except Exception as e: # 예상치 못한 오류 처리
             logger.error(f"Google API 요청 중 예상치 못한 오류 발생: {e}", exc_info=True, extra=extra_log_data)
             raise e # 예상치 못한 오류도 재시도 고려

    async def search_news(self, keyword: str, max_results: int, trace_id: str) -> List[Dict[str, str]]:
        """Google Custom Search API를 사용하여 뉴스 기사를 검색합니다."""
        # CSE ID 없으면 검색 불가
        if not self.cse_id:
            logger.warning("settings에 GOOGLE_CSE_ID가 설정되지 않았습니다. 뉴스 검색을 수행할 수 없습니다.", extra={'trace_id': trace_id})
            return []

        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": keyword,
            "num": min(max_results, 10), # API 호출당 최대 10개 제한
            "sort": "date" # 날짜순 정렬
            # 뉴스 검색을 위한 다른 파라미터 추가 가능 (예: dateRestrict)
        }
        data = await self._google_api_request(self.CSE_SEARCH_URL, params, trace_id) # API 요청
        results = []
        if data and "items" in data: # 결과 데이터 처리
            for item in data["items"]:
                 # 결과 목록에 추가 (URL, 제목, 스니펫)
                 results.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "Google News" # 출처 명시
                 })
        logger.info(f"CSE를 통해 '{keyword}'에 대해 {len(results)}개의 뉴스 결과를 찾았습니다.", extra={'trace_id': trace_id})
        return results

    async def search_youtube_videos(self, keyword: str, max_results: int, trace_id: str) -> List[Dict[str, str]]:
        """YouTube Data API를 사용하여 유튜브 비디오를 검색합니다."""
        params = {
            "key": self.api_key,
            "part": "snippet", # 필요한 정보 부분(snippet 등)
            "q": keyword, # 검색어
            "maxResults": min(max_results, 50), # API 최대 50개 제한 적용
            "type": "video", # 비디오 타입만 검색
            "order": "relevance" # 정렬 기준 (relevance, date, rating, viewCount 등)
        }
        data = await self._google_api_request(self.Youtube_URL, params, trace_id) # API 요청
        results = []
        if data and "items" in data: # 결과 데이터 처리
            for item in data["items"]:
                video_id = item.get("id", {}).get("videoId") # 비디오 ID 추출
                snippet = item.get("snippet", {}) # 비디오 정보(snippet) 추출
                if video_id and snippet:
                    # 표준 유튜브 시청 URL 사용
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    # 결과 목록에 추가
                    results.append({
                        "url": video_url,
                        "title": snippet.get("title", ""), # 제목
                        "description": snippet.get("description", ""), # 설명
                        "published_at": snippet.get("publishedAt", ""), # 게시일
                        "channel_title": snippet.get("channelTitle", ""), # 채널명
                        "video_id": video_id, # 비디오 ID
                        "source": "YouTube" # 출처 명시
                    })
        logger.info(f"'{keyword}'에 대해 {len(results)}개의 YouTube 비디오를 찾았습니다.", extra={'trace_id': trace_id})
        return results

    async def search_blogs_via_cse(self, keyword: str, max_results: int, trace_id: str) -> List[Dict[str, str]]:
        """Google Custom Search를 사용하여 잠재적인 블로그 콘텐츠를 검색합니다."""
        # CSE ID 없으면 검색 불가
        if not self.cse_id:
            logger.warning("settings에 GOOGLE_CSE_ID가 설정되지 않았습니다. 블로그 검색을 수행할 수 없습니다.", extra={'trace_id': trace_id})
            return []

        params = {
            "key": self.api_key,
            "cx": self.cse_id, # Custom Search Engine ID
            "q": keyword, # 검색어 (노드에서 필요 시 "블로그", "후기" 등 추가 가능)
            "num": min(max_results, 10) # API 호출당 최대 10개 제한
        }
        data = await self._google_api_request(self.CSE_SEARCH_URL, params, trace_id) # API 요청
        results = []
        if data and "items" in data: # 결과 데이터 처리
            for item in data["items"]:
                # 결과 목록에 추가 (URL, 제목, 스니펫)
                results.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "GoogleCSE_BlogSearch" # 잠재적 출처 표시
                })
        logger.info(f"CSE를 통해 '{keyword}'에 대해 {len(results)}개의 잠재적 블로그 결과를 찾았습니다.", extra={'trace_id': trace_id})
        return results

    async def search_communities_via_cse(self, keyword: str, max_results: int, trace_id: str) -> List[Dict[str, str]]:
        """Google Custom Search를 사용하여 대상 온라인 커뮤니티를 검색합니다."""
        # CSE ID 또는 대상 도메인 없으면 검색 불가
        if not self.cse_id:
            logger.warning("settings에 GOOGLE_CSE_ID가 설정되지 않았습니다. 커뮤니티 검색을 수행할 수 없습니다.", extra={'trace_id': trace_id})
            return []
        if not self.target_community_domains:
             logger.warning("settings에 TARGET_COMMUNITY_DOMAINS가 설정되지 않았습니다. 커뮤니티 검색을 수행할 수 없습니다.", extra={'trace_id': trace_id})
             return []

        # site: 검색 쿼리 구성
        site_query = " OR ".join([f"site:{domain.strip()}" for domain in self.target_community_domains])
        full_query = f"{keyword} ({site_query})" # 키워드와 site: 쿼리 결합

        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": full_query, # 최종 쿼리 사용
            "num": min(max_results, 10) # API 호출당 최대 10개 제한
        }
        data = await self._google_api_request(self.CSE_SEARCH_URL, params, trace_id) # API 요청
        results = []
        if data and "items" in data: # 결과 데이터 처리
            for item in data["items"]:
                # 결과 목록에 추가 (URL, 제목, 스니펫)
                results.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "GoogleCSE_CommunitySearch" # 출처 표시
                })
        logger.info(f"CSE를 통해 '{keyword}'에 대해 {len(results)}개의 커뮤니티 결과를 찾았습니다.", extra={'trace_id': trace_id})
        return results

    async def get_youtube_details(self, video_id: str, trace_id: str) -> Optional[Dict[str, Any]]:
        """YouTube Data API를 사용하여 비디오 상세 정보(snippet, statistics)를 가져옵니다."""
        if not self.api_key: return None  # API 키 없으면 불가
        extra_log_data = {'trace_id': trace_id, 'video_id': video_id}
        logger.debug("API를 통해 YouTube 비디오 상세 정보 가져오는 중...", extra=extra_log_data)

        # 비디오 상세 정보 요청 엔드포인트
        video_url = "https://www.googleapis.com/youtube/v3/videos"
        # snippet과 statistics 파트 요청
        params = {"key": self.api_key, "part": "snippet,statistics", "id": video_id}

        # 내부 _google_api_request 사용 (재시도 로직 포함됨)
        data = await self._google_api_request(video_url, params, trace_id)

        if data and data.get("items"):  # 결과가 있고 items 리스트가 존재하면
            item = data["items"][0]  # 첫 번째 아이템 사용
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            # 필요한 정보 추출하여 반환
            return {
                "text": snippet.get("description", ""),  # 설명을 주요 텍스트로 간주
                "title": snippet.get("title", ""),  # 제목 포함
                "author": snippet.get("channelTitle", ""),  # 채널 제목을 작성자로 간주
                "timestamp": snippet.get("publishedAt", None),  # 게시일
                # 좋아요 수는 비공개일 수 있음
                "likes": int(stats.get("likeCount", 0)) if stats.get("likeCount") is not None else 0,
                "raw_data": item  # 원본 데이터 저장
            }
        else:
            logger.warning("API를 통해 YouTube 비디오 상세 정보를 찾을 수 없음", extra=extra_log_data)
            return None

    async def close(self):
        """내부적으로 생성된 aiohttp ClientSession을 닫습니다."""
        # 내부 생성 플래그 확인 및 세션 상태 확인 후 close 호출
        if self._session and self._created_session and not self._session.closed:
            await self._session.close()
            logger.info("GoogleSearchTool 내부에서 생성된 aiohttp ClientSession을 닫았습니다.")
        elif self._session and not self._created_session:
             # 외부에서 주입된 세션은 여기서 닫지 않음
             logger.debug("외부에서 주입된 aiohttp 세션은 GoogleSearchTool에서 닫지 않습니다.")