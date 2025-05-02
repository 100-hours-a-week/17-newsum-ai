# app/nodes/06_opinion_scraper_node.py

import asyncio
import random
import re
import traceback
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime, timezone
from collections import defaultdict

# --- 리팩토링된 임포트 ---
from app.utils.logger import get_logger
from app.workflows.state import ComicState
# 필요한 Tool 임포트 (경로 및 파일명 확인)
from app.tools.social.twitter import TwitterTool
from app.tools.social.reddit import RedditTool
from app.tools.search.google import GoogleSearchTool
from app.tools.scraping.selenium_scraper import SeleniumScraperTool # 신규 Selenium 도구
from app.config.settings import settings

# 로거 설정
logger = get_logger("OpinionScraperNode")

class OpinionScraperNode:
    """
    (리팩토링됨) 수집된 의견 URL에 대해 상세 콘텐츠(텍스트, 작성자, 시간, 좋아요 등)를 스크랩합니다.
    API 우선 접근 방식을 사용하며, 실패 또는 해당 없을 시 Selenium 기반 웹 스크래핑으로 대체합니다.
    주입된 도구들을 사용하여 실제 API 호출 및 스크래핑 작업을 위임합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["opinion_urls", "trace_id", "used_links", "config", "processing_stats"]
    outputs: List[str] = ["opinions_raw", "used_links", "processing_stats", "error_message"]

    def __init__(
        self,
        twitter_tool: Optional[TwitterTool] = None,
        reddit_tool: Optional[RedditTool] = None,
        google_tool: Optional[GoogleSearchTool] = None,
        selenium_tool: Optional[SeleniumScraperTool] = None # Selenium 도구 주입
    ):
        """
        OpinionScraperNode 초기화. API 및 스크래핑 도구를 주입받습니다.

        Args:
            twitter_tool (Optional[TwitterTool]): 트위터 상호작용 도구.
            reddit_tool (Optional[RedditTool]): 레딧 상호작용 도구.
            google_tool (Optional[GoogleSearchTool]): 구글 검색 도구 (주로 YouTube API용).
            selenium_tool (Optional[SeleniumScraperTool]): Selenium 웹 스크래핑 도구.
        """
        self.twitter_tool = twitter_tool
        self.reddit_tool = reddit_tool
        self.google_tool = google_tool
        self.selenium_tool = selenium_tool

        # 필수 도구 누락 시 경고 로깅
        if not self.selenium_tool:
            logger.warning("SeleniumScraperTool이 주입되지 않았습니다. 웹 스크래핑 대체 기능이 비활성화됩니다.")
        # API 도구 누락 시 해당 API 우선 접근 불가
        if not self.twitter_tool: logger.warning("TwitterTool이 주입되지 않아 트위터 API 우선 접근이 비활성화됩니다.")
        if not self.reddit_tool: logger.warning("RedditTool이 주입되지 않아 레딧 API 우선 접근이 비활성화됩니다.")
        if not self.google_tool: logger.warning("GoogleSearchTool이 주입되지 않아 유튜브 API 우선 접근이 비활성화됩니다.")


    # --- 플랫폼 식별 및 ID 추출 (노드 레벨 유지) ---
    def _identify_platform(self, url: str) -> str:
        """URL을 기반으로 플랫폼을 식별합니다."""
        try:
            domain = urlparse(url).netloc.lower()
            if 'twitter.com' in domain or 'x.com' in domain: return 'Twitter'
            if 'reddit.com' in domain: return 'Reddit'
            # googleusercontent URL 형식 개선 (Node 04 반환 형식과 일치 필요)
            if 'youtube.com' in domain or 'youtu.be' in domain or 'googleusercontent.com/youtube.com' in url: return 'YouTube'

            # TODO: 설정에서 로드된 커뮤니티 도메인 목록과 비교하여 'Community' 식별 로직 추가
            # community_domains = settings.TARGET_COMMUNITY_DOMAINS
            # if any(domain == comm_domain or domain.endswith('.' + comm_domain) for comm_domain in community_domains):
            #     return 'Community'

            # 블로그 플랫폼 식별 (단순 버전)
            blog_platforms = [ "blogspot.com", "wordpress.com", "medium.com", "tumblr.com",
                               "tistory.com", "blog.naver.com", "brunch.co.kr", "velog.io", "substack.com" ]
            if any(domain == platform or domain.endswith('.' + platform) for platform in blog_platforms):
                return 'Blog'

            # 위 경우에 해당하지 않으면 일반 웹사이트로 간주
            return 'OtherWeb'
        except Exception:
            logger.warning(f"URL '{url}'에서 플랫폼 식별 실패")
            return 'Unknown'

    def _extract_platform_ids(self, url: str, platform: str) -> Dict[str, Optional[str]]:
        """플랫폼에 따라 URL에서 관련 ID를 추출합니다."""
        ids = {"tweet_id": None, "submission_id": None, "comment_id": None, "video_id": None}
        try:
            if platform == 'Twitter':
                # 트윗 ID 추출 (status/statuses 다음 숫자)
                match = re.search(r'/status(?:es)?/(\d+)', url)
                if match: ids['tweet_id'] = match.group(1)
            elif platform == 'Reddit':
                # 레딧 게시글/댓글 ID 추출
                match = re.search(r'/comments/([a-z0-9]+)(?:/[^/]+/([a-z0-9]+))?', url)
                if match:
                     ids['submission_id'] = match.group(1) # 게시글 ID
                     ids['comment_id'] = match.group(2) # 댓글 ID (있을 경우)
            elif platform == 'YouTube':
                 # 유튜브 비디오 ID 추출 (다양한 URL 형식 지원)
                 match_v = re.search(r'[?&]v=([^&]+)', url) # ?v= 형식
                 match_short = re.search(r'youtu\.be/([^?&]+)', url) # youtu.be/ 형식
                 match_embed = re.search(r'/embed/([^?&]+)', url) # /embed/ 형식
                 # googleusercontent 형식 (가정)
                 match_guc = re.search(r'googleusercontent.com/youtube.com/\d([a-zA-Z0-9_-]+)', url)

                 if match_v: ids['video_id'] = match_v.group(1)
                 elif match_short: ids['video_id'] = match_short.group(1)
                 elif match_embed: ids['video_id'] = match_embed.group(1)
                 elif match_guc: ids['video_id'] = match_guc.group(1)

        except Exception as e:
            logger.warning(f"'{platform}' URL '{url}'에서 ID 추출 실패: {e}")
        return ids

    # --- API 및 Selenium 호출 로직 (도구 위임) ---

    async def _fetch_opinion_api(self, platform: str, ids: Dict[str, Optional[str]], trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """플랫폼별 API 도구를 호출하여 상세 정보를 가져옵니다."""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'platform': platform, 'ids': ids}
        api_result = None
        try:
            if platform == 'Twitter' and ids.get('tweet_id') and self.twitter_tool:
                 api_result = await self.twitter_tool.get_tweet_details(ids['tweet_id'], trace_id)
            elif platform == 'Reddit' and self.reddit_tool:
                 if ids.get('comment_id'): # 댓글 ID가 있으면 댓글 정보 우선 조회
                      api_result = await self.reddit_tool.get_comment_details(ids['comment_id'], trace_id)
                 elif ids.get('submission_id'): # 댓글 ID 없고 게시글 ID만 있으면 게시글 정보 조회
                      api_result = await self.reddit_tool.get_submission_details(ids['submission_id'], trace_id)
            elif platform == 'YouTube' and ids.get('video_id') and self.google_tool:
                 api_result = await self.google_tool.get_youtube_details(ids['video_id'], trace_id)

            if api_result:
                 logger.info(f"API 통해 데이터 가져오기 성공", extra=extra_log_data)
            return api_result
        except Exception as api_err:
            # 도구 내부에서 재시도 후에도 실패한 경우
            logger.warning(f"API 호출 실패 (재시도 후): {api_err}", extra=extra_log_data)
            return None # 실패 시 None 반환

    async def _scrape_with_selenium(self, url: str, platform: str, trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """SeleniumScraperTool을 호출하여 웹 스크래핑을 수행합니다."""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, 'platform': platform}
        if not self.selenium_tool:
            logger.warning("Selenium 도구가 없어 스크래핑 불가", extra=extra_log_data)
            return None

        logger.debug(f"Selenium 스크래핑 시도...", extra=extra_log_data)
        try:
            # Selenium 도구의 scrape_url 메서드 호출
            scraped_data = await self.selenium_tool.scrape_url(url, platform, trace_id, comic_id)
            if scraped_data:
                logger.info(f"Selenium 스크래핑 성공", extra=extra_log_data)
            else:
                logger.warning(f"Selenium 스크래핑 결과 없음", extra=extra_log_data)
            return scraped_data
        except Exception as sel_err:
            # Selenium 도구 내부에서 재시도 후에도 실패한 경우
            logger.error(f"Selenium 스크래핑 최종 실패: {sel_err}", exc_info=True, extra=extra_log_data)
            return None # 실패 시 None 반환

    # --- 단일 URL 처리 로직 ---
    async def _process_url(self, url_info: Dict[str, str], config: Dict, trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """
        단일 의견 URL 처리: 플랫폼 식별, ID 추출, API 우선 시도 후 Selenium 대체.
        """
        url = url_info.get('url')
        source = url_info.get('source', 'Unknown') # Node 04에서 전달된 소스 정보
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, 'source': source}
        if not url: return None

        logger.info(f"의견 URL 처리 시작...", extra=extra_log_data)
        platform = self._identify_platform(url) # 플랫폼 식별
        ids = self._extract_platform_ids(url, platform) # ID 추출
        scraped_data: Optional[Dict[str, Any]] = None
        method_used = "Unknown" # 사용된 방법 기록

        # 1. API 우선 시도 (해당 플랫폼 및 ID 존재 시)
        if platform in ['Twitter', 'Reddit', 'YouTube'] and any(v for k, v in ids.items() if k.endswith('_id')):
            logger.debug(f"API 우선 접근 시도 ({platform})", extra=extra_log_data)
            scraped_data = await self._fetch_opinion_api(platform, ids, trace_id, comic_id)
            if scraped_data:
                method_used = f"API ({platform})"

        # 2. Selenium 대체 시도 (API 실패 또는 해당 없음 & Selenium 도구 사용 가능 시)
        if not scraped_data and self.selenium_tool:
             logger.debug(f"API 실패 또는 해당 없음. Selenium 대체 스크래핑 시도 ({platform})", extra=extra_log_data)
             scraped_data = await self._scrape_with_selenium(url, platform, trace_id, comic_id)
             if scraped_data:
                  method_used = f"Selenium ({platform})"
        elif not scraped_data and not self.selenium_tool:
             logger.warning("API 접근 불가 및 Selenium 도구 없음. 스크래핑 불가.", extra=extra_log_data)

        # 3. 결과 포맷팅 및 반환
        if scraped_data:
             # 결과 딕셔너리 구성
             result = {
                 "url": url, # Node 04에서 받은 원본 URL
                 "source": source, # Node 04에서 받은 원본 소스
                 "platform": platform, # 식별된 플랫폼
                 "method": method_used, # 사용된 추출 방법
                 "text": scraped_data.get("text", ""), # 추출된 텍스트
                 "author": scraped_data.get("author"), # 추출된 작성자
                 "timestamp": scraped_data.get("timestamp"), # 추출된 타임스탬프
                 "likes": scraped_data.get("likes", 0), # 추출된 좋아요 수
                 "title": scraped_data.get("title"), # 추출된 제목 (있을 경우)
                 # "raw_data": scraped_data.get("raw_data") # 원본 API/HTML 데이터 (필요 시)
             }
             # 최소 텍스트 길이 검증
             if not result["text"] or len(result["text"]) < settings.MIN_EXTRACTED_TEXT_LENGTH:
                  logger.warning(f"추출된 텍스트가 너무 짧음 ({len(result['text'])}자). 처리 실패 간주. ({method_used})", extra=extra_log_data)
                  return None
             return result
        else:
            # 최종 실패
            logger.warning(f"모든 방법으로 데이터 추출 실패.", extra=extra_log_data)
            return None

    # --- 메인 실행 메서드 ---
    async def execute(self, state: ComicState) -> Dict[str, Any]:
        """주입된 도구를 사용하여 의견 스크래핑 프로세스를 실행합니다."""
        start_time = datetime.now(timezone.utc)
        comic_id = state.comic_id
        trace_id = state.trace_id or comic_id or "unknown_trace"
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}

        logger.info("OpinionScraperNode 실행 시작...", extra=extra_log_data)

        # 상태에서 필요한 정보 로드
        opinion_urls = state.opinion_urls
        config = state.config
        current_used_links = state.used_links
        processing_stats = state.processing_stats

        # 스크랩할 URL 없으면 종료
        if not opinion_urls:
            logger.warning("스크랩할 의견 URL이 없습니다.", extra=extra_log_data)
            return {"opinions_raw": [], "used_links": current_used_links}

        logger.info(f"{len(opinion_urls)}개의 의견 URL 스크래핑 시작...", extra=extra_log_data)

        # 동시성 제한 설정 (설정값 또는 기본값 사용)
        concurrency_limit = config.get('opinion_scraper_concurrency', settings.OPINION_SCRAPER_CONCURRENCY)
        semaphore = asyncio.Semaphore(concurrency_limit)

        # --- 태스크 생성 및 실행 ---
        tasks = []
        for url_info in opinion_urls:
             async def task_with_semaphore(ui): # 세마포 래퍼
                  async with semaphore:
                       return await self._process_url(ui, config, trace_id, comic_id)
             tasks.append(task_with_semaphore(url_info)) # 래핑된 태스크 추가

        # 모든 태스크 병렬 실행 및 결과 취합
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # --- 결과 처리 ---
        opinions_raw: List[Dict[str, Any]] = [] # 최종 결과 목록
        successful_urls = set() # 성공한 원본 URL 추적용
        failed_count = 0 # 실패 카운터

        # gather 결과 순회
        for i, res in enumerate(results):
            original_url = opinion_urls[i].get('url', 'Unknown URL')
            if isinstance(res, dict) and res is not None: # 성공 시
                opinions_raw.append(res)
                successful_urls.add(original_url) # 성공 목록에 원본 URL 추가
            elif isinstance(res, Exception): # gather가 예외 반환 시
                logger.error(f"스크래핑 태스크 실패 (gather 예외): {original_url} | 예외: {res}", exc_info=res, extra=extra_log_data)
                failed_count += 1
            # _process_url에서 None 반환 시 (실패 또는 건너뜀)는 이미 로깅됨

        successful_count = len(opinions_raw)
        total_attempted = len(opinion_urls)
        final_failed_count = total_attempted - successful_count # 최종 실패 수 계산

        logger.info(f"의견 스크래핑 완료. 성공: {successful_count}. 실패/건너<0xEB><0x9C><0x94>: {final_failed_count}.", extra=extra_log_data)

        # --- used_links 상태 업데이트 ---
        updated_used_links = []
        for link in current_used_links:
            url = link.get('url')
            # 이 노드의 입력 URL이었는지 확인
            is_opinion_input = any(op_url.get('url') == url for op_url in opinion_urls)

            if is_opinion_input: # 이 노드에서 처리 시도한 URL인 경우
                 if url in successful_urls: # 성공했으면
                      link['purpose'] = link.get('purpose', '').replace('(Opinion)', '(Scraped Opinion)') # 목적 업데이트
                      link['status'] = "processed" # 상태: 처리됨
                 else: # 실패했으면
                      link['purpose'] = link.get('purpose', '') + " (Opinion Scraping Failed)" # 실패 정보 추가
                      link['status'] = "failed" # 상태: 실패
            updated_used_links.append(link) # 업데이트된 정보 또는 원본 링크 추가


        # --- Selenium 드라이버 종료 (노드 실행 끝날 때) ---
        if self.selenium_tool:
            # Selenium 도구의 close 메서드 호출 (내부적으로 quit_driver 호출)
            await self.selenium_tool.close(trace_id, comic_id)

        # --- 시간 기록 및 상태 업데이트 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['node_06_opinion_scraper_time'] = node_processing_time # 통계 기록
        logger.info(f"OpinionScraperNode 완료. 소요 시간: {node_processing_time:.2f} 초.", extra=extra_log_data)

        # 상태 업데이트 딕셔너리 생성
        updates = {
            "opinions_raw": opinions_raw, # 스크랩된 의견 목록
            "used_links": updated_used_links, # 업데이트된 사용 링크 목록
            "processing_stats": processing_stats # 업데이트된 처리 통계
        }
        # 이 노드 관련 이전 오류 메시지 상태 초기화
        current_error = state.error_message
        if current_error and "OpinionScraperNode" in current_error:
             updates["error_message"] = None # 성공 시 None으로 설정

        return updates # 상태 업데이트 반환