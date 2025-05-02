# app/nodes/06_opinion_scraper_node.py (Improved Version)

import asyncio
import random
import re
import traceback
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
from datetime import datetime, timezone
from collections import defaultdict

# 프로젝트 구성 요소 임포트
from app.utils.logger import get_logger
from app.workflows.state import ComicState
from app.tools.social.twitter import TwitterTool
from app.tools.social.reddit import RedditTool
from app.tools.search.google import GoogleSearchTool
from app.tools.scraping.selenium_scraper import SeleniumScraperTool
from app.config.settings import settings # 기본 설정 참조용

# 로거 설정
logger = get_logger(__name__)

class OpinionScraperNode:
    """
    수집된 의견 URL에 대해 상세 콘텐츠(텍스트, 작성자 등)를 스크랩합니다.
    API 우선 접근 방식을 사용하며, 실패 시 Selenium 기반 웹 스크래핑으로 대체합니다.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["opinion_urls", "trace_id", "comic_id", "used_links", "config", "processing_stats"]
    outputs: List[str] = ["opinions_raw", "used_links", "processing_stats", "error_message"]

    # 의존성 주입
    def __init__(
        self,
        twitter_tool: Optional[TwitterTool] = None,
        reddit_tool: Optional[RedditTool] = None,
        Google_Search_tool: Optional[GoogleSearchTool] = None, # 이름 일관성 유지
        selenium_tool: Optional[SeleniumScraperTool] = None
        # langsmith_service: Optional[LangSmithService] = None
    ):
        self.twitter_tool = twitter_tool
        self.reddit_tool = reddit_tool
        self.Google_Search_tool = Google_Search_tool # 속성명도 통일
        self.selenium_tool = selenium_tool
        # self.langsmith = langsmith_service

        if not self.selenium_tool: logger.warning("SeleniumScraperTool not injected. Web scraping fallback disabled.")
        if not self.twitter_tool: logger.warning("TwitterTool not injected. Twitter API priority access disabled.")
        if not self.reddit_tool: logger.warning("RedditTool not injected. Reddit API priority access disabled.")
        if not self.Google_Search_tool: logger.warning("GoogleSearchTool not injected. YouTube API priority access disabled.")
        logger.info("OpinionScraperNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        # 노드별 동시성 설정
        self.concurrency_limit = config.get('opinion_scraper_concurrency', settings.OPINION_SCRAPER_CONCURRENCY)
        # 최소 텍스트 길이 설정
        self.min_text_length = config.get('min_extracted_text_length', settings.MIN_EXTRACTED_TEXT_LENGTH)
        # 타겟 커뮤니티/블로그 도메인 (필터링용)
        self.target_community_domains = config.get('target_community_domains', settings.TARGET_COMMUNITY_DOMAINS)
        self.target_blog_domains = config.get('target_blog_domains', settings.TARGET_BLOG_DOMAINS)

        logger.debug(f"Runtime config loaded. Concurrency: {self.concurrency_limit}, MinTextLen: {self.min_text_length}")
        logger.debug(f"Target Community Domains: {self.target_community_domains}")
        logger.debug(f"Target Blog Domains: {self.target_blog_domains}")


    # --- 플랫폼 식별 및 ID 추출 ---
    # 참고: 이 로직은 플랫폼 URL 구조 변경에 취약할 수 있음. 지속적인 관리가 필요.
    def _identify_platform(self, url: str) -> str:
        """URL 기반 플랫폼 식별 (개선된 휴리스틱)"""
        if not url: return 'Unknown'
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            if 'twitter.com' in domain or 'x.com' in domain: return 'Twitter'
            if 'reddit.com' in domain: return 'Reddit'
            # youtube.com 또는 youtu.be
            if 'youtube.com' in domain or 'youtu.be' in domain: return 'YouTube'
            # 설정된 커뮤니티 도메인 확인
            if any(domain == d or domain.endswith('.' + d) for d in self.target_community_domains): return 'Community'
            # 설정된 블로그 도메인 확인
            if any(domain == d or domain.endswith('.' + d) for d in self.target_blog_domains): return 'Blog'
            # 기타 기본적인 블로그 패턴 확인 (보수적)
            if '/blog/' in parsed.path.lower() or '/post/' in parsed.path.lower(): return 'Blog'

            return 'OtherWeb' # 그 외 웹사이트
        except Exception:
            logger.warning(f"Failed to identify platform for URL: {url}")
            return 'Unknown'

    def _extract_platform_ids(self, url: str, platform: str) -> Dict[str, Optional[str]]:
        """플랫폼에 따라 URL에서 관련 ID 추출 (개선된 정규식)"""
        ids = {"tweet_id": None, "submission_id": None, "comment_id": None, "video_id": None}
        if not url: return ids
        try:
            if platform == 'Twitter':
                match = re.search(r'/status(?:es)?/(\d+)', url)
                if match: ids['tweet_id'] = match.group(1)
            elif platform == 'Reddit':
                match = re.search(r'/comments/([a-zA-Z0-9]+)(?:/[^/]+/([a-zA-Z0-9]+))?', url)
                if match:
                     ids['submission_id'] = match.group(1)
                     ids['comment_id'] = match.group(2) # 댓글 ID (Optional)
            elif platform == 'YouTube':
                 # 다양한 YouTube URL 형식 처리
                 patterns = [
                     r'[?&]v=([^&]+)',           # youtube.com/watch?v=VIDEO_ID
                     r'youtu\.be/([^?&]+)',      # youtu.be/VIDEO_ID
                     r'/embed/([^?&]+)',         # youtube.com/embed/VIDEO_ID
                     r'/shorts/([^?&]+)'        # youtube.com/shorts/VIDEO_ID
                 ]
                 for pattern in patterns:
                      match = re.search(pattern, url)
                      if match:
                           ids['video_id'] = match.group(1)
                           break # 첫 번째 매칭 사용
        except Exception as e:
            logger.warning(f"Failed to extract IDs from {platform} URL '{url}': {e}")
        return ids

    # --- API 및 Selenium 호출 로직 (도구 위임) ---
    async def _fetch_opinion_api(self, platform: str, ids: Dict[str, Optional[str]], trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """플랫폼별 API 도구를 호출하여 상세 정보 가져오기 (도구 내 재시도 가정)"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'platform': platform, 'ids': ids}
        api_result = None
        try:
            if platform == 'Twitter' and ids.get('tweet_id') and self.twitter_tool:
                 api_result = await self.twitter_tool.get_tweet_details(ids['tweet_id'], trace_id)
            elif platform == 'Reddit' and self.reddit_tool:
                 # 댓글 ID 우선 조회, 없으면 게시글 ID 조회
                 comment_id = ids.get('comment_id')
                 submission_id = ids.get('submission_id')
                 if comment_id: api_result = await self.reddit_tool.get_comment_details(comment_id, trace_id)
                 elif submission_id: api_result = await self.reddit_tool.get_submission_details(submission_id, trace_id)
            elif platform == 'YouTube' and ids.get('video_id') and self.Google_Search_tool:
                 # YouTube 상세 정보는 GoogleSearchTool의 다른 메서드 사용 가정
                 api_result = await self.Google_Search_tool.get_youtube_details(ids['video_id'], trace_id)

            if api_result: logger.info(f"Successfully fetched data via API.", extra=extra_log_data)
            return api_result
        except Exception as api_err:
            logger.warning(f"API call failed (after internal retries): {api_err}", extra=extra_log_data)
            return None

    async def _scrape_with_selenium(self, url: str, platform: str, trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """SeleniumScraperTool을 호출하여 웹 스크래핑 수행 (도구 내 재시도 가정)"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, 'platform': platform}
        if not self.selenium_tool:
            logger.warning("Selenium tool unavailable, cannot scrape.", extra=extra_log_data)
            return None

        logger.debug(f"Attempting Selenium scraping...", extra=extra_log_data)
        try:
            scraped_data = await self.selenium_tool.scrape_url(url, platform, trace_id, comic_id)
            if scraped_data: logger.info(f"Selenium scraping successful.", extra=extra_log_data)
            else: logger.warning(f"Selenium scraping returned no data.", extra=extra_log_data)
            return scraped_data
        except Exception as sel_err:
            logger.error(f"Selenium scraping ultimately failed: {sel_err}", exc_info=True, extra=extra_log_data)
            return None

    # --- 단일 URL 처리 로직 ---
    async def _process_url(self, url_info: Dict[str, Any], trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """단일 의견 URL 처리: 플랫폼 식별, ID 추출, API 우선 시도 후 Selenium 대체."""
        url = url_info.get('url')
        # Node 04에서 전달된 원본 소스 및 키워드 포함
        source_info = {
            "original_source": url_info.get('source', 'Unknown'),
            "search_keyword": url_info.get('search_keyword', 'N/A')
        }
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, **source_info}
        if not url: return None

        logger.info(f"Processing opinion URL: {url[:80]}...", extra=extra_log_data)
        platform = self._identify_platform(url)
        ids = self._extract_platform_ids(url, platform)
        scraped_data: Optional[Dict[str, Any]] = None
        method_used = "Unknown"

        # 1. API 우선 시도
        can_try_api = platform in ['Twitter', 'Reddit', 'YouTube'] and any(v for k, v in ids.items() if k.endswith('_id'))
        if can_try_api:
            logger.debug(f"Attempting API access for {platform}...", extra=extra_log_data)
            scraped_data = await self._fetch_opinion_api(platform, ids, trace_id, comic_id)
            if scraped_data: method_used = f"API ({platform})"

        # 2. Selenium 대체 시도
        if not scraped_data and self.selenium_tool:
             logger.debug(f"API failed or not applicable. Trying Selenium fallback for {platform}...", extra=extra_log_data)
             scraped_data = await self._scrape_with_selenium(url, platform, trace_id, comic_id)
             if scraped_data: method_used = f"Selenium ({platform})"
        elif not scraped_data:
             logger.warning(f"No data fetched via API and Selenium tool unavailable/failed.", extra=extra_log_data)

        # 3. 결과 포맷팅 및 반환
        if scraped_data:
             text_content = scraped_data.get("text", "")
             # 최소 텍스트 길이 검증
             if not text_content or len(text_content) < self.min_text_length:
                  logger.warning(f"Extracted text too short ({len(text_content)} chars, min: {self.min_text_length}). Discarding. ({method_used})", extra=extra_log_data)
                  return None

             # 최종 결과 구조화
             result = {
                 "url": url,
                 "platform": platform,
                 "method": method_used,
                 "text": text_content,
                 "author": scraped_data.get("author"),
                 "timestamp": scraped_data.get("timestamp"),
                 "likes": scraped_data.get("likes", 0),
                 "title": scraped_data.get("title"),
                 # 원본 소스 정보 포함
                 **source_info
             }
             return result
        else:
            logger.warning(f"Failed to extract data using all available methods.", extra=extra_log_data)
            return None

    # --- 메인 실행 메서드 (run으로 이름 변경) ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """주입된 도구를 사용하여 의견 스크래핑 프로세스를 실행합니다."""
        start_time = datetime.now(timezone.utc)
        comic_id = state.comic_id
        trace_id = state.trace_id
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        log_prefix = f"[{trace_id}]"

        logger.info(f"{log_prefix} OpinionScraperNode starting...")

        opinion_urls = state.opinion_urls or []
        config = state.config or {}
        current_used_links = state.used_links or []
        processing_stats = state.processing_stats or {}

        if not opinion_urls:
            logger.warning(f"{log_prefix} No opinion URLs to scrape.", extra=extra_log_data)
            processing_stats['opinion_scraper_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"opinions_raw": [], "used_links": current_used_links, "processing_stats": processing_stats}

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        # --------------------------

        logger.info(f"{log_prefix} Starting scraping for {len(opinion_urls)} opinion URLs (Concurrency: {self.concurrency_limit})...", extra=extra_log_data)

        tasks = []
        for url_info in opinion_urls:
             async def task_with_semaphore(ui):
                  async with semaphore:
                       return await self._process_url(ui, trace_id, comic_id)
             tasks.append(task_with_semaphore(url_info))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        opinions_raw: List[Dict[str, Any]] = []
        successful_urls = set()
        failed_count = 0
        task_errors: List[str] = []

        for i, res in enumerate(results):
            original_url = opinion_urls[i].get('url', f'Unknown_URL_{i}')
            if isinstance(res, dict) and res is not None:
                opinions_raw.append(res)
                successful_urls.add(original_url)
            else:
                 failed_count += 1
                 # 래퍼 내부 또는 gather 에서 발생한 예외 로깅 및 에러 메시지 기록
                 if isinstance(res, Exception):
                      err_msg = f"Scraping task failed for {original_url}: {res}"
                      logger.error(f"{log_prefix} {err_msg}", exc_info=res)
                      task_errors.append(err_msg)
                 # else: res is None (처리 실패 또는 필터링됨, _process_url에서 로깅)

        logger.info(f"{log_prefix} Opinion scraping complete. Success: {len(successful_urls)}, Failed/Skipped: {failed_count}.", extra=extra_log_data)

        # --- used_links 상태 업데이트 ---
        updated_used_links = []
        links_updated_count = 0
        for link in current_used_links:
            url = link.get('url')
            is_opinion_input = any(op_url.get('url') == url for op_url in opinion_urls)

            if is_opinion_input:
                 links_updated_count += 1
                 if url in successful_urls:
                      link['purpose'] = link.get('purpose', '').replace('(Opinion)', '(Scraped Opinion)')
                      link['status'] = "processed"
                 else:
                      link['purpose'] += " (Opinion Scraping Failed/Filtered)"
                      link['status'] = "failed_or_filtered" # 상태 명시
            updated_used_links.append(link)
        logger.info(f"{log_prefix} Updated status for {links_updated_count} opinion link entries in used_links.")

        # --- Selenium 드라이버 종료 (존재 시) ---
        if self.selenium_tool:
            try:
                 # close 메서드가 비동기일 수 있음
                 await self.selenium_tool.close(trace_id, comic_id)
            except Exception as close_err:
                 logger.error(f"{log_prefix} Error closing Selenium tool: {close_err}", exc_info=True)

        # --- 시간 기록 및 상태 업데이트 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['opinion_scraper_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} OpinionScraperNode finished. Elapsed time: {processing_stats['opinion_scraper_node_time']:.2f} seconds.", extra=extra_log_data)

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"{log_prefix} Some errors occurred during opinion scraping: {final_error_message}")

        # 상태 업데이트 딕셔너리 생성
        updates = {
            "opinions_raw": opinions_raw,
            "used_links": updated_used_links,
            "processing_stats": processing_stats,
            "error_message": final_error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in updates.items() if k in valid_keys}