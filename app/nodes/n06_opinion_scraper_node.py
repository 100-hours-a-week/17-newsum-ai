# app/nodes/06_opinion_scraper_node.py (Refactored)

import asyncio
import random
import re
import traceback
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
from datetime import datetime, timezone
from collections import defaultdict

# 프로젝트 구성 요소 임포트
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState
from app.tools.social.twitter import TwitterTool
from app.tools.social.reddit import RedditTool
from app.tools.search.google import GoogleSearchTool
from app.tools.scraping.selenium_scraper import SeleniumScraperTool
from app.config.settings import settings

logger = get_logger(__name__)

class OpinionScraperNode:
    """
    수집된 의견 URL에 대해 상세 콘텐츠(텍스트, 작성자 등)를 스크랩합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["opinion_urls", "trace_id", "comic_id", "used_links", "config"]
    outputs: List[str] = ["opinions_raw", "used_links", "node6_processing_stats", "error_message"]

    def __init__(
        self,
        twitter_tool: Optional[TwitterTool] = None,
        reddit_tool: Optional[RedditTool] = None,
        Google_Search_tool: Optional[GoogleSearchTool] = None,
        selenium_tool: Optional[SeleniumScraperTool] = None
    ):
        self.twitter_tool = twitter_tool
        self.reddit_tool = reddit_tool
        self.Google_Search_tool = Google_Search_tool
        self.selenium_tool = selenium_tool

        if not self.selenium_tool: logger.warning("SeleniumScraperTool not injected. Web scraping fallback disabled.")
        if not self.twitter_tool: logger.warning("TwitterTool not injected. Twitter API priority access disabled.")
        if not self.reddit_tool: logger.warning("RedditTool not injected. Reddit API priority access disabled.")
        if not self.Google_Search_tool: logger.warning("GoogleSearchTool not injected. YouTube API priority access disabled.")
        logger.info("OpinionScraperNode initialized.")

    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict): # MODIFIED: Added extra_log_data
        self.concurrency_limit = config.get('opinion_scraper_concurrency', settings.OPINION_SCRAPER_CONCURRENCY)
        self.min_text_length = config.get('min_extracted_text_length', settings.MIN_EXTRACTED_TEXT_LENGTH)
        self.target_community_domains = config.get('target_community_domains', settings.TARGET_COMMUNITY_DOMAINS)
        self.target_blog_domains = config.get('target_blog_domains', settings.TARGET_BLOG_DOMAINS)
        # Ensure domains are lists/sets
        if not isinstance(self.target_community_domains, (list, set)):
            logger.warning(f"target_community_domains is not a list/set: {self.target_community_domains}. Using default.", extra=extra_log_data) # MODIFIED
            self.target_community_domains = settings.TARGET_COMMUNITY_DOMAINS
        if not isinstance(self.target_blog_domains, (list, set)):
            logger.warning(f"target_blog_domains is not a list/set: {self.target_blog_domains}. Using default.", extra=extra_log_data) # MODIFIED
            self.target_blog_domains = settings.TARGET_BLOG_DOMAINS

        logger.debug(f"Runtime config loaded. Concurrency: {self.concurrency_limit}, MinTextLen: {self.min_text_length}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Target Community Domains: {self.target_community_domains}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Target Blog Domains: {self.target_blog_domains}", extra=extra_log_data) # MODIFIED


    def _identify_platform(self, url: str) -> str:
        if not url or not isinstance(url, str): return 'Unknown' # Added check
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            if 'twitter.com' in domain or 'x.com' in domain: return 'Twitter'
            if 'reddit.com' in domain: return 'Reddit'
            if 'youtube.com' in domain or 'youtu.be' in domain: return 'YouTube' # Simplified YouTube check

            # Check target domains BEFORE generic blog patterns
            if any(domain == d or domain.endswith('.' + d) for d in self.target_community_domains): return 'Community'
            if any(domain == d or domain.endswith('.' + d) for d in self.target_blog_domains): return 'Blog'

            # Generic patterns (less reliable)
            if '/blog/' in parsed.path.lower() or '/post/' in parsed.path.lower() or 'blog.' in domain: return 'Blog'

            return 'OtherWeb'
        except Exception as e: # Catch specific errors if possible
            logger.warning(f"Failed to identify platform for URL: {url}. Error: {e}")
            return 'Unknown'

    def _extract_platform_ids(self, url: str, platform: str) -> Dict[str, Optional[str]]:
        ids = {"tweet_id": None, "submission_id": None, "comment_id": None, "video_id": None}
        if not url or not isinstance(url, str): return ids # Added check
        try:
            if platform == 'Twitter':
                match = re.search(r'/status(?:es)?/(\d+)', url)
                if match: ids['tweet_id'] = match.group(1)
            elif platform == 'Reddit':
                # Improved regex: handles trailing slashes, optional username part
                match = re.search(r'/comments/([a-zA-Z0-9]+)(?:/[^/]+/([a-zA-Z0-9]+))?/?', url)
                if match:
                     ids['submission_id'] = match.group(1)
                     ids['comment_id'] = match.group(2) # Can be None if it's a submission link
            elif platform == 'YouTube':
                 patterns = [
                     r'[?&]v=([^&/#]+)',      # Standard watch?v=...
                     r'youtu\.be/([^?&/#]+)', # Shortened youtu.be/...
                     r'/embed/([^?&/#]+)',    # Embed URL /embed/...
                     r'/shorts/([^?&/#]+)'   # Shorts URL /shorts/...
                 ]
                 for pattern in patterns:
                      match = re.search(pattern, url)
                      if match:
                           # Basic validation: ensure it's not just gibberish (e.g., > 5 chars)
                           video_id_candidate = match.group(1)
                           if video_id_candidate and len(video_id_candidate) > 5:
                               ids['video_id'] = video_id_candidate
                               break # Use first valid match
        except Exception as e:
            logger.warning(f"Failed to extract IDs from {platform} URL '{url}': {e}")
        return ids

    async def _fetch_opinion_api(self, platform: str, ids: Dict[str, Optional[str]], trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """플랫폼별 API 도구를 호출하여 상세 정보 가져오기"""
        api_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'platform': platform, 'ids': ids} # MODIFIED
        api_result = None
        tool_called = False # ADDED: Track if a relevant tool was called
        try:
            if platform == 'Twitter' and ids.get('tweet_id') and self.twitter_tool:
                 tool_called = True # ADDED
                 # Assume get_tweet_details accepts trace_id/comic_id if needed by the tool
                 api_result = await self.twitter_tool.get_tweet_details(ids['tweet_id'], trace_id)
            elif platform == 'Reddit' and self.reddit_tool:
                 comment_id = ids.get('comment_id')
                 submission_id = ids.get('submission_id')
                 if comment_id:
                      tool_called = True # ADDED
                      api_result = await self.reddit_tool.get_comment_details(comment_id, trace_id)
                 elif submission_id: # Only fetch submission if no comment ID
                      tool_called = True # ADDED
                      api_result = await self.reddit_tool.get_submission_details(submission_id, trace_id)
            elif platform == 'YouTube' and ids.get('video_id') and self.Google_Search_tool:
                 tool_called = True # ADDED
                 # Assume get_youtube_details accepts trace_id/comic_id
                 api_result = await self.Google_Search_tool.get_youtube_details(ids['video_id'], trace_id)

            if tool_called and api_result:
                 logger.info(f"Successfully fetched data via API.", extra=api_log_data) # MODIFIED
            elif tool_called and not api_result:
                 # Tool was called but returned None/empty, indicates API failure within tool
                 logger.warning(f"API call using tool returned no data.", extra=api_log_data) # MODIFIED
            # If tool_called is False, no relevant API tool was available/applicable

            return api_result # Returns None if no tool called or tool failed
        except Exception as api_err:
            # Catch errors during the API call itself (e.g., network issues if not handled by tool)
            logger.warning(f"API call failed: {api_err}", exc_info=True, extra=api_log_data) # MODIFIED (Warning, as Selenium might follow)
            return None # Indicate failure

    async def _scrape_with_selenium(self, url: str, platform: str, trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """SeleniumScraperTool을 호출하여 웹 스크래핑 수행"""
        sel_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, 'platform': platform} # MODIFIED
        if not self.selenium_tool:
            logger.warning("Selenium tool unavailable, cannot scrape.", extra=sel_log_data) # MODIFIED
            return None

        logger.debug(f"Attempting Selenium scraping...", extra=sel_log_data) # MODIFIED
        try:
            # Assume scrape_url accepts trace_id/comic_id
            scraped_data = await self.selenium_tool.scrape_url(url, platform, trace_id, comic_id)
            if scraped_data:
                 logger.info(f"Selenium scraping successful.", extra=sel_log_data) # MODIFIED
            else:
                 # Scraper ran but found nothing matching its rules
                 logger.warning(f"Selenium scraping returned no data (scraper rules might not match).", extra=sel_log_data) # MODIFIED
            return scraped_data
        except Exception as sel_err:
            # Catch errors during the Selenium process itself
            logger.error(f"Selenium scraping failed: {sel_err}", exc_info=True, extra=sel_log_data) # MODIFIED (Error, as this is the fallback)
            return None # Indicate failure

    async def _process_url(self, url_info: Dict[str, Any], trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """단일 의견 URL 처리. Returns processed data or None on failure/filter."""
        url = url_info.get('url')
        source_info = {
            "original_source": url_info.get('source', 'Unknown'),
            "search_keyword": url_info.get('search_keyword', 'N/A')
        }
        # Combine all context for logging this URL processing task
        process_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, **source_info} # MODIFIED

        # --- Input validation for the URL itself ---
        if not url or not isinstance(url, str):
             logger.warning("Invalid URL provided in url_info. Skipping.", extra=process_log_data) # MODIFIED
             return None
        # -----------------------------------------

        logger.info(f"Processing opinion URL: {url[:80]}...", extra=process_log_data) # MODIFIED
        platform = self._identify_platform(url)
        ids = self._extract_platform_ids(url, platform)
        scraped_data: Optional[Dict[str, Any]] = None
        method_used = "None" # Start with None

        # 1. API 우선 시도
        can_try_api = platform in ['Twitter', 'Reddit', 'YouTube'] and any(v for k, v in ids.items() if k.endswith('_id'))
        if can_try_api:
            logger.debug(f"Attempting API access for {platform}...", extra=process_log_data) # MODIFIED
            # Pass IDs for logging within API fetcher
            scraped_data = await self._fetch_opinion_api(platform, ids, trace_id, comic_id)
            if scraped_data:
                method_used = f"API ({platform})"
            else:
                # Log API failure at this level too
                logger.warning(f"API access failed or returned no data for {platform}.", extra=process_log_data) # MODIFIED

        # 2. Selenium 대체 시도 (only if API failed/not applicable AND Selenium tool exists)
        if not scraped_data and self.selenium_tool:
             logger.info(f"API failed or not applicable. Trying Selenium fallback for {platform}...", extra=process_log_data) # MODIFIED (Info level)
             # Pass IDs for logging within Selenium scraper
             scraped_data = await self._scrape_with_selenium(url, platform, trace_id, comic_id)
             if scraped_data:
                 method_used = f"Selenium ({platform})"
             else:
                 # Log Selenium failure
                 logger.warning(f"Selenium fallback failed or returned no data for {platform}.", extra=process_log_data) # MODIFIED

        # 3. 결과 처리 및 반환
        if scraped_data:
             text_content = scraped_data.get("text", "")
             # Validate text length
             if not text_content or len(str(text_content)) < self.min_text_length: # Check len of str representation
                  logger.warning(f"Extracted text too short ({len(str(text_content))} chars, min: {self.min_text_length}). Discarding. Method: {method_used}", extra=process_log_data) # MODIFIED
                  return None # Indicate filtered out

             # Ensure basic fields exist
             result = {
                 "url": url,
                 "platform": platform,
                 "method": method_used,
                 "text": str(text_content), # Ensure string
                 "author": scraped_data.get("author"),
                 "timestamp": scraped_data.get("timestamp"), # Should be ISO format ideally
                 "likes": scraped_data.get("likes", 0),
                 "title": scraped_data.get("title"),
                 **source_info # Include original source/keyword
             }
             logger.info(f"Successfully processed URL using {method_used}.", extra=process_log_data) # ADDED Success log
             return result
        else:
            # Failed using all available methods
            logger.error(f"Failed to extract data using all available methods (API/Selenium).", extra=process_log_data) # MODIFIED (Error level)
            return None # Indicate failure

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """주입된 도구를 사용하여 의견 스크래핑 프로세스를 실행합니다."""
        start_time = datetime.now(timezone.utc)
        # --- MODIFIED: Get trace_id and comic_id safely ---
        comic_id = getattr(state, 'comic_id', 'unknown_comic')
        trace_id = getattr(state, 'trace_id', comic_id)
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        # -------------------------------------------------

        node_class_name = self.__class__.__name__
        # --- ADDED: Start Logging ---
        logger.info(f"--- Executing {node_class_name} ---", extra=extra_log_data)
        logger.debug(f"Entering state:\n{summarize_for_logging(state, is_state_object=True)}", extra=extra_log_data)
        # --------------------------

        opinion_urls = getattr(state, 'opinion_urls', []) # Safe access
        config = getattr(state, 'config', {}) or {}
        current_used_links = getattr(state, 'used_links', []) or []

        # --- ADDED: Input Validation ---
        if not opinion_urls:
            error_message = "No opinion URLs provided for scraping."
            # Warning, as it might be valid that no opinions were found
            logger.warning(error_message, extra=extra_log_data)
            end_time = datetime.now(timezone.utc)
            node6_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "opinions_raw": [],
                "used_links": current_used_links,
                "node6_processing_stats": node6_processing_stats,
                "error_message": error_message
            }
            # --- ADDED: End Logging (Early Exit) ---
            logger.debug(f"Returning updates (no URLs):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (No URLs) --- (Elapsed: {node6_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        # --------------------------

        logger.info(f"Starting scraping for {len(opinion_urls)} opinion URLs (Concurrency: {self.concurrency_limit})...", extra=extra_log_data)

        tasks = []
        for url_info in opinion_urls:
             async def task_with_semaphore(ui):
                  async with semaphore:
                       # Pass trace_id, comic_id
                       return await self._process_url(ui, trace_id, comic_id)
             tasks.append(task_with_semaphore(url_info))

        # --- MODIFIED: Use return_exceptions=True ---
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # ------------------------------------------

        opinions_raw: List[Dict[str, Any]] = []
        successful_urls = set()
        failed_count = 0
        task_errors: List[str] = [] # Collect summary errors from gather

        # --- MODIFIED: Process results carefully ---
        for i, res in enumerate(results):
            original_url_info = opinion_urls[i] if i < len(opinion_urls) else {}
            original_url = original_url_info.get('url', f'Unknown_URL_{i}')

            if isinstance(res, dict) and res is not None:
                opinions_raw.append(res)
                successful_urls.add(original_url) # Add the URL that succeeded
            else:
                 failed_count += 1
                 if res is None:
                      # Failure or filter within _process_url (already logged there)
                      # Optionally add generic error here if needed
                      # task_errors.append(f"Processing failed or filtered for {original_url}")
                      pass # Avoid adding duplicate errors if _process_url logged it
                 elif isinstance(res, Exception):
                      # Exception from gather itself or outside _process_url's try/except
                      err_msg = f"Gather exception for URL {original_url}: {res}"
                      logger.error(err_msg, exc_info=res, extra=extra_log_data) # Log details
                      task_errors.append(f"Exception for {original_url}: {res}") # Summary
                 else:
                      # Unexpected type
                      logger.warning(f"Unexpected result type for {original_url}: {type(res)}", extra=extra_log_data)
                      task_errors.append(f"Unknown error for {original_url}")
        # --------------------------------------------

        logger.info(f"Opinion scraping complete. Success: {len(successful_urls)}, Failed/Skipped/Exception: {failed_count}.", extra=extra_log_data)

        # --- Update used_links status ---
        updated_used_links = []
        links_updated_count = 0
        input_urls_processed = {op_url.get('url') for op_url in opinion_urls if op_url.get('url')} # Set of URLs this node *should* have processed

        for link in current_used_links:
            url = link.get('url')
            # Only update status if this URL was part of this node's input
            if url in input_urls_processed:
                 links_updated_count += 1
                 updated_link = link.copy()
                 purpose = updated_link.get('purpose', '')
                 if url in successful_urls:
                      updated_link['purpose'] = purpose.replace('(Opinion)', '(Scraped Opinion)') if '(Opinion)' in purpose else f"{purpose} (Scraped Opinion)"
                      updated_link['status'] = "processed"
                 else:
                      # Mark as failed if it was input but not in successful_urls
                      updated_link['purpose'] = f"{purpose} (Opinion Scraping Failed/Filtered)"
                      updated_link['status'] = "failed_or_filtered"
                 updated_used_links.append(updated_link)
            else:
                 # Pass through links not processed by this node
                 updated_used_links.append(link)
        logger.info(f"Updated status for {links_updated_count} opinion link entries in used_links related to this node's processing.", extra=extra_log_data)

        # --- Selenium 드라이버 종료 (존재 시) ---
        if self.selenium_tool:
            logger.debug("Attempting to close Selenium tool...", extra=extra_log_data)
            try:
                 # Assume close is async and accepts trace_id/comic_id
                 await self.selenium_tool.close(trace_id, comic_id)
                 logger.info("Selenium tool closed successfully.", extra=extra_log_data)
            except Exception as close_err:
                 logger.error(f"Error closing Selenium tool: {close_err}", exc_info=True, extra=extra_log_data)

        # --- 시간 기록 및 상태 업데이트 반환 ---
        end_time = datetime.now(timezone.utc)
        node6_processing_stats = (end_time - start_time).total_seconds()

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"Some errors occurred during opinion scraping task execution: {final_error_message}", extra=extra_log_data)

        # 상태 업데이트 딕셔너리 생성
        update_data = {
            "opinions_raw": opinions_raw,
            "used_links": updated_used_links,
            "node6_processing_stats": node6_processing_stats,
            "error_message": final_error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if final_error_message else logger.info
        log_level(f"Opinion scraping result: {len(opinions_raw)} opinions scraped. Errors: {final_error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node6_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}