# app/nodes/04_opinion_collector_node.py (Refactored)

import asyncio
import random
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone
from collections import defaultdict
import re
from urllib.parse import urlparse

# 프로젝트 구성 요소 임포트
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState
from app.tools.social.twitter import TwitterTool
from app.tools.social.reddit import RedditTool
from app.tools.search.google import GoogleSearchTool
from app.config.settings import settings

logger = get_logger(__name__)

class OpinionCollectorNode:
    """
    주입된 도구(Twitter, Reddit, Google Search)를 사용하여 키워드 관련 의견 URL/데이터를 수집합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["search_keywords", "trace_id", "comic_id", "used_links", "config"]
    outputs: List[str] = ["opinion_urls", "used_links", "node4_processing_stats", "error_message"]

    def __init__(
        self,
        twitter_tool: Optional[TwitterTool] = None,
        reddit_tool: Optional[RedditTool] = None,
        Google_Search_tool: Optional[GoogleSearchTool] = None
    ):
        self.twitter_tool = twitter_tool
        self.reddit_tool = reddit_tool
        self.Google_Search_tool = Google_Search_tool

        if not self.twitter_tool: logger.warning("TwitterTool not injected. Twitter search will be skipped.")
        if not self.reddit_tool: logger.warning("RedditTool not injected. Reddit search will be skipped.")
        if not self.Google_Search_tool: logger.warning("GoogleSearchTool not injected. YouTube, Blog, Community search will be skipped.")
        logger.info("OpinionCollectorNode initialized.")

    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict): # MODIFIED: Added extra_log_data
        self.yt_max = config.get('YOUTUBE_OPINION_MAX_RESULTS', settings.YOUTUBE_OPINION_MAX_RESULTS)
        self.tw_max = config.get('TWITTER_OPINION_MAX_RESULTS', settings.TWITTER_OPINION_MAX_RESULTS)
        self.rd_max = config.get('REDDIT_OPINION_MAX_RESULTS', settings.REDDIT_OPINION_MAX_RESULTS)
        self.bl_max = config.get('BLOG_OPINION_MAX_RESULTS', settings.BLOG_OPINION_MAX_RESULTS)
        self.cm_max = config.get('COMMUNITY_OPINION_MAX_RESULTS', settings.COMMUNITY_OPINION_MAX_RESULTS)
        self.max_total = config.get('MAX_TOTAL_OPINIONS_TARGET', settings.MAX_TOTAL_OPINIONS_TARGET)
        self.max_per_platform = config.get('MAX_OPINIONS_PER_PLATFORM_SAMPLING', settings.MAX_OPINIONS_PER_PLATFORM_SAMPLING)
        self.min_per_platform = config.get('MIN_OPINIONS_PER_PLATFORM_SAMPLING', settings.MIN_OPINIONS_PER_PLATFORM_SAMPLING)
        self.concurrency = config.get('OPINION_COLLECTOR_CONCURRENCY', settings.OPINION_COLLECTOR_CONCURRENCY)
        self.target_blog_domains = config.get('target_blog_domains', settings.TARGET_BLOG_DOMAINS)
        self.target_community_domains = config.get('target_community_domains', settings.TARGET_COMMUNITY_DOMAINS)

        if not isinstance(self.max_per_platform, dict): self.max_per_platform = {}
        if not isinstance(self.min_per_platform, dict): self.min_per_platform = {}

        logger.debug(f"Runtime config loaded. MaxTotal: {self.max_total}, Concurrency: {self.concurrency}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Max per platform: {self.max_per_platform}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Min per platform: {self.min_per_platform}", extra=extra_log_data) # MODIFIED

    async def _fetch_with_tool(self, tool_method, keyword: str, max_results: int, trace_id: str, comic_id: str, source_name: str) -> List[Dict[str, Any]]:
        """특정 도구 메서드를 호출하고 결과를 처리하는 공통 래퍼"""
        # Combine IDs and other info into single log data dict
        fetch_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'keyword': keyword, 'source': source_name} # MODIFIED
        if not tool_method: return []
        try:
            logger.debug(f"Fetching {source_name} for '{keyword}'...", extra=fetch_log_data) # MODIFIED
            # Assume tool methods accept trace_id and potentially comic_id
            results = await tool_method(keyword=keyword, max_results=max_results, trace_id=trace_id) # comic_id=comic_id
            processed_results = []
            for r in results:
                 r.setdefault("source", source_name)
                 r.setdefault("search_keyword", keyword)
                 processed_results.append(r)
            logger.debug(f"Received {len(processed_results)} results from {source_name} for '{keyword}'.", extra=fetch_log_data) # MODIFIED
            return processed_results
        except Exception as e:
            # Log detailed error here, the calling function will aggregate summary errors
            logger.error(f"Error calling {source_name} tool for '{keyword}': {e}", exc_info=True, extra=fetch_log_data) # MODIFIED
            # Return empty list on failure for this specific call
            return [] # MODIFIED: Return empty list instead of raising

    def _is_target_domain(self, url: str, target_domains: List[str]) -> bool:
        if not url or not target_domains: return False
        try:
            domain = urlparse(url).netloc.lower().replace('www.', '')
            return any(domain == target or domain.endswith('.' + target) for target in target_domains)
        except Exception: return False

    def _balance_and_sample_results(
        self, platform_results: Dict[str, List[Dict[str, Any]]],
        trace_id: str, comic_id: str
        ) -> List[Dict[str, Any]]:
        """설정된 제한에 따라 결과를 밸런싱 및 샘플링"""
        balance_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        balanced_results: List[Dict[str, Any]] = []
        final_urls: Set[str] = set()

        platform_order = list(platform_results.keys())
        random.shuffle(platform_order)

        logger.debug("Balancing Pass 1: Ensuring min results per platform...", extra=balance_log_data) # MODIFIED
        temp_platform_results = {k: list(v) for k, v in platform_results.items()}

        for platform in platform_order:
            results = temp_platform_results.get(platform, [])
            if not results: continue
            min_needed = self.min_per_platform.get(platform, settings.DEFAULT_MIN_OPINIONS_PER_PLATFORM)
            random.shuffle(results)
            added_count = 0
            items_to_remove_from_temp = []
            for item in results:
                 if len(balanced_results) >= self.max_total: break
                 if added_count >= min_needed: break
                 url = item.get("url")
                 if url and url not in final_urls:
                      balanced_results.append(item)
                      final_urls.add(url)
                      items_to_remove_from_temp.append(item)
                      added_count += 1

            temp_platform_results[platform] = [item for item in results if item not in items_to_remove_from_temp]
            logger.debug(f"Pass 1 ({platform}): Added {added_count} (Min: {min_needed}). Remaining: {len(temp_platform_results[platform])}", extra=balance_log_data) # MODIFIED

        logger.debug("Balancing Pass 2: Filling up to max limits...", extra=balance_log_data) # MODIFIED
        all_remaining_items = []
        platform_current_counts = defaultdict(int)
        for item in balanced_results:
             source_platform = item.get("source", "Unknown")
             platform_current_counts[source_platform] += 1

        for platform in platform_order:
            results = temp_platform_results.get(platform, [])
            if not results: continue
            max_allowed = self.max_per_platform.get(platform, settings.DEFAULT_MAX_OPINIONS_PER_PLATFORM)
            current_count = platform_current_counts.get(platform, 0)
            can_add_count = max(0, max_allowed - current_count)
            if can_add_count > 0:
                items_to_consider = results[:can_add_count]
                all_remaining_items.extend(items_to_consider)
                logger.debug(f"Pass 2 ({platform}): Max={max_allowed}, Current={current_count}, CanAdd={can_add_count}. Considering {len(items_to_consider)} items.", extra=balance_log_data) # MODIFIED

        random.shuffle(all_remaining_items)
        added_pass_2 = 0
        for item in all_remaining_items:
            if len(balanced_results) >= self.max_total: break
            url = item.get("url")
            if url and url not in final_urls:
                 source_platform = item.get("source", "Unknown")
                 if platform_current_counts[source_platform] < self.max_per_platform.get(source_platform, settings.DEFAULT_MAX_OPINIONS_PER_PLATFORM):
                      balanced_results.append(item)
                      final_urls.add(url)
                      platform_current_counts[source_platform] += 1
                      added_pass_2 += 1

        logger.info(f"Balancing complete. Added {added_pass_2} in Pass 2. Final sampled count: {len(balanced_results)}", extra=balance_log_data) # MODIFIED
        random.shuffle(balanced_results)
        return balanced_results

    def _remove_duplicates(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_urls: Set[str] = set()
        unique_list = []
        for item in items:
            url = item.get("url")
            # Ensure URL is a non-empty string before adding
            if url and isinstance(url, str) and url not in seen_urls:
                seen_urls.add(url)
                unique_list.append(item)
        return unique_list

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """주입된 도구를 사용하여 의견 수집 프로세스를 실행합니다."""
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

        search_keywords = getattr(state, 'search_keywords', [])
        config = getattr(state, 'config', {}) or {}
        current_used_links = getattr(state, 'used_links', []) or []

        # --- ADDED: Input Validation ---
        if not search_keywords:
            error_message = "Search keywords are missing or empty."
            logger.warning(error_message, extra=extra_log_data) # Warning as maybe it can proceed without keywords? Changed to warning.
            end_time = datetime.now(timezone.utc)
            node4_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "opinion_urls": [],
                "used_links": current_used_links,
                "node4_processing_stats": node4_processing_stats,
                "error_message": error_message # Pass the warning/error
            }
            # --- ADDED: End Logging (Early Exit) ---
            logger.debug(f"Returning updates (no keywords):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (No Keywords) --- (Elapsed: {node4_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        semaphore = asyncio.Semaphore(self.concurrency)
        # --------------------------

        tasks = []
        task_errors: List[str] = [] # Collect summary error messages

        async def throttled_task(coro):
             async with semaphore: return await coro

        tool_methods = []
        if self.twitter_tool: tool_methods.append((self.twitter_tool.search_recent_tweets, self.tw_max, "Twitter"))
        if self.reddit_tool: tool_methods.append((self.reddit_tool.search_posts, self.rd_max, "Reddit"))
        if self.Google_Search_tool:
            tool_methods.append((self.Google_Search_tool.search_youtube_videos, self.yt_max, "YouTube"))
            tool_methods.append((self.Google_Search_tool.search_blogs_via_cse, self.bl_max, "Blog"))
            tool_methods.append((self.Google_Search_tool.search_communities_via_cse, self.cm_max, "Community"))

        # --- ADDED: Tool Availability Check ---
        if not tool_methods:
             error_message = "No opinion collection tools available or injected."
             logger.error(error_message, extra=extra_log_data)
             end_time = datetime.now(timezone.utc)
             node4_processing_stats = (end_time - start_time).total_seconds()
             update_data = {
                 "opinion_urls": [],
                 "used_links": current_used_links,
                 "node4_processing_stats": node4_processing_stats,
                 "error_message": error_message
             }
             # --- ADDED: End Logging (Error Case) ---
             logger.debug(f"Returning updates (no tools):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
             logger.info(f"--- Finished {node_class_name} (Error: No Tools) --- (Elapsed: {node4_processing_stats:.2f}s)", extra=extra_log_data)
             # ----------------------------------------
             valid_keys = set(ComicState.model_fields.keys())
             return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------------

        for keyword in search_keywords:
             for tool_method, max_res, source_name in tool_methods:
                 # Pass trace_id and comic_id to the fetcher
                 tasks.append(throttled_task(self._fetch_with_tool(tool_method, keyword, max_res, trace_id, comic_id, source_name)))

        logger.info(f"Executing {len(tasks)} opinion collection tasks (Concurrency: {self.concurrency})...", extra=extra_log_data)
        # --- MODIFIED: Use return_exceptions=True ---
        api_results_nested = await asyncio.gather(*tasks, return_exceptions=True)
        # ------------------------------------------

        platform_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        total_collected_count = 0

        # --- MODIFIED: Handle exceptions from gather ---
        for result in api_results_nested:
            if isinstance(result, Exception):
                # _fetch_with_tool already logged the detailed error
                err_msg = f"Task failed during gather: {result}"
                logger.error(err_msg, exc_info=result, extra=extra_log_data) # Log exception details here too
                task_errors.append(f"Fetch Task Failed: {result}") # Add summary error
            elif isinstance(result, list):
                 total_collected_count += len(result)
                 for item in result:
                      source = item.get("source", "Unknown")
                      url_to_check = item.get('url') # ADDED: Check URL before filtering
                      # Filter *before* adding to platform_results
                      if source == "Blog":
                           if not url_to_check or not self._is_target_domain(url_to_check, self.target_blog_domains):
                                logger.debug(f"Skipping non-target Blog URL: {url_to_check}", extra=extra_log_data)
                                continue
                      if source == "Community":
                           if not url_to_check or not self._is_target_domain(url_to_check, self.target_community_domains):
                                logger.debug(f"Skipping non-target Community URL: {url_to_check}", extra=extra_log_data)
                                continue
                      # Passed filters or not Blog/Community
                      platform_results[source].append(item)
            else:
                # Should not happen with return_exceptions=True, but handle defensively
                logger.warning(f"Unexpected item type received from gather: {type(result)}", extra=extra_log_data)
        # ------------------------------------------------

        logger.info(f"Initial collection complete. Raw items collected (pre-filter/dedupe): {total_collected_count}. Task errors: {len(task_errors)}.", extra=extra_log_data)

        unique_platform_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        total_after_dedupe = 0
        for platform, items in platform_results.items():
             unique_items = self._remove_duplicates(items)
             unique_platform_results[platform] = unique_items
             total_after_dedupe += len(unique_items)
             logger.debug(f"{platform}: {len(items)} -> {len(unique_items)} after deduplication.", extra=extra_log_data)
        logger.info(f"Total items after deduplication: {total_after_dedupe}.", extra=extra_log_data)

        # Pass IDs for logging within balancing
        balanced_sampled_opinions = self._balance_and_sample_results(
            unique_platform_results, trace_id=trace_id, comic_id=comic_id
        )
        logger.info(f"Final unique opinion items after balancing/sampling: {len(balanced_sampled_opinions)} (Target: {self.max_total})", extra=extra_log_data)

        # --- 사용 링크 추적 업데이트 ---
        updated_used_links = list(current_used_links)
        added_link_count = 0
        existing_link_urls = {link.get('url') for link in updated_used_links if link.get('url')}
        for item in balanced_sampled_opinions:
            url = item.get('url')
            if url and isinstance(url, str) and url not in existing_link_urls: # Check type too
                link_info = {
                    'url': url,
                    'purpose': f"Collected from {item.get('source', 'Unknown')} for keyword '{item.get('search_keyword', 'Unknown')}' (Opinion)",
                    'node': node_class_name, # Use dynamic name
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'collected'
                }
                updated_used_links.append(link_info)
                existing_link_urls.add(url)
                added_link_count += 1
        logger.info(f"Added {added_link_count} new opinion URLs to used_links.", extra=extra_log_data)

        # --- 시간 기록 및 반환 ---
        end_time = datetime.now(timezone.utc)
        node4_processing_stats = (end_time - start_time).total_seconds()

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"Some errors occurred during opinion collection: {final_error_message}", extra=extra_log_data)

        # 상태 업데이트 준비
        update_data = {
            "opinion_urls": balanced_sampled_opinions,
            "used_links": updated_used_links,
            "node4_processing_stats": node4_processing_stats,
            "error_message": final_error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if final_error_message else logger.info
        log_level(f"Opinion collection result: {len(balanced_sampled_opinions)} URLs collected. Errors: {final_error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node4_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}