# app/nodes/03_news_collector_node.py (Refactored)

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Set, Optional
from urllib.parse import urlparse
import aiohttp

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.tools.search.google import GoogleSearchTool
from app.tools.search.naver import NaverSearchTool
from app.tools.search.rss import RssSearchTool
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

class NewsCollectorNode:
    """
    전용 검색 도구를 사용하여 뉴스 기사 URL을 수집합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["search_keywords", "trace_id", "comic_id", "used_links", "config"] # MODIFIED: Added comic_id
    outputs: List[str] = ["fact_urls", "used_links", "node3_processing_stats", "error_message"]

    def __init__(
        self,
        Google_Search_tool: GoogleSearchTool,
        naver_search_tool: NaverSearchTool,
        rss_search_tool: RssSearchTool,
    ):
        self.Google_Search_tool = Google_Search_tool
        self.naver_search_tool = naver_search_tool
        self.rss_search_tool = rss_search_tool
        self.trusted_domains: List[str] = []
        logger.info("NewsCollectorNode initialized with search tools (Google, Naver, RSS).")

    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict): # MODIFIED: Added extra_log_data
         self.trusted_domains = config.get('trusted_news_domains', settings.TRUSTED_NEWS_DOMAINS)
         self.max_results_per_call = config.get('search_result_count', settings.DEFAULT_SEARCH_RESULTS)
         self.target_urls_per_kw = config.get('target_urls_per_keyword', settings.DEFAULT_TARGET_URLS_PER_KW)
         self.http_timeout = config.get('http_timeout', settings.DEFAULT_HTTP_TIMEOUT)

         if not self.trusted_domains:
             logger.warning("Trusted news domains list is empty (check config/settings). Filtering disabled.", extra=extra_log_data) # MODIFIED
         else:
             logger.debug(f"Loaded trusted domains: {len(self.trusted_domains)} domains.", extra=extra_log_data) # MODIFIED
         logger.debug(f"Max results/call: {self.max_results_per_call}, Target URLs/kw: {self.target_urls_per_kw}, Timeout: {self.http_timeout}s", extra=extra_log_data) # MODIFIED

    def _is_valid_news_url(self, url: str) -> bool:
        if not url or not isinstance(url, str): return False
        if not self.trusted_domains:
             try:
                  parsed = urlparse(url)
                  return bool(parsed.scheme in ['http', 'https'] and parsed.netloc)
             except Exception: return False

        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc or parsed.scheme not in ['http', 'https']: return False
            domain = parsed.netloc.lower().replace('www.', '')
            return any(domain == trusted or domain.endswith('.' + trusted) for trusted in self.trusted_domains)
        except Exception: return False

    def _remove_duplicates(self, url_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_urls: Set[str] = set()
        unique_list = []
        for item in url_list:
            url = item.get("url")
            if url and isinstance(url, str) and url not in seen_urls:
                seen_urls.add(url)
                unique_list.append(item)
        return unique_list

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """검색 도구를 사용하여 뉴스 URL 수집 실행"""
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

        search_keywords = getattr(state, 'search_keywords', []) # Safe access
        config = getattr(state, 'config', {}) or {}
        current_used_links = getattr(state, 'used_links', []) or []

        # --- ADDED: Input Validation ---
        if not search_keywords:
            error_message = "Search keywords are missing or empty."
            logger.error(error_message, extra=extra_log_data)
            end_time = datetime.now(timezone.utc)
            node3_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "fact_urls": [],
                "used_links": current_used_links, # Return existing links
                "node3_processing_stats": node3_processing_stats,
                "error_message": error_message
            }
            # --- ADDED: End Logging (Error Case) ---
            logger.debug(f"Returning updates on error:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Error: Missing Keywords) --- (Elapsed: {node3_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        # --------------------------

        logger.info(f"Starting news URL collection using tools for keywords: {search_keywords}", extra=extra_log_data)

        all_collected_items: List[Dict[str, str]] = []
        task_errors: List[str] = [] # Collect error messages

        # Use a single session for HTTP requests
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.http_timeout)) as session:
            async def search_for_keyword(keyword: str):
                keyword_items = []
                processed_urls_in_kw = set()
                # Add comic_id to keyword-specific log data
                kw_log_data = {**extra_log_data, 'keyword': keyword}
                urls_needed = self.target_urls_per_kw

                # 1. Google 검색
                try:
                    # Pass trace_id and potentially comic_id if tool supports it
                    google_results = await self.Google_Search_tool.search_news(
                        keyword, self.max_results_per_call, trace_id=trace_id #, comic_id=comic_id
                    )
                    added_count = 0
                    for item in google_results:
                         url = item.get('url')
                         # Check validity *before* adding
                         if url and self._is_valid_news_url(url) and url not in processed_urls_in_kw:
                              item['source'] = item.get('source', 'Google')
                              item['search_keyword'] = keyword
                              keyword_items.append(item)
                              processed_urls_in_kw.add(url)
                              added_count += 1
                              if len(keyword_items) >= urls_needed: break
                    logger.debug(f"Google Search: {added_count} valid URLs added.", extra=kw_log_data)
                except Exception as e:
                    # Log detailed error here, add summary to task_errors
                    msg = f"Google Search tool failed for '{keyword}': {e}"
                    logger.error(msg, exc_info=True, extra=kw_log_data)
                    task_errors.append(f"Google Search failed: {e}") # Summary

                # 2. Naver 검색 (if needed)
                if len(keyword_items) < urls_needed:
                    try:
                        # Pass trace_id and potentially comic_id
                        naver_results = await self.naver_search_tool.search_news(
                            keyword, self.max_results_per_call, session, trace_id=trace_id #, comic_id=comic_id
                        )
                        added_count = 0
                        for item in naver_results:
                            url = item.get('url')
                            if url and self._is_valid_news_url(url) and url not in processed_urls_in_kw:
                                item['source'] = item.get('source', 'Naver')
                                item['search_keyword'] = keyword
                                keyword_items.append(item)
                                processed_urls_in_kw.add(url)
                                added_count += 1
                                if len(keyword_items) >= urls_needed: break
                        logger.debug(f"Naver Search: {added_count} valid URLs added.", extra=kw_log_data)
                    except Exception as e:
                        msg = f"Naver Search tool failed for '{keyword}': {e}"
                        logger.error(msg, exc_info=True, extra=kw_log_data)
                        task_errors.append(f"Naver Search failed: {e}") # Summary

                # 3. RSS 검색 (if needed)
                if len(keyword_items) < urls_needed:
                    try:
                        needed_for_rss = urls_needed - len(keyword_items)
                        # Pass trace_id and potentially comic_id
                        rss_results = await self.rss_search_tool.search(
                            keyword, needed_for_rss * 2, trace_id=trace_id #, comic_id=comic_id
                        )
                        added_count = 0
                        for item in rss_results:
                            url = item.get('url')
                            if url and self._is_valid_news_url(url) and url not in processed_urls_in_kw:
                                item['source'] = item.get('source', 'RSS')
                                item['search_keyword'] = keyword
                                keyword_items.append(item)
                                processed_urls_in_kw.add(url)
                                added_count += 1
                                if len(keyword_items) >= urls_needed: break
                        logger.debug(f"RSS Search: {added_count} valid URLs added.", extra=kw_log_data)
                    except Exception as e:
                        msg = f"RSS Search tool failed for '{keyword}': {e}"
                        logger.error(msg, exc_info=True, extra=kw_log_data)
                        task_errors.append(f"RSS Search failed: {e}") # Summary

                logger.info(f"Collected {len(keyword_items)} total valid URLs (Target: {urls_needed}).", extra=kw_log_data)
                return keyword_items

            # --- Execute tasks concurrently ---
            tasks = [search_for_keyword(keyword) for keyword in search_keywords]
            results_per_keyword = await asyncio.gather(*tasks, return_exceptions=True)

        # --- 결과 종합 및 처리 ---
        for i, result in enumerate(results_per_keyword):
            keyword = search_keywords[i]
            if isinstance(result, list):
                all_collected_items.extend(result)
            elif isinstance(result, Exception):
                # Errors from search_for_keyword itself (less likely if internal errors are caught)
                # or exceptions during gather setup.
                err_msg = f"Error gathering results for keyword '{keyword}': {result}"
                logger.error(err_msg, exc_info=result, extra=extra_log_data) # Log exception details
                task_errors.append(f"Gather error for '{keyword}': {result}") # Summary

        unique_items = self._remove_duplicates(all_collected_items)
        logger.info(f"Total unique news URLs collected: {len(unique_items)}", extra=extra_log_data)

        # --- 사용된 링크 추적 업데이트 ---
        updated_used_links = list(current_used_links) # Make a copy
        existing_link_urls = {link.get('url') for link in updated_used_links if link.get('url')}
        added_link_count = 0
        for item in unique_items:
            url = item.get('url')
            if url and url not in existing_link_urls:
                updated_used_links.append({
                    'url': url,
                    'purpose': f"Collected from {item.get('source', 'Unknown')} for '{item.get('search_keyword', 'N/A')}' (Fact)",
                    'node': node_class_name, # Use dynamic name
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'collected'
                })
                existing_link_urls.add(url) # Add to set to prevent duplicates within this run
                added_link_count += 1
        logger.info(f"Added {added_link_count} new URLs to used_links.", extra=extra_log_data)

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        node3_processing_stats = (end_time - start_time).total_seconds()

        # Aggregate error messages
        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"Some errors occurred during URL collection: {final_error_message}", extra=extra_log_data)

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "fact_urls": unique_items,
            "used_links": updated_used_links,
            "node3_processing_stats": node3_processing_stats,
            "error_message": final_error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if final_error_message else logger.info
        log_level(f"News collection result: {len(unique_items)} URLs collected. Errors: {final_error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node3_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}