# app/nodes/05_news_scraper_node.py (Refactored)

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# 프로젝트 구성 요소 임포트
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState
from app.tools.scraping.article_scraper import ArticleScraperTool
from app.config.settings import settings

logger = get_logger(__name__)

class NewsScraperNode:
    """
    수집된 뉴스 URL 목록에 대해 ArticleScraperTool을 사용하여 기사 콘텐츠 스크래핑을 조율합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["fact_urls", "trace_id", "comic_id", "used_links", "config"]
    outputs: List[str] = ["articles", "used_links", "node5_processing_stats", "error_message"]

    def __init__(self, scraper_tool: ArticleScraperTool):
        if not scraper_tool:
             # This is a critical dependency, raise error if not provided
             logger.critical("ArticleScraperTool dependency not provided to NewsScraperNode!")
             raise ValueError("ArticleScraperTool is required for NewsScraperNode")
        self.scraper_tool = scraper_tool
        logger.info("NewsScraperNode initialized.")

    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict): # MODIFIED: Added extra_log_data
        self.allowed_languages = config.get('language_filter', settings.LANGUAGE_FILTER)
        self.concurrency_limit = config.get('scraper_concurrency', settings.SCRAPER_CONCURRENCY)
        # Ensure allowed_languages is a list or set for 'in' check
        if not isinstance(self.allowed_languages, (list, set)):
             logger.warning(f"language_filter is not a list or set: {self.allowed_languages}. Using default: {settings.LANGUAGE_FILTER}", extra=extra_log_data) # MODIFIED
             self.allowed_languages = settings.LANGUAGE_FILTER
        logger.debug(f"Runtime config loaded. Allowed languages: {self.allowed_languages}, Concurrency: {self.concurrency_limit}", extra=extra_log_data) # MODIFIED

    async def _process_single_url_wrapper(
        self,
        url_info: Dict[str, Any],
        trace_id: str,
        comic_id: str
    ) -> Optional[Dict[str, Any]]:
        """단일 URL 처리 래퍼: 스크래핑 및 언어 필터링. Returns scraped data or None on failure/filter."""
        original_url = url_info.get("url")
        # Combine IDs and URL info for logging this specific task
        wrapper_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': original_url} # MODIFIED

        # --- Input Validation (within wrapper) ---
        if not original_url or not isinstance(original_url, str):
            logger.warning("URL info is missing 'url' field or URL is not a string. Skipping.", extra=wrapper_log_data) # MODIFIED
            return None # Indicate failure for this URL
        # -----------------------------------------

        try:
            # Assume scrape_article handles its own retries and logs internal attempts/failures
            # Pass trace_id and comic_id for tool's internal logging/tracing
            article_data = await self.scraper_tool.scrape_article(original_url, trace_id, comic_id)

            if not article_data or not article_data.get('text'):
                # Log failure reason if possible (scraper tool might return error info)
                fail_reason = article_data.get('error', 'no text extracted') if article_data else 'scraper returned None'
                logger.warning(f"Scraping failed or no text extracted for {original_url}. Reason: {fail_reason}", extra=wrapper_log_data) # MODIFIED
                return None # Indicate failure

            # Language Filtering
            language = article_data.get('language', 'und') # Default to 'und' if not detected
            # Ensure allowed_languages is iterable
            is_allowed = language == 'und' or (isinstance(self.allowed_languages, (list, set)) and language in self.allowed_languages)

            if not is_allowed:
                logger.info(f"Skipping scraped article: Language '{language}' not in allowed list {self.allowed_languages}. URL: {article_data.get('url')}", extra=wrapper_log_data) # MODIFIED
                return None # Indicate filtered out

            # Add source info from the original url_info
            article_data['source_search_keyword'] = url_info.get('search_keyword')
            article_data['source_type'] = url_info.get('source') # e.g., "Google", "Naver"
            article_data['original_url_from_source'] = original_url # Explicitly keep original URL

            logger.debug(f"Successfully scraped and filtered article.", extra=wrapper_log_data) # ADDED
            return article_data

        except Exception as e:
            # Catch unexpected errors during the wrapper execution
            logger.exception(f"Unexpected error processing URL {original_url}: {e}", extra=wrapper_log_data) # MODIFIED use exception
            return None # Indicate failure

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """수집된 뉴스 URL 목록에 대해 스크래핑 프로세스를 실행합니다."""
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

        fact_urls = getattr(state, 'fact_urls', []) # Safe access
        config = getattr(state, 'config', {}) or {}
        current_used_links = getattr(state, 'used_links', []) or []

        # --- ADDED: Input Validation ---
        if not fact_urls:
            error_message = "No fact URLs provided for scraping."
            # Use warning as it might be valid that no fact URLs were found previously
            logger.warning(error_message, extra=extra_log_data)
            end_time = datetime.now(timezone.utc)
            node5_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "articles": [],
                "used_links": current_used_links,
                "node5_processing_stats": node5_processing_stats,
                "error_message": error_message # Pass warning/error
            }
            # --- ADDED: End Logging (Early Exit) ---
            logger.debug(f"Returning updates (no URLs):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (No URLs) --- (Elapsed: {node5_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        # --------------------------

        logger.info(f"Starting scraping for {len(fact_urls)} news URLs (Concurrency: {self.concurrency_limit})...", extra=extra_log_data)

        tasks = []
        for url_info in fact_urls:
             # Define async function inside loop to capture url_info correctly
             async def task_with_semaphore(ui):
                  async with semaphore:
                       # Pass trace_id, comic_id to the wrapper
                       return await self._process_single_url_wrapper(ui, trace_id, comic_id)
             tasks.append(task_with_semaphore(url_info))

        # --- MODIFIED: Use return_exceptions=True ---
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # ------------------------------------------

        scraped_articles: List[Dict[str, Any]] = []
        # Map original URL to its processing outcome
        processed_url_map: Dict[str, str] = {} # status: 'processed', 'failed', 'filtered', 'exception'
        task_errors: List[str] = [] # Collect summary error messages from gather

        # --- MODIFIED: Process results carefully ---
        for i, res in enumerate(results):
            # Get the original URL info corresponding to this result
            # Handle potential index errors if fact_urls list was modified (shouldn't happen)
            original_url_info = fact_urls[i] if i < len(fact_urls) else {}
            original_url = original_url_info.get('url', f'Unknown_URL_{i}')

            if isinstance(res, dict) and res is not None:
                scraped_articles.append(res)
                processed_url_map[original_url] = 'processed'
            elif res is None:
                # This means _process_single_url_wrapper returned None (failure or filtered)
                # The wrapper already logged the specific reason.
                processed_url_map[original_url] = 'failed_or_filtered'
                # Optionally add a generic message to task_errors if needed, but might be noisy
                # task_errors.append(f"Processing failed or filtered for {original_url}")
            elif isinstance(res, Exception):
                # Exception occurred *outside* the wrapper's try/except or during gather setup
                err_msg = f"Gather exception for URL {original_url}: {res}"
                logger.error(err_msg, exc_info=res, extra=extra_log_data) # Log exception details
                processed_url_map[original_url] = 'exception'
                task_errors.append(f"Exception for {original_url}: {res}") # Summary error
            else:
                # Unexpected result type
                 logger.warning(f"Unexpected result type for {original_url}: {type(res)}", extra=extra_log_data)
                 processed_url_map[original_url] = 'unknown_error'
                 task_errors.append(f"Unknown error for {original_url}")
        # --------------------------------------------

        successful_count = len(scraped_articles)
        failed_count = len(fact_urls) - successful_count # All non-successes
        logger.info(f"Scraping complete. Success: {successful_count}, Failed/Filtered/Exception: {failed_count}.", extra=extra_log_data)

        # --- Update used_links status ---
        updated_used_links = []
        links_updated_count = 0
        processed_in_this_node = set(processed_url_map.keys())

        for link in current_used_links:
            url = link.get('url')
            # Update status only if this node attempted to process this URL
            if url in processed_in_this_node:
                 status = processed_url_map[url]
                 # Create a new dict to avoid modifying the original state directly
                 updated_link = link.copy()
                 purpose = updated_link.get('purpose', '')
                 if status == 'processed':
                      updated_link['purpose'] = purpose.replace('(Fact)', '(Scraped Fact)') if '(Fact)' in purpose else f"{purpose} (Scraped Fact)"
                      updated_link['status'] = "processed"
                 elif status == 'failed_or_filtered':
                      updated_link['purpose'] = f"{purpose} (Scraping Failed/Filtered)"
                      updated_link['status'] = "failed_or_filtered"
                 elif status == 'exception':
                      updated_link['purpose'] = f"{purpose} (Scraping Exception)"
                      updated_link['status'] = "failed" # Treat exception as failure
                 else: # unknown_error
                      updated_link['purpose'] = f"{purpose} (Scraping Unknown Error)"
                      updated_link['status'] = "failed"

                 updated_used_links.append(updated_link)
                 links_updated_count += 1
            else:
                 # If not processed by this node, pass the link through unchanged
                 updated_used_links.append(link)

        logger.info(f"Updated status for {links_updated_count} entries in used_links related to this node's processing.", extra=extra_log_data)


        # --- 시간 기록 및 반환 ---
        end_time = datetime.now(timezone.utc)
        node5_processing_stats = (end_time - start_time).total_seconds()

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"Some errors occurred during scraping task execution: {final_error_message}", extra=extra_log_data)

        # 상태 업데이트 준비
        update_data = {
            "articles": scraped_articles,
            "used_links": updated_used_links,
            "node5_processing_stats": node5_processing_stats,
            "error_message": final_error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if final_error_message else logger.info
        log_level(f"News scraping result: {successful_count} articles scraped. Errors: {final_error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node5_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}