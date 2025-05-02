# app/nodes/05_news_scraper_node.py (Improved Version)

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# 프로젝트 구성 요소 임포트
from app.utils.logger import get_logger
from app.workflows.state import ComicState
from app.tools.scraping.article_scraper import ArticleScraperTool
from app.config.settings import settings # 기본 설정을 위해 임포트

# 로거 설정
logger = get_logger(__name__)

class NewsScraperNode:
    """
    수집된 뉴스 URL 목록에 대해 ArticleScraperTool을 사용하여 기사 콘텐츠 스크래핑을 조율합니다.
    결과를 취합하고 언어 필터링을 적용한 후 상태를 업데이트합니다.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["fact_urls", "trace_id", "comic_id", "used_links", "config", "processing_stats"]
    outputs: List[str] = ["articles", "used_links", "processing_stats", "error_message"]

    # 의존성 주입
    def __init__(self, scraper_tool: ArticleScraperTool):
        if not scraper_tool:
             logger.error("ArticleScraperTool is required but not provided!")
             # 필수 의존성이므로 에러 발생시키는 것이 좋을 수 있음
             raise ValueError("ArticleScraperTool is required for NewsScraperNode")
        self.scraper_tool = scraper_tool
        logger.info("NewsScraperNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.allowed_languages = config.get('language_filter', settings.LANGUAGE_FILTER)
        self.concurrency_limit = config.get('scraper_concurrency', settings.SCRAPER_CONCURRENCY)
        logger.debug(f"Runtime config loaded. Allowed languages: {self.allowed_languages}, Concurrency: {self.concurrency_limit}")

    async def _process_single_url_wrapper(
        self,
        url_info: Dict[str, Any], # url_info는 source, search_keyword 등을 포함할 수 있음
        trace_id: str,
        comic_id: str
    ) -> Optional[Dict[str, Any]]:
        """단일 URL 처리 래퍼: 스크래핑 및 언어 필터링"""
        original_url = url_info.get("url")
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'original_url': original_url}
        if not original_url:
            logger.warning("URL info is missing 'url' field. Skipping.", extra=extra_log_data)
            return None

        try:
            # 스크래핑 도구 호출 (도구 내부에 재시도 로직 포함 가정)
            # ArticleScraperTool.scrape_article은 trace_id, comic_id를 받을 수 있어야 함
            article_data = await self.scraper_tool.scrape_article(original_url, trace_id, comic_id)

            if not article_data or not article_data.get('text'):
                # 스크래핑 실패 또는 텍스트 없음 (도구 내부에서 로깅됨 가정)
                logger.warning(f"Scraping failed or no text extracted for {original_url}.", extra=extra_log_data)
                return None

            # 언어 필터링 적용
            language = article_data.get('language', 'und')
            if language != 'und' and language not in self.allowed_languages:
                logger.info(f"Skipping scraped article: Language '{language}' not in allowed list {self.allowed_languages}. URL: {article_data.get('url')}", extra=extra_log_data)
                return None # 필터링됨

            # 원본 URL 소스 정보 추가 (어떤 검색어로 찾았는지 등)
            article_data['source_search_keyword'] = url_info.get('search_keyword')
            article_data['source_type'] = url_info.get('source') # 예: "Google", "Naver", "RSS"
            # 스크랩 성공 시 원본 URL도 명시적으로 추가 (used_links 업데이트용)
            article_data['original_url_from_source'] = original_url

            return article_data

        except Exception as e:
            # 래퍼 레벨 또는 도구 호출의 최종 예외 처리
            logger.exception(f"Unexpected error processing URL: {original_url} | Error: {e}", extra=extra_log_data)
            return None

    # --- 메인 실행 메서드 (run으로 이름 변경) ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """수집된 뉴스 URL 목록에 대해 스크래핑 프로세스를 실행합니다."""
        start_time = datetime.now(timezone.utc)
        comic_id = state.comic_id
        trace_id = state.trace_id
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        log_prefix = f"[{trace_id}]" # run 메서드 내에서는 log_prefix 사용

        logger.info(f"{log_prefix} NewsScraperNode starting...")

        fact_urls = state.fact_urls or []
        config = state.config or {}
        current_used_links = state.used_links or []
        processing_stats = state.processing_stats or {}

        if not fact_urls:
            logger.warning(f"{log_prefix} No fact URLs to scrape. Skipping.", extra=extra_log_data)
            processing_stats['news_scraper_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"articles": [], "used_links": current_used_links, "processing_stats": processing_stats}

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        # --------------------------

        logger.info(f"{log_prefix} Starting scraping for {len(fact_urls)} news URLs (Concurrency: {self.concurrency_limit})...", extra=extra_log_data)

        tasks = []
        for url_info in fact_urls:
             async def task_with_semaphore(ui):
                  async with semaphore:
                       # wrapper 함수에 trace_id, comic_id 전달
                       return await self._process_single_url_wrapper(ui, trace_id, comic_id)
             tasks.append(task_with_semaphore(url_info))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped_articles: List[Dict[str, Any]] = []
        processed_url_map: Dict[str, str] = {} # key: original_url, value: status ('processed', 'failed', 'filtered')
        task_errors: List[str] = []

        for i, res in enumerate(results):
            original_url = fact_urls[i].get('url', f'Unknown_URL_{i}')
            if isinstance(res, dict) and res is not None:
                scraped_articles.append(res)
                processed_url_map[original_url] = 'processed'
            elif isinstance(res, Exception):
                # 래퍼에서 이미 로깅됨, 여기서는 상태 기록 및 에러 메시지 추가
                processed_url_map[original_url] = 'failed'
                task_errors.append(f"Scraping failed for {original_url}: {res}")
            else: # res is None (스크랩 실패 또는 필터링됨)
                 processed_url_map[original_url] = 'filtered_or_failed' # 좀 더 명확하게

        successful_count = len(scraped_articles)
        failed_or_filtered_count = len(fact_urls) - successful_count
        logger.info(f"{log_prefix} Scraping complete. Successfully processed articles: {successful_count}. Failed/Filtered URLs: {failed_or_filtered_count}.", extra=extra_log_data)

        # used_links 상태 업데이트
        updated_used_links = []
        links_updated_count = 0
        for link in current_used_links:
            url = link.get('url')
            if url in processed_url_map: # 이 노드에서 처리 시도한 URL
                 status = processed_url_map[url]
                 if status == 'processed':
                      link['purpose'] = link.get('purpose', '').replace('(Fact)', '(Scraped Fact)')
                      link['status'] = "processed"
                 elif status == 'failed':
                      link['purpose'] += " (Scraping Failed)"
                      link['status'] = "failed"
                 else: # filtered_or_failed
                      link['purpose'] += " (Scraping Failed/Filtered)"
                      link['status'] = "filtered_or_failed" # 상태 명시
                 updated_used_links.append(link)
                 links_updated_count += 1
            else: # 이 노드에서 처리 안 한 링크
                 updated_used_links.append(link)
        logger.info(f"{log_prefix} Updated status for {links_updated_count} entries in used_links.")


        # --- 시간 기록 및 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['news_scraper_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} NewsScraperNode finished. Elapsed time: {processing_stats['news_scraper_node_time']:.2f} seconds.", extra=extra_log_data)

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"{log_prefix} Some errors occurred during scraping: {final_error_message}")

        # 상태 업데이트 준비
        updates = {
            "articles": scraped_articles,
            "used_links": updated_used_links,
            "processing_stats": processing_stats,
            "error_message": final_error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in updates.items() if k in valid_keys}