# app/nodes/05_news_scraper_node.py

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

# --- 리팩토링된 임포트 ---
from app.utils.logger import get_logger
from app.workflows.state import ComicState
# ArticleScraperTool 임포트 (경로 확인 필요)
from app.tools.scraping.article_scraper import ArticleScraperTool
from app.config.settings import settings # 기본 설정을 위해 임포트

# 로거 설정
logger = get_logger("NewsScraperNode") # 노드 이름으로 로거 가져오기

class NewsScraperNode:
    """
    (리팩토링됨) 수집된 뉴스 URL 목록에 대해 ArticleScraperTool을 사용하여
    기사 콘텐츠 스크래핑을 조율합니다.
    결과를 취합하고 언어 필터링을 적용한 후 상태를 업데이트합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["fact_urls", "trace_id", "used_links", "config", "processing_stats"]
    outputs: List[str] = ["articles", "used_links", "processing_stats", "error_message"]

    def __init__(self, scraper_tool: ArticleScraperTool):
        """
        NewsScraperNode 초기화.

        Args:
            scraper_tool (ArticleScraperTool): 기사 스크래핑 및 추출을 위한 도구.
        """
        if not scraper_tool:
             # 필수 도구이므로, 없으면 초기화 실패 처리 또는 로깅 강화
             logger.error("ArticleScraperTool이 주입되지 않았습니다! NewsScraperNode가 정상 작동할 수 없습니다.")
             # raise ValueError("ArticleScraperTool is required for NewsScraperNode")
        self.scraper_tool = scraper_tool

    async def _process_single_url_wrapper(
        self,
        url_info: Dict[str, str],
        config: Dict[str, Any],
        trace_id: str,
        comic_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        단일 URL 처리를 위한 래퍼 함수. ArticleScraperTool 호출 및 언어 필터링 수행.
        """
        original_url = url_info.get("url")
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'original_url': original_url}
        if not original_url:
            logger.warning("URL 정보가 누락되어 처리를 건너<0xEB><0x9C><0x95>니다.", extra=extra_log_data)
            return None

        if not self.scraper_tool:
             logger.error("Scraper tool is not available.", extra=extra_log_data)
             return None # 도구 없으면 처리 불가

        try:
            # 스크래핑 도구 호출
            article_data = await self.scraper_tool.scrape_article(original_url, trace_id, comic_id)

            if not article_data:
                # 스크래핑 실패 (이미 도구 내부에서 로깅됨)
                return None

            # 언어 필터링 적용
            language = article_data.get('language', 'und')
            allowed_languages = config.get('language_filter', settings.LANGUAGE_FILTER) # 설정 또는 기본값 사용

            if language != 'und' and language not in allowed_languages:
                logger.info(f"스크랩된 기사 건너<0xEB><0x9C><0x95>뛰기: 언어 '{language}'가 허용 목록 {allowed_languages}에 없습니다. URL: {article_data.get('url')}", extra=extra_log_data)
                # 언어 필터링으로 건너뛰는 경우도 실패로 간주하지 않고 None 반환
                return None

            # 원본 URL 소스 정보 추가 (어떤 검색어로 찾았는지 등)
            article_data['source_search_keyword'] = url_info.get('search_keyword')
            article_data['source_type'] = url_info.get('source') # 예: "Google News", "NAVER News"

            return article_data

        except Exception as e:
            # 래퍼 레벨에서 예외 발생 시 로깅 (보통은 scraper_tool 내부에서 처리됨)
            logger.exception(f"URL 처리 중 예상치 못한 오류 발생: {original_url} | 오류: {e}", extra=extra_log_data)
            return None


    async def execute(self, state: ComicState) -> Dict[str, Any]:
        """
        수집된 뉴스(fact) URL 목록에 대해 스크래핑 프로세스를 실행합니다.

        Args:
            state (ComicState): 현재 워크플로우 상태 객체.

        Returns:
            Dict[str, Any]: 상태 업데이트를 위한 사전 (articles, used_links, processing_stats 등).
        """
        start_time = datetime.now(timezone.utc) # 시작 시간 기록
        comic_id = state.comic_id
        trace_id = state.trace_id or comic_id or "unknown_trace" # trace_id 폴백
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # 로깅용 공통 데이터

        logger.info("NewsScraperNode 실행 시작...", extra=extra_log_data)

        # 상태에서 필요한 정보 안전하게 가져오기
        fact_urls = state.fact_urls
        config = state.config
        current_used_links = state.used_links
        processing_stats = state.processing_stats

        # 스크랩할 URL 없으면 종료
        if not fact_urls:
            logger.warning("스크랩할 뉴스 URL이 없습니다. NewsScraperNode를 건너<0xEB><0x9C><0x95>니다.", extra=extra_log_data)
            return {"articles": [], "used_links": current_used_links} # 오류가 아닌 빈 결과 반환

        logger.info(f"{len(fact_urls)}개의 뉴스 URL 스크래핑 시작...", extra=extra_log_data)

        # 설정에서 동시성 제한 가져오기
        concurrency_limit = config.get('scraper_concurrency', settings.SCRAPER_CONCURRENCY)
        semaphore = asyncio.Semaphore(concurrency_limit) # 세마포 생성

        tasks = [] # 비동기 태스크 목록
        # 각 URL 처리를 위한 태스크 생성
        for url_info in fact_urls:
             # 래퍼 함수를 세마포와 함께 실행하는 태스크 추가
             async def task_with_semaphore(ui):
                  async with semaphore:
                       return await self._process_single_url_wrapper(ui, config, trace_id, comic_id)
             tasks.append(task_with_semaphore(url_info))

        # 모든 스크래핑 태스크 병렬 실행 (세마포 제한 준수)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 처리: 성공/실패 집계 및 최종 기사 목록 생성
        scraped_articles: List[Dict[str, Any]] = [] # 성공적으로 스크랩된 기사 목록
        successful_urls = set() # 성공적으로 처리된 URL 집합 (원본 또는 AMP URL)
        failed_count = 0 # 실패 카운터

        for i, res in enumerate(results):
            original_url = fact_urls[i].get('url', 'Unknown URL') # 입력된 원본 URL
            if isinstance(res, dict) and res is not None: # 성공적으로 기사 데이터 반환 시
                scraped_articles.append(res)
                # 성공 URL 집합에 추가 (스크랩된 URL 및 원본 URL 모두)
                successful_urls.add(res.get('url'))
                if res.get('original_url') != res.get('url'):
                     successful_urls.add(res.get('original_url'))
            elif isinstance(res, Exception): # gather에서 예외 반환 시
                logger.error(f"스크래핑 태스크 실패 (gather 예외): {original_url} | 예외: {res}", exc_info=res, extra=extra_log_data)
                failed_count += 1
            # res가 None인 경우 (래퍼 함수에서 None 반환)는 이미 로깅되었으므로 여기서는 무시

        logger.info(f"스크래핑 완료. 성공적으로 처리된 기사 수: {len(scraped_articles)}. 실패/건너<0xEB><0x9C><0x94> URL 수: {len(fact_urls) - len(scraped_articles)}.", extra=extra_log_data)

        # used_links 상태 업데이트 (스크랩 성공/실패 상태 반영)
        updated_used_links = []
        processed_original_urls = set(fact_urls[i].get('url') for i, res in enumerate(results) if not isinstance(res, Exception) and res is not None)
        failed_original_urls = set(fact_urls[i].get('url') for i, res in enumerate(results) if isinstance(res, Exception) or res is None)


        for link in current_used_links:
            url = link.get('url')
            if url in processed_original_urls:
                 # 성공적으로 처리된 경우 (이 노드의 입력 URL 목록에 있었던 경우)
                 link['purpose'] = link.get('purpose', '').replace('(Fact)', '(Scraped Fact)') # 목적 업데이트
                 link['status'] = "processed" # 상태: 처리됨
                 updated_used_links.append(link)
            elif url in failed_original_urls:
                  # 처리 실패한 경우
                  link['purpose'] = link.get('purpose', '') + " (Scraping Failed)" # 실패 정보 추가
                  link['status'] = "failed" # 상태: 실패
                  updated_used_links.append(link)
            else:
                 # 이 노드에서 처리 대상이 아니었던 링크는 그대로 유지
                 updated_used_links.append(link)


        # --- 시간 기록 및 반환 ---
        end_time = datetime.now(timezone.utc) # 종료 시간 기록
        node_processing_time = (end_time - start_time).total_seconds() # 노드 처리 시간 계산
        processing_stats['node_05_news_scraper_time'] = node_processing_time # 통계 기록 (고유 키 사용)
        logger.info(f"NewsScraperNode 완료. 소요 시간: {node_processing_time:.2f} 초.", extra=extra_log_data)

        # 상태 업데이트 준비
        updates = {
            "articles": scraped_articles, # 스크랩된 기사 목록
            "used_links": updated_used_links, # 업데이트된 사용 링크 목록
            "processing_stats": processing_stats # 업데이트된 처리 통계
        }
        # 이 노드 관련 이전 오류 메시지 상태 초기화
        current_error = state.error_message
        if current_error and "NewsScraperNode" in current_error:
             updates["error_message"] = None # 성공 시 None으로 설정

        return updates # 상태 업데이트 반환