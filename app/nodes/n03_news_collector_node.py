# app/nodes/03_news_collector_node.py (Improved Version)

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
import aiohttp # 세션 관리를 위해 유지

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 기본값 참조용
from app.tools.search.google import GoogleSearchTool
from app.tools.search.naver import NaverSearchTool
from app.tools.search.rss import RssSearchTool
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# 로거 설정
logger = get_logger(__name__)

class NewsCollectorNode:
    """
    전용 검색 도구를 사용하여 뉴스 기사 URL을 수집합니다.
    - GoogleSearchTool, NaverSearchTool, RssSearchTool 사용 (의존성 주입).
    - 도메인 화이트리스트 적용 및 URL 중복 제거.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """
    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["search_keywords", "trace_id", "used_links", "config", "processing_stats"]
    outputs: List[str] = ["fact_urls", "used_links", "processing_stats", "error_message"]

    # 의존성 주입 (일관된 파라미터명 사용)
    def __init__(
        self,
        Google_Search_tool: GoogleSearchTool, # 파라미터명 변경 (Google_Search_tool)
        naver_search_tool: NaverSearchTool,   # 파라미터명 변경 (naver_search_tool)
        rss_search_tool: RssSearchTool,     # 파라미터명 변경 (rss_search_tool)
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        # 내부 속성명도 일관되게 변경
        self.Google_Search_tool = Google_Search_tool
        self.naver_search_tool = naver_search_tool
        self.rss_search_tool = rss_search_tool
        # self.langsmith = langsmith_service
        # 신뢰 도메인 목록은 run 메서드 시작 시 config를 통해 로드
        self.trusted_domains: List[str] = []
        logger.info("NewsCollectorNode initialized with search tools (Google, Naver, RSS).")

    def _load_runtime_config(self, config: Dict[str, Any]):
         """실행에 필요한 설정을 config 딕셔너리에서 로드하여 인스턴스 변수에 저장"""
         # state.config 우선, 없으면 settings 사용
         self.trusted_domains = config.get('trusted_news_domains', settings.TRUSTED_NEWS_DOMAINS)
         self.max_results_per_call = config.get('search_result_count', settings.DEFAULT_SEARCH_RESULTS)
         self.target_urls_per_kw = config.get('target_urls_per_keyword', settings.DEFAULT_TARGET_URLS_PER_KW)
         self.http_timeout = config.get('http_timeout', settings.DEFAULT_HTTP_TIMEOUT)

         if not self.trusted_domains:
             logger.warning("Trusted news domains list is empty (check config/settings). Filtering disabled.")
         else:
             logger.debug(f"Loaded trusted domains: {len(self.trusted_domains)} domains.")
         logger.debug(f"Max results/call: {self.max_results_per_call}, Target URLs/kw: {self.target_urls_per_kw}, Timeout: {self.http_timeout}s")

    def _is_valid_news_url(self, url: str) -> bool:
        """URL 유효성 검사 및 신뢰 도메인 확인"""
        if not url or not isinstance(url, str): return False
        # 신뢰 도메인 목록이 비어 있으면 모든 유효한 형식의 URL 허용
        if not self.trusted_domains:
             try:
                  parsed = urlparse(url)
                  return bool(parsed.scheme in ['http', 'https'] and parsed.netloc)
             except Exception: return False

        # 신뢰 도메인 목록이 있으면 해당 도메인만 허용
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc or parsed.scheme not in ['http', 'https']: return False
            domain = parsed.netloc.lower().replace('www.', '')
            return any(domain == trusted or domain.endswith('.' + trusted) for trusted in self.trusted_domains)
        except Exception: return False

    def _remove_duplicates(self, url_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """URL 리스트에서 중복 제거 (URL 기준)"""
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
        trace_id = state.trace_id # 초기화 노드에서 반드시 설정됨
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing NewsCollectorNode...")

        search_keywords = state.search_keywords or []
        config = state.config or {}
        current_used_links = state.used_links or []
        processing_stats = state.processing_stats or {}

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        # --------------------------

        if not search_keywords:
            logger.error(f"{log_prefix} No search keywords provided.")
            processing_stats['news_collector_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"error_message": "Search keywords are missing.", "processing_stats": processing_stats}

        logger.info(f"{log_prefix} Starting news URL collection using tools for keywords: {search_keywords}")

        all_collected_items: List[Dict[str, str]] = []
        task_errors: List[str] = []

        # aiohttp 세션을 사용하여 타임아웃 관리
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.http_timeout)) as session:
            async def search_for_keyword(keyword: str):
                keyword_items = []
                processed_urls_in_kw = set()
                log_kw_prefix = f"{log_prefix}[KW:{keyword}]"
                urls_needed = self.target_urls_per_kw

                # 1. Google 검색 (변경된 속성명 사용)
                try:
                    # GoogleSearchTool.search 는 session과 trace_id를 받을 수 있도록 수정되었다고 가정
                    google_results = await self.Google_Search_tool.search(keyword, self.max_results_per_call, session, trace_id)
                    added_count = 0
                    for item in google_results:
                         url = item.get('url')
                         if url and url not in processed_urls_in_kw and self._is_valid_news_url(url):
                              item['source'] = item.get('source', 'Google') # 소스 명시
                              item['search_keyword'] = keyword # 검색어 추가
                              keyword_items.append(item)
                              processed_urls_in_kw.add(url)
                              added_count += 1
                              if len(keyword_items) >= urls_needed: break # 목표 개수 도달 시 중단
                    logger.debug(f"{log_kw_prefix} Google Search: {added_count} valid URLs added.")
                except Exception as e:
                    msg = f"Google Search tool failed: {e}"
                    logger.error(f"{log_kw_prefix} {msg}", exc_info=True) # 상세 오류 로깅
                    task_errors.append(msg)

                # 2. Naver 검색 (필요한 경우)
                if len(keyword_items) < urls_needed:
                    try:
                        naver_results = await self.naver_search_tool.search(keyword, self.max_results_per_call, session, trace_id)
                        added_count = 0
                        for item in naver_results:
                            url = item.get('url')
                            if url and url not in processed_urls_in_kw and self._is_valid_news_url(url):
                                item['source'] = item.get('source', 'Naver')
                                item['search_keyword'] = keyword
                                keyword_items.append(item)
                                processed_urls_in_kw.add(url)
                                added_count += 1
                                if len(keyword_items) >= urls_needed: break
                        logger.debug(f"{log_kw_prefix} Naver Search: {added_count} valid URLs added.")
                    except Exception as e:
                        msg = f"Naver Search tool failed: {e}"
                        logger.error(f"{log_kw_prefix} {msg}", exc_info=True)
                        task_errors.append(msg)

                # 3. RSS 검색 (필요한 경우)
                if len(keyword_items) < urls_needed:
                    try:
                        # RSS는 session 불필요 가정, 필요한 만큼 요청
                        needed_for_rss = urls_needed - len(keyword_items)
                        # 약간 더 많이 요청 (필터링 감안)
                        rss_results = await self.rss_search_tool.search(keyword, needed_for_rss * 2, trace_id)
                        added_count = 0
                        for item in rss_results:
                            url = item.get('url')
                            if url and url not in processed_urls_in_kw and self._is_valid_news_url(url):
                                item['source'] = item.get('source', 'RSS')
                                item['search_keyword'] = keyword
                                keyword_items.append(item)
                                processed_urls_in_kw.add(url)
                                added_count += 1
                                if len(keyword_items) >= urls_needed: break
                        logger.debug(f"{log_kw_prefix} RSS Search: {added_count} valid URLs added.")
                    except Exception as e:
                        msg = f"RSS Search tool failed: {e}"
                        logger.error(f"{log_kw_prefix} {msg}", exc_info=True)
                        task_errors.append(msg)

                logger.info(f"{log_kw_prefix} Collected {len(keyword_items)} total valid URLs (Target: {urls_needed}).")
                return keyword_items

            tasks = [search_for_keyword(keyword) for keyword in search_keywords]
            results_per_keyword = await asyncio.gather(*tasks, return_exceptions=True)

        # --- 결과 종합 및 처리 ---
        for i, result in enumerate(results_per_keyword):
            keyword = search_keywords[i]
            if isinstance(result, list):
                all_collected_items.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"{log_prefix} Error gathering results for keyword '{keyword}': {result}")
                task_errors.append(f"Gather error for '{keyword}': {result}")

        unique_items = self._remove_duplicates(all_collected_items)
        logger.info(f"{log_prefix} Total unique news URLs collected: {len(unique_items)}")

        # --- 사용된 링크 추적 업데이트 ---
        updated_used_links = list(current_used_links)
        existing_link_urls = {link.get('url') for link in updated_used_links if link.get('url')}
        added_link_count = 0
        for item in unique_items:
            url = item.get('url')
            if url and url not in existing_link_urls:
                # link 정보에 source와 keyword 추가
                updated_used_links.append({
                    'url': url,
                    'purpose': f"Collected from {item.get('source', 'Unknown')} for '{item.get('search_keyword', 'N/A')}' (Fact)",
                    'node': 'NewsCollectorNode',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'collected' # 초기 상태
                })
                existing_link_urls.add(url)
                added_link_count += 1
        logger.info(f"{log_prefix} Added {added_link_count} new URLs to used_links.")

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['news_collector_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} NewsCollectorNode finished in {processing_stats['news_collector_node_time']:.2f} seconds.")

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"{log_prefix} Some errors occurred during URL collection: {final_error_message}")

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "fact_urls": unique_items, # source, search_keyword 포함된 아이템 리스트
            "used_links": updated_used_links,
            "processing_stats": processing_stats,
            "error_message": final_error_message # 오류 요약 메시지
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}