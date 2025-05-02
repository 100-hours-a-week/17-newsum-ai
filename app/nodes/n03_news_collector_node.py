# app/nodes/03_news_collector_node.py

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, TypedDict
from urllib.parse import urlparse
import aiohttp # 세션 관리를 위해 유지

# --- Tool Clients and Utilities ---
# 변경된 경로에서 도구 클래스 임포트
from app.tools.search.google import GoogleSearchTool
from app.tools.search.naver import NaverSearchTool
from app.tools.search.rss import RssSearchTool
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# --- 로거 설정 ---
logger = get_logger("NewsCollectorNode")

class NewsCollectorNode:
    """
    (Refactored for Tools - Naming Updated) 전용 검색 도구를 사용하여 뉴스 기사 URL을 수집합니다.
    - GoogleSearchTool, NaverSearchTool, RssSearchTool 사용.
    - 도메인 화이트리스트 적용 및 URL 중복 제거.
    """
    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["search_keywords", "trace_id", "used_links", "config", "processing_stats"]
    outputs: List[str] = ["fact_urls", "used_links", "processing_stats", "error_message"]

    # --- 요청하신 __init__ 시그니처 및 속성명 적용 ---
    def __init__(
        self,
        Google: GoogleSearchTool, # 파라미터명: Google (대문자 시작)
        naver: NaverSearchTool,   # 파라미터명: naver (소문자 시작)
        rss: RssSearchTool,     # 파라미터명: rss (소문자 시작)
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        # 속성명도 파라미터명과 동일하게 할당
        self.Google = Google
        self.naver = naver
        self.rss = rss
        # self.langsmith = langsmith_service
        # 신뢰 도메인 목록은 run 메서드에서 config를 통해 로드
        self.trusted_domains: List[str] = []
        logger.info("NewsCollectorNode initialized with search tools (Google, naver, rss).")

    def _load_config(self, config: Dict[str, Any]):
         """실행에 필요한 설정을 config 딕셔너리에서 로드하여 인스턴스 변수에 저장"""
         self.trusted_domains = config.get('trusted_news_domains', [])
         if not self.trusted_domains:
             logger.warning("Trusted news domains list is empty or not found in config.")
         else:
             logger.debug(f"Loaded trusted domains: {len(self.trusted_domains)} domains.")

    def _is_valid_news_url(self, url: str) -> bool:
        """URL 유효성 검사 및 신뢰 도메인 확인"""
        # 로직은 이전과 동일
        if not self.trusted_domains: return True
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc or parsed.scheme not in ['http', 'https']: return False
            domain = parsed.netloc.lower().replace('www.', '')
            return any(domain == trusted or domain.endswith('.' + trusted) for trusted in self.trusted_domains)
        except Exception: return False

    def _remove_duplicates(self, url_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """URL 리스트에서 중복 제거"""
        # 로직은 이전과 동일
        seen_urls: Set[str] = set()
        unique_list = []
        for item in url_list:
            url = item.get("url")
            if url and isinstance(url, str) and url not in seen_urls:
                seen_urls.add(url)
                unique_list.append(item)
        return unique_list

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """ (Refactored for Tools - Naming Updated) 검색 도구를 사용하여 뉴스 수집 실행 """
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing NewsCollectorNode...")

        search_keywords = state.search_keywords or []
        config = state.config or {}
        current_used_links = state.used_links or []
        processing_stats = state.processing_stats or {}

        self._load_config(config)
        max_results_per_api_call = config.get('search_result_count', 10)
        target_urls_per_keyword = config.get('target_urls_per_keyword', 5)
        http_timeout = config.get('http_timeout', 15)

        if not search_keywords:
            logger.error(f"{log_prefix} No search keywords provided.")
            return {"error_message": "Search keywords are missing."}
        # trusted_domains 로딩 실패 시 경고는 _load_config에서 로깅

        logger.info(f"{log_prefix} Starting news URL collection using tools for keywords: {search_keywords}")

        all_collected_items: List[Dict[str, str]] = []
        task_errors: List[str] = []

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=http_timeout)) as session:
            async def search_for_keyword(keyword: str):
                keyword_items = []
                processed_urls_in_kw = set()
                log_kw_prefix = f"{log_prefix}[KW:{keyword}]"

                # 1. Google 검색 (변경된 속성명 self.Google 사용)
                try:
                    google_results = await self.Google.search(keyword, max_results_per_api_call, session, state.trace_id)
                    for item in google_results:
                         url = item.get('url')
                         if url and url not in processed_urls_in_kw and self._is_valid_news_url(url):
                              keyword_items.append(item)
                              processed_urls_in_kw.add(url)
                    logger.debug(f"{log_kw_prefix} Google Search returned {len(google_results)} raw, collected {len(keyword_items)} valid.")
                except Exception as e:
                    msg = f"Google Search tool failed: {e}"
                    logger.error(f"{log_kw_prefix} {msg}")
                    task_errors.append(msg)

                # 2. Naver 검색 (변경된 속성명 self.naver 사용)
                if len(keyword_items) < target_urls_per_keyword:
                    try:
                        naver_results = await self.naver.search(keyword, max_results_per_api_call, session, state.trace_id)
                        for item in naver_results:
                            url = item.get('url')
                            if url and url not in processed_urls_in_kw and self._is_valid_news_url(url):
                                keyword_items.append(item)
                                processed_urls_in_kw.add(url)
                        logger.debug(f"{log_kw_prefix} Naver Search returned {len(naver_results)} raw, added valid count.")
                    except Exception as e:
                        msg = f"Naver Search tool failed: {e}"
                        logger.error(f"{log_kw_prefix} {msg}")
                        task_errors.append(msg)

                # 3. RSS 검색 (변경된 속성명 self.rss 사용)
                if len(keyword_items) < target_urls_per_keyword:
                    try:
                        needed = target_urls_per_keyword - len(keyword_items)
                        rss_results = await self.rss.search(keyword.lower(), needed * 2, state.trace_id)
                        added_from_rss = 0
                        for item in rss_results:
                            url = item.get('url')
                            if url and url not in processed_urls_in_kw and self._is_valid_news_url(url):
                                keyword_items.append(item)
                                processed_urls_in_kw.add(url)
                                added_from_rss += 1
                                if len(keyword_items) >= target_urls_per_keyword: break
                        logger.debug(f"{log_kw_prefix} RSS Search returned {len(rss_results)} raw, added {added_from_rss} new valid.")
                    except Exception as e:
                        msg = f"RSS Search tool failed: {e}"
                        logger.error(f"{log_kw_prefix} {msg}")
                        task_errors.append(msg)

                logger.info(f"{log_kw_prefix} Collected {len(keyword_items)} total valid URLs.")
                return keyword_items

            tasks = [search_for_keyword(keyword) for keyword in search_keywords]
            results_per_keyword = await asyncio.gather(*tasks, return_exceptions=True)

        # --- 결과 종합 및 처리 ---
        # (로직 동일)
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
        # (로직 동일)
        updated_used_links = list(current_used_links)
        existing_link_urls = {link.get('url') for link in updated_used_links if link.get('url')}
        added_link_count = 0
        for item in unique_items:
            url = item.get('url')
            if url and url not in existing_link_urls:
                updated_used_links.append({
                    'url': url,
                    'purpose': f"Collected from {item.get('source', 'Unknown')} for '{item.get('search_keyword', 'N/A')}' (News)"
                })
                existing_link_urls.add(url)
                added_link_count += 1
        logger.info(f"{log_prefix} Added {added_link_count} new URLs to used_links.")

        # --- 처리 시간 및 상태 반환 ---
        # (로직 동일)
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['news_collector_node_time'] = node_processing_time
        logger.info(f"{log_prefix} NewsCollectorNode finished in {node_processing_time:.2f} seconds.")

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"{log_prefix} Some errors occurred during URL collection: {final_error_message}")

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "fact_urls": unique_items,
            "used_links": updated_used_links,
            "processing_stats": processing_stats,
            "error_message": final_error_message # 오류 요약 메시지
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}