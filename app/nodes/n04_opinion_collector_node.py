# app/nodes/04_opinion_collector_node.py (Improved Version)

import asyncio
import random
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone
from collections import defaultdict
import re
from urllib.parse import urlparse

# 프로젝트 구성 요소 임포트
from app.utils.logger import get_logger
from app.workflows.state import ComicState
from app.tools.social.twitter import TwitterTool
from app.tools.social.reddit import RedditTool
from app.tools.search.google import GoogleSearchTool
from app.config.settings import settings # 기본 설정을 위해 임포트

# 로거 설정
logger = get_logger(__name__)

class OpinionCollectorNode:
    """
    주입된 도구(Twitter, Reddit, Google Search)를 사용하여 키워드 관련 의견 URL/데이터를 수집합니다.
    다양한 도구 호출을 조율하고, 결과를 취합하며, 밸런싱 및 중복 제거 로직을 적용하고,
    워크플로우 상태를 업데이트합니다.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["search_keywords", "trace_id", "comic_id", "used_links", "config", "processing_stats"]
    outputs: List[str] = ["opinion_urls", "used_links", "processing_stats", "error_message"]

    # 의존성 주입 (일관된 파라미터명 사용 권장)
    def __init__(
        self,
        twitter_tool: Optional[TwitterTool] = None,
        reddit_tool: Optional[RedditTool] = None,
        Google_Search_tool: Optional[GoogleSearchTool] = None # 파라미터명 일관되게 변경
        # langsmith_service: Optional[LangSmithService] = None
    ):
        self.twitter_tool = twitter_tool
        self.reddit_tool = reddit_tool
        self.Google_Search_tool = Google_Search_tool # 내부 속성명도 통일
        # self.langsmith = langsmith_service

        # 도구 누락 시 경고 로깅
        if not self.twitter_tool: logger.warning("TwitterTool not injected. Twitter search will be skipped.")
        if not self.reddit_tool: logger.warning("RedditTool not injected. Reddit search will be skipped.")
        if not self.Google_Search_tool: logger.warning("GoogleSearchTool not injected. YouTube, Blog, Community search will be skipped.")
        logger.info("OpinionCollectorNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.yt_max = config.get('YOUTUBE_OPINION_MAX_RESULTS', settings.YOUTUBE_OPINION_MAX_RESULTS)
        self.tw_max = config.get('TWITTER_OPINION_MAX_RESULTS', settings.TWITTER_OPINION_MAX_RESULTS)
        self.rd_max = config.get('REDDIT_OPINION_MAX_RESULTS', settings.REDDIT_OPINION_MAX_RESULTS)
        self.bl_max = config.get('BLOG_OPINION_MAX_RESULTS', settings.BLOG_OPINION_MAX_RESULTS)
        self.cm_max = config.get('COMMUNITY_OPINION_MAX_RESULTS', settings.COMMUNITY_OPINION_MAX_RESULTS)
        self.max_total = config.get('MAX_TOTAL_OPINIONS_TARGET', settings.MAX_TOTAL_OPINIONS_TARGET)
        # 딕셔너리 형태 설정 로드 주의 (get으로 가져오면 None일 수 있음)
        self.max_per_platform = config.get('MAX_OPINIONS_PER_PLATFORM_SAMPLING', settings.MAX_OPINIONS_PER_PLATFORM_SAMPLING)
        self.min_per_platform = config.get('MIN_OPINIONS_PER_PLATFORM_SAMPLING', settings.MIN_OPINIONS_PER_PLATFORM_SAMPLING)
        self.concurrency = config.get('OPINION_COLLECTOR_CONCURRENCY', settings.OPINION_COLLECTOR_CONCURRENCY)
        self.target_blog_domains = config.get('target_blog_domains', settings.TARGET_BLOG_DOMAINS)
        self.target_community_domains = config.get('target_community_domains', settings.TARGET_COMMUNITY_DOMAINS)

        # 딕셔너리 설정 유효성 검사
        if not isinstance(self.max_per_platform, dict): self.max_per_platform = {}
        if not isinstance(self.min_per_platform, dict): self.min_per_platform = {}

        logger.debug(f"Runtime config loaded. MaxTotal: {self.max_total}, Concurrency: {self.concurrency}")
        logger.debug(f"Max per platform: {self.max_per_platform}")
        logger.debug(f"Min per platform: {self.min_per_platform}")

    # --- 도구 호출 래퍼 ---
    async def _fetch_with_tool(self, tool_method, keyword: str, max_results: int, trace_id: str, comic_id: str, source_name: str) -> List[Dict[str, Any]]:
        """특정 도구 메서드를 호출하고 결과를 처리하는 공통 래퍼"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'keyword': keyword, 'source': source_name}
        if not tool_method: return []
        try:
            logger.debug(f"Fetching {source_name} for '{keyword}'...", extra=extra_log_data)
            # 도구 메서드 호출 (일부 도구는 추가 인자 필요 없을 수 있음)
            # 도구 인터페이스 통일 필요: keyword, max_results, trace_id 정도는 공통으로 받도록
            results = await tool_method(keyword=keyword, max_results=max_results, trace_id=trace_id)
            processed_results = []
            for r in results:
                 # 각 결과에 source와 keyword 정보 추가 (도구가 안 해주면 여기서)
                 r.setdefault("source", source_name)
                 r.setdefault("search_keyword", keyword)
                 processed_results.append(r)
            logger.debug(f"Received {len(processed_results)} results from {source_name} for '{keyword}'.", extra=extra_log_data)
            return processed_results
        except Exception as e:
            logger.error(f"Error calling {source_name} tool for '{keyword}': {e}", exc_info=True, extra=extra_log_data)
            return []

    # --- 헬퍼 함수 (중복제거, 밸런싱, URL 확인) ---
    def _is_target_domain(self, url: str, target_domains: List[str]) -> bool:
        """URL이 지정된 타겟 도메인 목록에 속하는지 확인"""
        if not url or not target_domains: return False
        try:
            domain = urlparse(url).netloc.lower().replace('www.', '')
            return any(domain == target or domain.endswith('.' + target) for target in target_domains)
        except Exception: return False

    def _balance_and_sample_results(
        self, platform_results: Dict[str, List[Dict[str, Any]]],
        trace_id: str, comic_id: str
        ) -> List[Dict[str, Any]]:
        """설정된 제한(min/max per platform, max total)에 따라 결과를 밸런싱 및 샘플링"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        balanced_results: List[Dict[str, Any]] = []
        final_urls: Set[str] = set()

        # 플랫폼 순서 랜덤화 (매번 동일한 플랫폼 우선 방지)
        platform_order = list(platform_results.keys())
        random.shuffle(platform_order)

        # Pass 1: 플랫폼별 최소 결과 수 보장
        logger.debug("Balancing Pass 1: Ensuring min results per platform...", extra=extra_log_data)
        temp_platform_results = {k: list(v) for k, v in platform_results.items()} # 원본 수정 방지용 복사

        for platform in platform_order:
            results = temp_platform_results.get(platform, [])
            if not results: continue
            min_needed = self.min_per_platform.get(platform, settings.DEFAULT_MIN_OPINIONS_PER_PLATFORM) # settings 기본값 사용
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

            # 사용된 아이템은 임시 결과에서 제거
            temp_platform_results[platform] = [item for item in results if item not in items_to_remove_from_temp]
            logger.debug(f"Pass 1 ({platform}): Added {added_count} (Min: {min_needed}). Remaining: {len(temp_platform_results[platform])}", extra=extra_log_data)

        # Pass 2: 플랫폼별 최대 및 전체 최대 결과 수까지 채움
        logger.debug("Balancing Pass 2: Filling up to max limits...", extra=extra_log_data)
        all_remaining_items = []
        platform_current_counts = defaultdict(int)
        for item in balanced_results: # 현재까지 추가된 항목 카운트
             source_platform = item.get("source", "Unknown") # 소스 이름 기준으로 카운트
             platform_current_counts[source_platform] += 1

        # 남은 항목들 리스트로 모으기 (플랫폼별 최대치 고려)
        for platform in platform_order:
            results = temp_platform_results.get(platform, [])
            if not results: continue
            max_allowed = self.max_per_platform.get(platform, settings.DEFAULT_MAX_OPINIONS_PER_PLATFORM)
            current_count = platform_current_counts.get(platform, 0)
            can_add_count = max(0, max_allowed - current_count)
            if can_add_count > 0:
                items_to_consider = results[:can_add_count] # 추가 가능한 만큼만 고려
                all_remaining_items.extend(items_to_consider)
                logger.debug(f"Pass 2 ({platform}): Max={max_allowed}, Current={current_count}, CanAdd={can_add_count}. Considering {len(items_to_consider)} items.", extra=extra_log_data)

        random.shuffle(all_remaining_items) # 모든 남은 항목 랜덤 셔플
        added_pass_2 = 0
        for item in all_remaining_items:
            if len(balanced_results) >= self.max_total: break # 전체 최대 도달 시 중단
            url = item.get("url")
            if url and url not in final_urls:
                 source_platform = item.get("source", "Unknown")
                 # 해당 플랫폼의 최대치 다시 확인
                 if platform_current_counts[source_platform] < self.max_per_platform.get(source_platform, settings.DEFAULT_MAX_OPINIONS_PER_PLATFORM):
                      balanced_results.append(item)
                      final_urls.add(url)
                      platform_current_counts[source_platform] += 1
                      added_pass_2 += 1

        logger.info(f"Balancing complete. Added {added_pass_2} in Pass 2. Final sampled count: {len(balanced_results)}", extra=extra_log_data)
        # 최종 결과도 랜덤하게 섞어줌 (선택적)
        random.shuffle(balanced_results)
        return balanced_results

    def _remove_duplicates(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """아이템 목록에서 URL 기준으로 중복을 제거합니다."""
        seen_urls: Set[str] = set()
        unique_list = []
        for item in items:
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_list.append(item)
        return unique_list

    # --- 메인 실행 메서드 (run으로 이름 변경) ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """주입된 도구를 사용하여 의견 수집 프로세스를 실행합니다."""
        start_time = datetime.now(timezone.utc)
        # comic_id 와 trace_id를 상태에서 안전하게 가져오기
        comic_id = getattr(state, 'comic_id', 'unknown_comic')
        trace_id = getattr(state, 'trace_id', comic_id) # comic_id로 fallback
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}

        logger.info("OpinionCollectorNode starting...", extra=extra_log_data)

        search_keywords = state.search_keywords
        if not search_keywords:
            logger.warning("No search keywords found in state. Skipping opinion collection.", extra=extra_log_data)
            processing_stats = getattr(state, 'processing_stats', {})
            processing_stats['opinion_collector_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"opinion_urls": [], "processing_stats": processing_stats, "error_message": "Search keywords are missing."}

        config = state.config or {}
        current_used_links = state.used_links or []
        processing_stats = state.processing_stats or {}

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        semaphore = asyncio.Semaphore(self.concurrency)
        # --------------------------

        # --- 태스크 정의 및 실행 ---
        tasks = []
        task_errors: List[str] = [] # 오류 메시지 수집

        async def throttled_task(coro):
             async with semaphore: return await coro

        # 사용 가능한 도구에 대해서만 태스크 생성
        tool_methods = []
        if self.twitter_tool: tool_methods.append((self.twitter_tool.search_recent_tweets, self.tw_max, "Twitter"))
        if self.reddit_tool: tool_methods.append((self.reddit_tool.search_posts, self.rd_max, "Reddit"))
        if self.Google_Search_tool:
            tool_methods.append((self.Google_Search_tool.search_youtube_videos, self.yt_max, "YouTube"))
            # 블로그와 커뮤니티는 GoogleSearchTool의 다른 메서드 사용 가정
            tool_methods.append((self.Google_Search_tool.search_blogs_via_cse, self.bl_max, "Blog"))
            tool_methods.append((self.Google_Search_tool.search_communities_via_cse, self.cm_max, "Community"))

        if not tool_methods:
             logger.error("No opinion collection tools available or injected.", extra=extra_log_data)
             processing_stats['opinion_collector_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
             return {"opinion_urls": [], "processing_stats": processing_stats, "error_message": "No opinion collection tools configured."}

        for keyword in search_keywords:
             for tool_method, max_res, source_name in tool_methods:
                 tasks.append(throttled_task(self._fetch_with_tool(tool_method, keyword, max_res, trace_id, comic_id, source_name)))

        logger.info(f"Executing {len(tasks)} opinion collection tasks (Concurrency: {self.concurrency})...", extra=extra_log_data)
        api_results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        # --- 결과 처리 및 취합 ---
        platform_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        total_collected_count = 0

        for result in api_results_nested:
            if isinstance(result, Exception):
                # _fetch_with_tool 에서 이미 로깅되었으므로, 여기서는 에러 메시지만 집계
                task_errors.append(f"Task failed: {result}")
            elif isinstance(result, list):
                 total_collected_count += len(result)
                 for item in result:
                      source = item.get("source", "Unknown")
                      # 블로그/커뮤니티 필터링
                      if source == "Blog" and not self._is_target_domain(item.get('url'), self.target_blog_domains):
                           continue # 타겟 블로그 아니면 건너뜀
                      if source == "Community" and not self._is_target_domain(item.get('url'), self.target_community_domains):
                           continue # 타겟 커뮤니티 아니면 건너뜀
                      platform_results[source].append(item) # source 이름 그대로 사용

        logger.info(f"Initial collection complete. Raw items collected: {total_collected_count}. Task errors: {len(task_errors)}.", extra=extra_log_data)

        # 플랫폼 내 중복 제거 (URL 기준)
        unique_platform_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        total_after_dedupe = 0
        for platform, items in platform_results.items():
             unique_items = self._remove_duplicates(items)
             unique_platform_results[platform] = unique_items
             total_after_dedupe += len(unique_items)
             logger.debug(f"{platform}: {len(items)} -> {len(unique_items)} after deduplication.", extra=extra_log_data)
        logger.info(f"Total items after deduplication: {total_after_dedupe}.")

        # 플랫폼 간 밸런싱 및 샘플링
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
            if url and url not in existing_link_urls:
                link_info = {
                    'url': url,
                    'purpose': f"Collected from {item.get('source', 'Unknown')} for keyword '{item.get('search_keyword', 'Unknown')}' (Opinion)",
                    'node': 'OpinionCollectorNode',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'collected' # 초기 상태
                }
                updated_used_links.append(link_info)
                existing_link_urls.add(url)
                added_link_count += 1
        logger.info(f"Added {added_link_count} new opinion URLs to used_links.", extra=extra_log_data)

        # --- 시간 기록 및 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['opinion_collector_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"OpinionCollectorNode finished. Elapsed time: {processing_stats['opinion_collector_node_time']:.2f} seconds.", extra=extra_log_data)

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"Some errors occurred during opinion collection: {final_error_message}")

        # 상태 업데이트 준비
        updates = {
            "opinion_urls": balanced_sampled_opinions, # 최종 샘플링된 결과
            "used_links": updated_used_links,
            "processing_stats": processing_stats,
            "error_message": final_error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in updates.items() if k in valid_keys}