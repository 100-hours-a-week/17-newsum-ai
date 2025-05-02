# app/nodes/04_opinion_collector_node.py

import asyncio
import random
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone
from collections import defaultdict
import re # _is_blog_url 확인용
from urllib.parse import urlparse # _is_blog_url 확인용

# --- 리팩토링된 임포트 ---
from app.utils.logger import get_logger
from app.workflows.state import ComicState
# Tool 클래스들이 아래 경로에 존재한다고 가정합니다 (이전 단계에서 구현됨)
from app.tools.social.twitter import TwitterTool # 파일명 변경 반영
from app.tools.social.reddit import RedditTool     # 파일명 변경 반영
from app.tools.search.google import GoogleSearchTool # 파일명 변경 반영
from app.config.settings import settings # 기본 설정을 위해 임포트

# 로거 설정
logger = get_logger("OpinionCollectorNode") # 노드 이름으로 로거 가져오기

class OpinionCollectorNode:
    """
    (리팩토링됨) 주입된 도구(Twitter, Reddit, Google Search)를 사용하여 키워드 관련 의견 URL을 수집합니다.
    다양한 도구 호출을 조율하고, 결과를 취합하며, 밸런싱 및 중복 제거 로직을 적용하고,
    워크플로우 상태를 업데이트합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["search_keywords", "trace_id", "used_links", "config", "processing_stats"]
    outputs: List[str] = ["opinion_urls", "used_links", "processing_stats", "error_message"]

    def __init__(
        self,
        twitter_tool: Optional[TwitterTool] = None,
        reddit_tool: Optional[RedditTool] = None,
        google_tool: Optional[GoogleSearchTool] = None # 파라미터 이름은 google_tool 유지
    ):
        """
        외부 플랫폼 접근을 위한 주입된 도구들로 노드를 초기화합니다.

        Args:
            twitter_tool (Optional[TwitterTool]): 트위터 상호작용 도구.
            reddit_tool (Optional[RedditTool]): 레딧 상호작용 도구.
            google_tool (Optional[GoogleSearchTool]): 구글 검색 도구 (YouTube, Blogs, Communities).
        """
        # --- 수정: 인스턴스 변수명을 파라미터명과 일치시키도록 변경 ---
        self.twitter_tool = twitter_tool
        self.reddit_tool = reddit_tool
        self.google_tool = google_tool
        # --- 수정 끝 ---

        # 도구 누락 시 경고 로깅 (수정된 변수명 사용)
        if not self.twitter_tool:
            logger.warning("TwitterTool이 주입되지 않았습니다. 트위터 검색을 건너<0xEB><0x9C><0x95>니다.")
        if not self.reddit_tool:
            logger.warning("RedditTool이 주입되지 않았습니다. 레딧 검색을 건너<0xEB><0x9C><0x95>니다.")
        if not self.google_tool:
            logger.warning("GoogleSearchTool이 주입되지 않았습니다. 유튜브, 블로그, 커뮤니티 검색을 건너<0xEB><0x9C><0x95>니다.")

    # --- 도구에 위임된 Fetching 로직 (수정된 변수명 사용) ---

    async def _fetch_tweets(self, keyword: str, max_results: int, trace_id: str, comic_id: str) -> List[Dict[str, str]]:
        """트위터 도구에 트윗 Fetching 위임"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'keyword': keyword}
        if not self.twitter_tool: return [] # 수정된 변수명 사용
        try:
            logger.debug(f"'{keyword}' 트위터 검색 위임", extra=extra_log_data)
            # Tool 반환 형식: List[{"url": str, "text": str, "tweet_id": str, "source": "Twitter", ...}]
            results = await self.twitter_tool.search_recent_tweets(keyword=keyword, max_results=max_results, trace_id=trace_id) # 수정된 변수명 사용
            for r in results: r.setdefault("search_keyword", keyword)
            logger.debug(f"TwitterTool로부터 '{keyword}' 결과 {len(results)}개 수신", extra=extra_log_data)
            return results
        except Exception as e:
            logger.error(f"'{keyword}' TwitterTool 호출 오류: {e}", exc_info=True, extra=extra_log_data)
            return []

    async def _fetch_reddit_posts(self, keyword: str, max_results: int, trace_id: str, comic_id: str) -> List[Dict[str, str]]:
        """레딧 도구에 게시글 Fetching 위임"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'keyword': keyword}
        if not self.reddit_tool: return [] # 수정된 변수명 사용
        try:
            logger.debug(f"'{keyword}' 레딧 검색 위임", extra=extra_log_data)
            # Tool 반환 형식: List[{"url": str, "title": str, "text": str, "source": "Reddit (r/...) ...}]
            results = await self.reddit_tool.search_posts(keyword=keyword, max_results=max_results, trace_id=trace_id) # 수정된 변수명 사용
            for r in results: r.setdefault("search_keyword", keyword)
            logger.debug(f"RedditTool로부터 '{keyword}' 결과 {len(results)}개 수신", extra=extra_log_data)
            return results
        except Exception as e:
            logger.error(f"'{keyword}' RedditTool 호출 오류: {e}", exc_info=True, extra=extra_log_data)
            return []

    async def _fetch_youtube_videos(self, keyword: str, max_results: int, trace_id: str, comic_id: str) -> List[Dict[str, str]]:
        """구글 검색 도구에 유튜브 비디오 Fetching 위임"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'keyword': keyword}
        if not self.google_tool: return [] # 수정된 변수명 사용
        try:
            logger.debug(f"'{keyword}' 유튜브 검색 위임", extra=extra_log_data)
            # Tool 반환 형식: List[{"url": str, "title": str, "description": str, "source": "YouTube", ...}]
            results = await self.google_tool.search_youtube_videos(keyword=keyword, max_results=max_results, trace_id=trace_id) # 수정된 변수명 사용
            for r in results: r.setdefault("search_keyword", keyword)
            logger.debug(f"GoogleSearchTool(YouTube)로부터 '{keyword}' 결과 {len(results)}개 수신", extra=extra_log_data)
            return results
        except Exception as e:
            logger.error(f"'{keyword}' GoogleSearchTool(YouTube) 호출 오류: {e}", exc_info=True, extra=extra_log_data)
            return []

    async def _fetch_blog_posts(self, keyword: str, max_results: int, trace_id: str, comic_id: str) -> List[Dict[str, str]]:
        """구글 검색 도구에 블로그 게시글 Fetching 위임 및 필터링"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'keyword': keyword}
        if not self.google_tool: return [] # 수정된 변수명 사용
        try:
            logger.debug(f"'{keyword}' 블로그 검색 위임", extra=extra_log_data)
            # Tool 반환 형식: List[{"url": str, "title": str, "snippet": str, "source": "GoogleCSE_BlogSearch"}]
            results = await self.google_tool.search_blogs_via_cse(keyword=keyword, max_results=max_results, trace_id=trace_id) # 수정된 변수명 사용
            filtered_results = []
            for r in results:
                 url = r.get("url")
                 if url and self._is_blog_url(url):
                     r["source"] = "Blog"
                     r["search_keyword"] = keyword
                     filtered_results.append(r)
            logger.debug(f"GoogleSearchTool(Blog)로부터 '{keyword}' 결과 필터링 후 {len(filtered_results)}개", extra=extra_log_data)
            return filtered_results
        except Exception as e:
            logger.error(f"'{keyword}' GoogleSearchTool(Blog) 호출 오류: {e}", exc_info=True, extra=extra_log_data)
            return []

    async def _fetch_community_posts(self, keyword: str, max_results: int, trace_id: str, comic_id: str) -> List[Dict[str, str]]:
        """구글 검색 도구에 커뮤니티 게시글 Fetching 위임"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'keyword': keyword}
        if not self.google_tool: return [] # 수정된 변수명 사용
        try:
            logger.debug(f"'{keyword}' 커뮤니티 검색 위임", extra=extra_log_data)
            # Tool 반환 형식: List[{"url": str, "title": str, "snippet": str, "source": "GoogleCSE_CommunitySearch"}]
            results = await self.google_tool.search_communities_via_cse(keyword=keyword, max_results=max_results, trace_id=trace_id) # 수정된 변수명 사용
            for r in results:
                 r["source"] = "Community"
                 r["search_keyword"] = keyword
            logger.debug(f"GoogleSearchTool(Community)로부터 '{keyword}' 결과 {len(results)}개 수신", extra=extra_log_data)
            return results
        except Exception as e:
            logger.error(f"'{keyword}' GoogleSearchTool(Community) 호출 오류: {e}", exc_info=True, extra=extra_log_data)
            return []

    # --- 헬퍼 함수 (중복제거, 밸런싱, 블로그 확인) ---
    # (이 부분의 코드는 변경 없음)

    def _is_blog_url(self, url: str) -> bool:
        """URL이 블로그 게시글을 가리키는지 확인합니다."""
        try:
            parsed = urlparse(url); domain = parsed.netloc.lower(); path = parsed.path.lower()
            blog_platforms = [ "blogspot.com", "wordpress.com", "medium.com", "tumblr.com",
                               "livejournal.com", "typepad.com", "substack.com", "ghost.io",
                               "tistory.com", "blog.naver.com", "brunch.co.kr", "velog.io" ]
            if any(domain == platform or domain.endswith('.' + platform) for platform in blog_platforms): return True
            if '/blog/' in path or '/post/' in path or '/entry/' in path:
                 if path.count('/') > 1 or len(path.split('/')[-1]) > 0 : return True
            if re.search(r'/\d{4}/\d{2}(/\d{2})?/', path): return True
            return False
        except Exception: return False

    def _balance_and_sample_results(
        self, platform_results: Dict[str, List[Dict[str, str]]],
        max_total: int, max_per_platform: Dict[str, int], min_per_platform: Dict[str, int],
        trace_id: str, comic_id: str
        ) -> List[Dict[str, str]]:
        """설정된 제한에 따라 플랫폼 간 결과를 밸런싱하고 샘플링합니다."""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        balanced_results = []; final_urls = set()

        # Pass 1: 플랫폼별 최소 결과 수 보장
        logger.debug("밸런싱 Pass 1 시작 (플랫폼별 최소)", extra=extra_log_data)
        for platform, results in platform_results.items():
            if not results: continue
            min_needed = min_per_platform.get(platform, 0)
            current_platform_results = results[:]; random.shuffle(current_platform_results)
            added_count = 0
            for item in current_platform_results:
                 if len(balanced_results) >= max_total: break
                 if added_count >= min_needed: break
                 url = item.get("url")
                 if url and url not in final_urls:
                      balanced_results.append(item); final_urls.add(url); added_count += 1
            platform_results[platform] = [item for item in results if item.get("url") not in final_urls]
            logger.debug(f"Pass 1 ({platform}): {added_count}개 추가 (최소: {min_needed}). 남은 결과: {len(platform_results[platform])}", extra=extra_log_data)

        # Pass 2: 플랫폼별 최대 및 전체 최대 결과 수까지 채움
        logger.debug("밸런싱 Pass 2 시작 (최대까지 채우기)", extra=extra_log_data)
        all_remaining_items = []; platform_order = list(platform_results.keys()); random.shuffle(platform_order)
        platform_current_counts = defaultdict(int)
        for item in balanced_results:
             source_platform = item.get("source", "Unknown").split(" ")[0]
             platform_current_counts[source_platform] += 1

        for platform in platform_order:
            results = platform_results.get(platform, [])
            if not results: continue
            max_allowed = max_per_platform.get(platform, 10)
            current_count = platform_current_counts.get(platform, 0)
            can_add_count = max(0, max_allowed - current_count)
            if can_add_count > 0:
                items_to_consider = results[:can_add_count]
                all_remaining_items.extend(items_to_consider)
                logger.debug(f"Pass 2 ({platform}): 최대={max_allowed}, 현재={current_count}, 추가 가능={can_add_count}. 고려 대상 {len(items_to_consider)}개.", extra=extra_log_data)

        random.shuffle(all_remaining_items); added_pass_2 = 0
        for item in all_remaining_items:
            if len(balanced_results) >= max_total: break
            url = item.get("url")
            if url and url not in final_urls:
                balanced_results.append(item); final_urls.add(url); added_pass_2 += 1

        logger.info(f"밸런싱 완료. Pass 2에서 {added_pass_2}개 추가. 최종 샘플링된 URL 개수: {len(balanced_results)}", extra=extra_log_data)
        return balanced_results

    def _remove_duplicates(self, url_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """사전 목록에서 URL 기준으로 중복을 제거합니다."""
        seen_urls: Set[str] = set(); unique_list = []
        for item in url_list:
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_list.append(item)
        return unique_list

    # --- 메인 실행 메서드 (수정된 변수명 사용) ---
    async def execute(self, state: ComicState) -> Dict[str, Any]:
        """주입된 도구를 사용하여 의견 수집 프로세스를 실행합니다."""
        start_time = datetime.now(timezone.utc)
        comic_id = state.comic_id
        trace_id = state.trace_id or comic_id or "unknown_trace"
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}

        logger.info("OpinionCollectorNode 실행 시작...", extra=extra_log_data)

        search_keywords = state.search_keywords
        if not search_keywords:
            logger.warning("상태에 검색 키워드가 없습니다. 의견 수집을 건너<0xEB><0x9C><0x95>니다.", extra=extra_log_data)
            return {"opinion_urls": [], "error_message": "검색 키워드가 누락되었습니다."}

        config = state.config
        current_used_links = state.used_links
        processing_stats = state.processing_stats

        # --- state.config 또는 settings 기본값에서 설정 읽기 ---
        yt_max = config.get('YOUTUBE_OPINION_MAX_RESULTS', settings.YOUTUBE_OPINION_MAX_RESULTS)
        tw_max = config.get('TWITTER_OPINION_MAX_RESULTS', settings.TWITTER_OPINION_MAX_RESULTS)
        rd_max = config.get('REDDIT_OPINION_MAX_RESULTS', settings.REDDIT_OPINION_MAX_RESULTS)
        bl_max = config.get('BLOG_OPINION_MAX_RESULTS', settings.BLOG_OPINION_MAX_RESULTS)
        cm_max = config.get('COMMUNITY_OPINION_MAX_RESULTS', settings.COMMUNITY_OPINION_MAX_RESULTS)
        max_total = config.get('MAX_TOTAL_OPINIONS_TARGET', settings.MAX_TOTAL_OPINIONS_TARGET)
        max_plat = config.get('MAX_OPINIONS_PER_PLATFORM_SAMPLING', settings.MAX_OPINIONS_PER_PLATFORM_SAMPLING)
        min_plat = config.get('MIN_OPINIONS_PER_PLATFORM_SAMPLING', settings.MIN_OPINIONS_PER_PLATFORM_SAMPLING)
        concurrency = config.get('OPINION_COLLECTOR_CONCURRENCY', settings.OPINION_COLLECTOR_CONCURRENCY)
        semaphore = asyncio.Semaphore(concurrency)

        # --- 태스크 정의 및 실행 ---
        tasks = []

        async def throttled_task(coro):
             async with semaphore: return await coro

        # 각 키워드에 대해 플랫폼별 검색 태스크 생성 (수정된 변수명 사용)
        for keyword in search_keywords:
             kw_extra_log = {**extra_log_data, 'keyword': keyword}
             if self.twitter_tool: # 수정된 변수명 사용
                  tasks.append(throttled_task(self._fetch_tweets(keyword, tw_max, trace_id, comic_id)))
             if self.reddit_tool: # 수정된 변수명 사용
                  tasks.append(throttled_task(self._fetch_reddit_posts(keyword, rd_max, trace_id, comic_id)))
             if self.google_tool: # 수정된 변수명 사용
                  tasks.append(throttled_task(self._fetch_youtube_videos(keyword, yt_max, trace_id, comic_id)))
                  tasks.append(throttled_task(self._fetch_blog_posts(keyword, bl_max, trace_id, comic_id)))
                  tasks.append(throttled_task(self._fetch_community_posts(keyword, cm_max, trace_id, comic_id)))

        # 실행할 태스크가 없으면 오류 처리 및 종료
        if not tasks:
             logger.warning("의견 수집에 사용할 수 있는 도구가 없거나 설정되지 않았습니다.", extra=extra_log_data)
             return {"opinion_urls": [], "error_message": "의견 수집 도구를 사용할 수 없습니다."}

        # 태스크 실행 및 결과 취합
        logger.info(f"{len(tasks)}개의 의견 수집 태스크 실행 (동시성 제한: {concurrency})...", extra=extra_log_data)
        api_results = await asyncio.gather(*tasks, return_exceptions=True)

        # --- 결과 처리 및 취합 ---
        # (이 부분의 코드는 변경 없음)
        platform_results: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        total_collected_count = 0; failed_tasks = 0
        platforms = ["Twitter", "Reddit", "YouTube", "Blog", "Community"]

        for result in api_results:
            if isinstance(result, Exception):
                logger.error(f"의견 수집 태스크 실패: {result}", exc_info=result, extra=extra_log_data)
                failed_tasks += 1
            elif isinstance(result, list) and result:
                source = result[0].get("source", "Unknown")
                platform_name = source.split(" ")[0]
                if platform_name in platforms:
                    platform_results[platform_name].extend(result); total_collected_count += len(result)
                else: logger.warning(f"결과에서 알 수 없는 출처 발견: {source}", extra=extra_log_data)
            elif isinstance(result, list) and not result:
                logger.debug("의견 수집 태스크가 결과를 반환하지 않았습니다.", extra=extra_log_data)
            else: logger.warning(f"태스크에서 예상치 못한 결과 타입 수신: {type(result)}", extra=extra_log_data)

        logger.info(f"초기 수집 완료. 총 원시 URL: {total_collected_count}. 실패 태스크: {failed_tasks}.", extra=extra_log_data)
        for platform, results in platform_results.items():
             logger.info(f"- {platform}: {len(results)}개 URL 수집됨 (플랫폼 내 중복 제거 전).", extra=extra_log_data)

        # 플랫폼 내 중복 우선 제거
        for platform in platform_results:
             initial_count = len(platform_results[platform])
             platform_results[platform] = self._remove_duplicates(platform_results[platform])
             logger.debug(f"{platform} 중복 제거: {initial_count} -> {len(platform_results[platform])}", extra=extra_log_data)

        # 플랫폼 간 밸런싱 및 샘플링
        balanced_sampled_urls = self._balance_and_sample_results(
            platform_results, max_total=max_total, max_per_platform=max_plat,
            min_per_platform=min_plat, trace_id=trace_id, comic_id=comic_id
        )
        logger.info(f"밸런싱 및 샘플링 후 최종 고유 의견 URL 개수: {len(balanced_sampled_urls)}", extra=extra_log_data)

        # --- 사용 링크 추적 업데이트 ---
        # (이 부분의 코드는 변경 없음)
        updated_used_links = list(current_used_links); added_link_count = 0
        existing_link_urls = {link.get('url') for link in updated_used_links if link.get('url')}
        for item in balanced_sampled_urls:
            url = item.get('url')
            if url and url not in existing_link_urls:
                link_info = { 'url': url,
                    'purpose': f"Collected from {item.get('source', 'Unknown')} for keyword '{item.get('search_keyword', 'Unknown')}' (Opinion)",
                    'node': 'OpinionCollectorNode', 'timestamp': datetime.now(timezone.utc).isoformat() }
                updated_used_links.append(link_info); existing_link_urls.add(url); added_link_count += 1
        logger.info(f"{added_link_count}개의 새로운 의견 URL을 used_links에 추가했습니다.", extra=extra_log_data)

        # --- 시간 기록 및 반환 ---
        # (이 부분의 코드는 변경 없음)
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['node_04_opinion_collector_time'] = node_processing_time
        logger.info(f"OpinionCollectorNode 완료. 소요 시간: {node_processing_time:.2f} 초.", extra=extra_log_data)

        # 상태 업데이트 준비
        updates = { "opinion_urls": balanced_sampled_urls,
                    "used_links": updated_used_links,
                    "processing_stats": processing_stats }
        current_error = state.error_message
        if current_error and "OpinionCollectorNode" in current_error:
             updates["error_message"] = None

        return updates