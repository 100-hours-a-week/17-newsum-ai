# ai/app/nodes_v2/n04_execute_search_node.py
import asyncio
from typing import Dict, Any, List, Optional, Coroutine
from datetime import datetime, timezone
from urllib.parse import urlparse
import re

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.services.postgresql_service import PostgreSQLService

logger = get_logger(__name__)

NODE_ORDER = 4
MIN_DOMAINS_TO_FETCH_N04 = 30  # N04용 DB 조회 최소 개수

# 도구별 폴백 카테고리
DEFAULT_FALLBACK_CATEGORIES = {
    "search_communities_via_cse": ["community", "forum"],
    "search_news_via_cse": ["mainstream_news", "news_agency", "broadcast_news"],
    "search_specific_sites_via_cse": []  # SiteSearch는 카테고리 없음
}

# 한국 도메인 판단용 대표 키워드
CORE_KOREAN_DOMAIN_NAMES = [
    "yna", "kbs", "sbs", "chosun", "joongang", "donga", "hankyoreh",
    "mk", "hankyung", "naver", "daum", "nate", "zum"
]


class N04ExecuteSearchNode:
    def __init__(
        self,
        search_tool: GoogleSearchTool,
        postgresql_service: Optional[PostgreSQLService] = None
    ):
        self.search_tool = search_tool
        self.postgresql_service = postgresql_service
        self.youtube_tool_identifier = "search_youtube_videos"

    async def _fetch_domains_from_db_generic_n04(
        self,
        work_id_log: str,
        country_code: Optional[str] = None,
        categories: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        limit: int = MIN_DOMAINS_TO_FETCH_N04
    ) -> List[str]:
        """
        ai_test_seed_domains 테이블에서 조건에 맞는 도메인을 조회 (N04용).
        DB 접속 실패 시 빈 리스트 반환.
        """
        if not self.postgresql_service:
            logger.error(f"[N04 DomainFetch] PostgreSQLService not injected. Cannot fetch domains.",
                         extra={"work_id": work_id_log})
            return []

        query_parts: List[str] = []
        params: List[Any] = []

        if country_code:
            query_parts.append("country = %s")
            params.append(country_code.upper())

        if categories:
            valid_categories = [
                cat for cat in categories
                if cat and cat.strip() and cat.lower() not in ['other', 'unknown', 'ambiguous']
            ]
            if valid_categories:
                query_parts.append("category = ANY(%s)")
                params.append(valid_categories)

        if keywords:
            processed_keywords = [kw.lower() for kw in keywords if kw]
            if processed_keywords:
                query_parts.append("tags && %s")
                params.append(processed_keywords)

        sql_where_clause = " AND ".join(query_parts) if query_parts else "TRUE"
        sql_query = f"""
            SELECT domain
            FROM ai_test_seed_domains
            WHERE {sql_where_clause}
            ORDER BY relevance_score DESC NULLS LAST, last_crawled_at DESC NULLS LAST
            LIMIT %s;
        """
        current_params = tuple(params + [limit])

        try:
            logger.debug(
                f"[N04 DomainFetch] Executing generic domain DB query: {sql_query} | Params: {current_params}",
                extra={"work_id": work_id_log}
            )
            results = await self.postgresql_service.fetch_all(sql_query, current_params)
            domains = [row[0] for row in results if row and row[0]]
            if domains:
                logger.info(f"[N04 DomainFetch] Fetched {len(domains)} domains: {domains}",
                            extra={"work_id": work_id_log})
            else:
                logger.info(f"[N04 DomainFetch] No domains found for criteria. Returning empty list.",
                            extra={"work_id": work_id_log})
            return domains
        except Exception as e:
            logger.error(
                f"[N04 DomainFetch] Error fetching domains: {e}. Returning empty list.",
                extra={"work_id": work_id_log, "sql_query": sql_query}
            )
            return []

    async def _determine_fallback_target_sites(
        self,
        writer_concept: Dict[str, Any],
        config_from_state: Dict[str, Any],
        tool_name: str,
        query_text: str,
        work_id_log: str
    ) -> Optional[List[str]]:
        """
        도구별 DB 기반 폴백 사이트 목록 결정.
        """
        logger.debug(
            f"[N04 Fallback] Tool: {tool_name}, Query: '{summarize_for_logging(query_text, 30)}'",
            extra={"work_id": work_id_log}
        )
        if not self.postgresql_service:
            logger.warning(
                f"[N04 Fallback] PostgreSQLService not available.",
                extra={"work_id": work_id_log}
            )
            return None

        fallback_categories = DEFAULT_FALLBACK_CATEGORIES.get(tool_name)
        if fallback_categories is not None or tool_name == "search_specific_sites_via_cse":
            is_kr_query = self._is_korean_query(query_text)
            country_code_filter = 'KR' if is_kr_query else None
            categories_to_use = fallback_categories if fallback_categories else None

            logger.info(
                f"[N04 Fallback] Fetching DB fallback sites for '{tool_name}', "
                f"categories={categories_to_use}, country={country_code_filter}",
                extra={"work_id": work_id_log}
            )

            db_sites = await self._fetch_domains_from_db_generic_n04(
                work_id_log=work_id_log,
                country_code=country_code_filter,
                categories=categories_to_use,
                keywords=None,
                limit=MIN_DOMAINS_TO_FETCH_N04
            )
            if db_sites:
                return db_sites
            else:
                logger.info(
                    f"[N04 Fallback] No fallback sites found for '{tool_name}'.",
                    extra={"work_id": work_id_log}
                )
        return None

    def _prepare_tone_modified_query(self, base_query: str, tool_name: str, tone: str) -> str:
        """
        풍자 톤(tone_suggestion)에 따라 키워드에 간단한 수식어를 덧붙인다.
        """
        tone_lower = tone.lower() if isinstance(tone, str) else ""
        modified_query = base_query

        # 유머/반어적 풍자(irony, humor) -> 블로그나 웹 검색 시 "유머" 추가
        if tool_name in ["search_web_via_cse", "search_blogs_via_cse"]:
            if "irony" in tone_lower or "humor" in tone_lower:
                modified_query = f"{base_query} 유머"
        # 냉소적 풍자(sarcasm, satire) -> 커뮤니티 검색 시 "비판" 추가
        if tool_name in ["search_communities_via_cse", "search_specific_sites_via_cse"]:
            if "sarcasm" in tone_lower or "satire" in tone_lower:
                modified_query = f"{base_query} 비판"

        return modified_query

    async def _create_search_task_wrapper(
        self,
        coro_to_await: Coroutine[Any, Any, List[Dict[str, Any]]],
        tool_name_val: str,
        query_text_val: str,
        log_ctx_val: dict
    ) -> List[Dict[str, Any]]:
        """
        실제 검색 코루틴 실행 후, 메타 정보를 각 결과에 붙여 반환.
        """
        items: List[Dict[str, Any]] = []
        try:
            results_list = await coro_to_await
            if results_list:
                for res_item in results_list:
                    if isinstance(res_item, dict):
                        res_item['query_source'] = query_text_val
                        res_item['tool_used'] = tool_name_val
                        res_item['retrieved_at'] = datetime.now(timezone.utc).isoformat()
                        res_item.setdefault('source_domain',
                                            urlparse(res_item.get("url", "")).netloc or 'N/A')
                        items.append(res_item)
                logger.debug(
                    f"[N04 Result] Tool '{tool_name_val}', Query '{summarize_for_logging(query_text_val, 30)}' -> {len(items)} items",
                    extra=log_ctx_val
                )
        except Exception as e_search:
            logger.error(
                f"[N04 Error] Search task error (Query: '{query_text_val}', Tool: {tool_name_val}): {e_search}",
                extra=log_ctx_val, exc_info=True
            )
        return items

    async def _run_Youtube_and_transcripts(
        self,
        query_text: str,
        max_videos: int,
        search_params: Dict[str, Any],
        trace_id: str,
        config_from_state: Dict[str, Any],
        log_ctx: dict
    ) -> List[Dict[str, Any]]:
        """
        YouTube 검색 후, 해당 비디오들의 자막을 병렬로 가져와 결과에 추가.
        """
        video_items_with_transcripts: List[Dict[str, Any]] = []
        try:
            logger.info(f"[N04 YouTube] Searching videos: '{query_text}' (max {max_videos})", extra=log_ctx)
            videos = await self.search_tool.search_youtube_videos(
                keyword=query_text, max_results=max_videos, trace_id=trace_id, **search_params
            )

            if not videos:
                logger.info(f"[N04 YouTube] No videos found for '{query_text}'", extra=log_ctx)
                return []

            logger.info(f"[N04 YouTube] Found {len(videos)} videos for '{query_text}'. Fetching transcripts...", extra=log_ctx)

            transcript_tasks = []
            for video_info in videos:
                video_id = video_info.get("video_id")
                if video_id:
                    preferred_languages = config_from_state.get("youtube_transcript_languages", ['ko', 'en'])
                    translate_to_lang = config_from_state.get("youtube_transcript_translate_to", 'ko')
                    transcript_tasks.append(
                        self.search_tool.get_youtube_transcript(
                            video_id,
                            languages=preferred_languages,
                            translate_to_language=translate_to_lang,
                            trace_id=trace_id
                        )
                    )
                else:
                    async def dummy_none():
                        return None
                    transcript_tasks.append(dummy_none())

            transcript_results = await asyncio.gather(*transcript_tasks, return_exceptions=True)

            for idx, video_data in enumerate(videos):
                full_item = {**video_data}
                full_item['query_source'] = query_text
                full_item['tool_used'] = self.youtube_tool_identifier
                full_item['retrieved_at'] = datetime.now(timezone.utc).isoformat()
                full_item.setdefault('source_domain', urlparse(video_data.get("url", "")).netloc or 'youtube.com')

                outcome = transcript_results[idx]
                if isinstance(outcome, dict) and outcome.get("text"):
                    full_item['transcript'] = outcome["text"]
                    full_item['transcript_language'] = outcome["language"]
                    logger.debug(
                        f"[N04 YouTube] Transcript added for video_id '{video_data.get('video_id')}', lang: {outcome['language']}",
                        extra=log_ctx
                    )
                elif isinstance(outcome, Exception):
                    logger.warning(
                        f"[N04 YouTube] Transcript fetch error for video_id '{video_data.get('video_id')}': {outcome}",
                        extra=log_ctx
                    )
                    full_item['transcript_error'] = str(outcome)
                else:
                    full_item['transcript'] = None
                    logger.debug(
                        f"[N04 YouTube] No transcript available for video_id '{video_data.get('video_id')}'",
                        extra=log_ctx
                    )

                video_items_with_transcripts.append(full_item)

        except Exception as e_youtube:
            logger.error(
                f"[N04 YouTube] Main error during search/transcript for '{query_text}': {e_youtube}",
                extra=log_ctx, exc_info=True
            )
        return video_items_with_transcripts

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        search_sec = state.search
        config_from_state = state.config.config or {}

        node_name = self.__class__.__name__
        work_id = meta_sec.work_id
        writer_id_from_config = config_from_state.get('writer_id', 'default_writer')
        extra_log_base = {
            'work_id': work_id,
            'writer_id': writer_id_from_config,
            'node_name': node_name,
            'node_order': NODE_ORDER
        }

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info("N04 노드 진입: 검색 전략 기반 실제 검색 실행 시작.", extra=extra_log_base)

        accumulated_raw_results: List[Dict[str, Any]] = []
        search_strategy = search_sec.search_strategy or {}

        if not isinstance(search_strategy, dict) or not search_strategy:
            logger.warning("검색 전략이 정의되지 않아 검색을 수행할 수 없습니다.", extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"
            search_sec.raw_search_results = []
            return {"search": search_sec.model_dump(), "meta": meta_sec.model_dump()}

        general_queries: List[str] = search_strategy.get("queries", [])
        target_seed_domains: Optional[List[str]] = search_strategy.get("target_seed_domains")
        selected_tools_from_strategy: List[str] = search_strategy.get(
            "selected_tools", ["search_web_via_cse"]
        )
        writer_concept: Dict[str, Any] = search_strategy.get("writer_concept", {})
        tool_parameters: Dict[str, Dict[str, Any]] = search_strategy.get("tool_parameters", {})
        # N03에서 query_context에 저장된 tone_suggestion
        tone_suggestion: str = state.query.query_context.get("tone_suggestion", "")

        logger.info(
            f"[N04 Strategy] Queries: {general_queries}, Seed domains: {len(target_seed_domains) if target_seed_domains else 0}, Tools: {selected_tools_from_strategy}",
            extra=extra_log_base
        )
        if target_seed_domains:
            logger.debug(f"[N04 Strategy] Seed domains sample: {target_seed_domains[:5]}", extra=extra_log_base)

        all_search_coroutines: List[Coroutine[Any, Any, List[Dict[str, Any]]]] = []

        # --- 1. Seed domain 기반 검색 ---
        if target_seed_domains and general_queries:
            logger.info("[N04] Seed domain 기반 검색 작업 생성 중...", extra=extra_log_base)
            for q_text_seed in general_queries:
                if not q_text_seed.strip():
                    continue
                log_ctx_seed = {
                    **extra_log_base,
                    "query": summarize_for_logging(q_text_seed, 50),
                    "search_mode": "seed_domains_primary"
                }
                # 파라미터 가져오기
                params_seed = tool_parameters.get("search_specific_sites_via_cse", {}).copy()
                adv_opts_seed = {}  # N03에서 미리 준비된 옵션 적용
                params_seed.update(adv_opts_seed)
                max_res_seed = params_seed.pop("max_results", 2)
                modified_query = self._prepare_tone_modified_query(q_text_seed, "search_specific_sites_via_cse", tone_suggestion)

                coro = self.search_tool.search_specific_sites_via_cse(
                    keyword=modified_query,
                    sites=target_seed_domains,
                    max_results=max_res_seed,
                    trace_id=work_id,
                    **params_seed
                )
                all_search_coroutines.append(
                    self._create_search_task_wrapper(
                        coro, "search_specific_sites_via_cse", modified_query, log_ctx_seed
                    )
                )

        # --- 2. 일반 검색 작업 ---
        if general_queries:
            logger.info("[N04] 일반 검색 작업 생성 중...", extra=extra_log_base)
            for q_text_general in general_queries:
                if not q_text_general.strip():
                    continue
                log_ctx_general_query = {
                    **extra_log_base,
                    "query": summarize_for_logging(q_text_general, 50),
                    "search_mode": "general_search_selected_tools"
                }
                for tool_name in selected_tools_from_strategy:
                    log_ctx_tool = {**log_ctx_general_query, "search_tool": tool_name}
                    params_general = tool_parameters.get(tool_name, {}).copy()
                    adv_opts_general = {}  # 이미 N03에서 적용된 고급 옵션
                    params_general.update(adv_opts_general)
                    max_res_general = params_general.pop("max_results", 3)

                    # 풍자 톤 반영해서 쿼리 수정
                    modified_query = self._prepare_tone_modified_query(q_text_general, tool_name, tone_suggestion)

                    coroutine_for_tool: Optional[Coroutine[Any, Any, List[Dict[str, Any]]]] = None

                    # YouTube
                    if tool_name == self.youtube_tool_identifier:
                        coroutine_for_tool = self._run_Youtube_and_transcripts(
                            modified_query, max_res_general, params_general, work_id, config_from_state, log_ctx_tool
                        )

                    # 특정 사이트 검색
                    elif tool_name == "search_specific_sites_via_cse":
                        if target_seed_domains:
                            coroutine_for_tool = self.search_tool.search_specific_sites_via_cse(
                                keyword=modified_query,
                                sites=target_seed_domains,
                                max_results=max_res_general,
                                trace_id=work_id,
                                **params_general
                            )
                        else:
                            logger.info(f"[N04] No seed domains for '{tool_name}'. Attempting fallback...", extra=log_ctx_tool)
                            fallback_sites = await self._determine_fallback_target_sites(
                                writer_concept, config_from_state, tool_name, modified_query, work_id
                            )
                            if fallback_sites:
                                coroutine_for_tool = self.search_tool.search_specific_sites_via_cse(
                                    keyword=modified_query,
                                    sites=fallback_sites,
                                    max_results=max_res_general,
                                    trace_id=work_id,
                                    **params_general
                                )

                    # 일반 웹 검색
                    elif tool_name == "search_web_via_cse":
                        coroutine_for_tool = self.search_tool.search_web_via_cse(
                            keyword=modified_query,
                            max_results=max_res_general,
                            trace_id=work_id,
                            **params_general
                        )

                    # 뉴스 검색
                    elif tool_name == "search_news_via_cse":
                        coroutine_for_tool = self.search_tool.search_news_via_cse(
                            keyword=modified_query,
                            max_results=max_res_general,
                            trace_id=work_id,
                            **params_general
                        )

                    # 커뮤니티 검색
                    elif tool_name == "search_communities_via_cse":
                        fallback_sites = await self._determine_fallback_target_sites(
                            writer_concept, config_from_state, tool_name, modified_query, work_id
                        )
                        if fallback_sites:
                            coroutine_for_tool = self.search_tool.search_specific_sites_via_cse(
                                keyword=modified_query,
                                sites=fallback_sites,
                                max_results=max_res_general,
                                trace_id=work_id,
                                **params_general
                            )
                        else:
                            if hasattr(self.search_tool, 'search_communities_via_cse'):
                                coroutine_for_tool = self.search_tool.search_communities_via_cse(
                                    keyword=modified_query,
                                    max_results=max_res_general,
                                    trace_id=work_id,
                                    **params_general
                                )
                            else:
                                logger.warning(
                                    f"[N04] Method search_communities_via_cse missing. Falling back to web search.",
                                    extra=log_ctx_tool
                                )
                                community_keywords = " OR ".join([
                                    f"site:{s}" for s in
                                    ["dcinside.com", "fmkorea.com", "ruliweb.com",
                                     "clien.net", "todayhumor.co.kr"]
                                ])
                                coroutine_for_tool = self.search_tool.search_web_via_cse(
                                    keyword=f"{modified_query} ({community_keywords})",
                                    max_results=max_res_general,
                                    trace_id=work_id,
                                    **params_general
                                )

                    # 블로그/리뷰 검색
                    elif tool_name == "search_blogs_via_cse":
                        coroutine_for_tool = self.search_tool.search_blogs_via_cse(
                            keyword=modified_query,
                            max_results=max_res_general,
                            trace_id=work_id,
                            **params_general
                        )

                    if coroutine_for_tool:
                        all_search_coroutines.append(
                            self._create_search_task_wrapper(
                                coroutine_for_tool, tool_name, modified_query, log_ctx_tool
                            )
                        )

        # --- 3. 모든 검색 작업 병렬 실행 및 결과 수집 ---
        if all_search_coroutines:
            logger.info(f"[N04] Running {len(all_search_coroutines)} search tasks in parallel.", extra=extra_log_base)
            gathered = await asyncio.gather(*all_search_coroutines, return_exceptions=True)
            for result_or_exc in gathered:
                if isinstance(result_or_exc, list):
                    accumulated_raw_results.extend(
                        item for item in result_or_exc if isinstance(item, dict)
                    )
                elif isinstance(result_or_exc, Exception):
                    logger.error(f"[N04] Exception during parallel search: {result_or_exc}", extra=extra_log_base, exc_info=True)
        else:
            logger.info("[N04] No search tasks to execute.", extra=extra_log_base)

        # --- 4. 한국 결과 우선순위 적용 및 중복 제거 ---
        unique_final_results = self._prioritize_and_deduplicate_korean_first(
            accumulated_raw_results, work_id
        )

        logger.info(
            f"[N04] Search completed. Collected: {len(accumulated_raw_results)}, Unique: {len(unique_final_results)}.",
            extra=extra_log_base
        )

        search_sec.raw_search_results = unique_final_results
        meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"
        logger.info("[N04] 검색 실행 완료. 다음 단계로 이동합니다.", extra=extra_log_base)

        return {"search": search_sec.model_dump(), "meta": meta_sec.model_dump()}

    def _is_korean_query(self, query: str) -> bool:
        """
        쿼리 내 한글 비율 30% 이상이면 한국어 쿼리로 판단.
        """
        korean_chars = len(re.findall(r'[가-힣]', query))
        total_chars = len(query.replace(' ', ''))
        return (korean_chars / max(total_chars, 1)) > 0.3

    def _prioritize_and_deduplicate_korean_first(
        self,
        results: List[Dict[str, Any]],
        work_id_log: str
    ) -> List[Dict[str, Any]]:
        """
        한국 도메인 결과를 먼저 모으고, 나머지 뒤에 붙인 뒤 URL/Video ID로 중복 제거.
        """
        korean_results = []
        other_results = []

        KNOWN_KOREAN_TLDS = [
            ".kr", ".co.kr", ".ne.kr", ".or.kr", ".re.kr", ".go.kr",
            ".ac.kr", ".pe.kr", ".ms.kr", ".hs.kr", ".es.kr", ".kro.kr"
        ]

        for item in results:
            domain = item.get('source_domain', '').lower()
            title = item.get('title', '')
            snippet = item.get('snippet', '')

            is_korean_priority = False
            if any(tld in domain for tld in KNOWN_KOREAN_TLDS):
                is_korean_priority = True
            elif any(name_part in domain for name_part in CORE_KOREAN_DOMAIN_NAMES):
                is_korean_priority = True
            else:
                if self._is_korean_query(title + " " + snippet):
                    is_korean_priority = True

            if is_korean_priority:
                korean_results.append(item)
            else:
                other_results.append(item)

        prioritized_list = korean_results + other_results

        unique_final_results: List[Dict[str, Any]] = []
        seen_identifiers_for_dedup = set()
        current_rank = 1
        for item_data in prioritized_list:
            item_identifier = None
            if item_data.get('tool_used') == self.youtube_tool_identifier and item_data.get('video_id'):
                item_identifier = f"youtube_{item_data['video_id']}"
            elif item_data.get('url'):
                item_identifier = item_data['url']

            if item_identifier and item_identifier not in seen_identifiers_for_dedup:
                item_data['rank'] = current_rank
                unique_final_results.append(item_data)
                seen_identifiers_for_dedup.add(item_identifier)
                current_rank += 1
            elif not item_identifier:
                item_data['rank'] = current_rank
                unique_final_results.append(item_data)
                current_rank += 1
                logger.warning(
                    f"[N04 Dedup] Item without identifier added (possible duplicate): {summarize_for_logging(item_data, 100)}",
                    extra={'work_id': work_id_log}
                )

        return unique_final_results
