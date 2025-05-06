# ai/app/nodes/n04_execute_search_node.py
import asyncio
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from urllib.parse import urlparse

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.tools.search.Google_Search_tool import GoogleSearchTool

logger = get_logger(__name__)


class N04ExecuteSearchNode:
    """
    N03에서 수립된 검색 전략에 따라 실제 검색을 수행하고 결과를 수집하는 노드.
    Google 일반 검색과 YouTube 비디오/자막 검색을 병렬적으로 수행하여 데이터 보강.
    """

    def __init__(self, search_tool: GoogleSearchTool):
        self.search_tool = search_tool
        self.default_site_preferences = {
            "tech_expert_3": {
                "code_related": ["stackoverflow.com", "github.com"],
                "research_paper": ["arxiv.org", "semanticscholar.org"],
            },
            "community_search_defaults": ["reddit.com"]
        }
        self.query_keyword_to_site_category_map = {
            "code_related": ["code", "python", "javascript", "error", "github", "stack overflow", "programming", "develop"],
            "research_paper": ["paper", "research", "study", "arxiv", "publication", "citation", "preprint"],
            "deep_dive_tech": ["deep dive", "technical analysis", "tutorial", "how it works", "internals"],
            "community": ["review", "opinion", "forum", "discussion", "vs", "recommendation", "advice"],
            "news": ["news", "latest update", "announcement"]
        }
        # YouTube 검색 도구 식별자 정의 (클래스 또는 모듈 레벨)
        self.youtube_tool_identifier = "YouTubeSearch"


    def _get_relevant_user_sites(self, user_preferences: Dict[str, List[str]], query_text: str) -> List[str]:
        relevant_sites = []
        query_lower = query_text.lower()
        for category, keywords in self.query_keyword_to_site_category_map.items():
            if any(kw in query_lower for kw in keywords):
                sites_in_category = user_preferences.get(category)
                if isinstance(sites_in_category, list):
                    relevant_sites.extend(sites_in_category)
        return list(set(relevant_sites))

    def _determine_target_sites(
            self, writer_concept: dict, config: dict, tool_name: str, query_text: str
    ) -> Optional[List[str]]:
        sites = []
        writer_id = config.get("writer_id", "default_writer")

        user_preferences = config.get("user_site_preferences")
        if isinstance(user_preferences, dict):
            logger.debug(f"Using user site preferences: {user_preferences}")
            user_relevant_sites = self._get_relevant_user_sites(user_preferences, query_text)
            if user_relevant_sites:
                logger.info(f"Applying user-defined sites for query '{query_text}': {user_relevant_sites}")
                return user_relevant_sites
            else:
                logger.debug("User site preferences found but no matching category for this query. Falling back.")

        if writer_id in self.default_site_preferences:
            default_prefs = self.default_site_preferences[writer_id]
            query_lower = query_text.lower()
            for category, keywords in self.query_keyword_to_site_category_map.items():
                if category in default_prefs and any(kw in query_lower for kw in keywords):
                    sites.extend(default_prefs[category])

        if tool_name == "GoogleCSE_CommunitySearch" and "site:" not in query_text.lower():
            sites.extend(self.default_site_preferences.get("community_search_defaults", []))

        final_sites = list(set(sites)) if sites else None
        if final_sites:
            logger.debug(f"Applying fallback/default sites for query '{query_text}': {final_sites}")
        return final_sites

    def _prepare_advanced_search_options(
            self, writer_concept: dict, tool_name: str, query_text: str
    ) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        depth = writer_concept.get("depth", "medium")
        trend_sensitivity = writer_concept.get("trend_sensitivity", "medium")

        if tool_name == "GoogleCSE_NewsSearch":
            if trend_sensitivity == "high": options["dateRestrict"] = "m1"
            elif trend_sensitivity == "medium": options["dateRestrict"] = "m3"
        elif trend_sensitivity == "high" and "latest" in query_text.lower():
            options["dateRestrict"] = "m3"

        if depth == "high" and any(kw in query_text.lower() for kw in self.query_keyword_to_site_category_map.get("research_paper", [])):
            options["fileType"] = "pdf"

        if tool_name == self.youtube_tool_identifier: # self.youtube_tool_identifier 사용
            if writer_concept.get("Youtube_order"):
                 options["order"] = writer_concept["Youtube_order"]
            if writer_concept.get("youtube_video_duration"):
                 options["videoDuration"] = writer_concept["youtube_video_duration"]

        if options:
            logger.debug(f"Prepared advanced search options for {tool_name}: {options}")
        return options

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')

        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        logger.info(f"Entering node. Executing searches based on strategy.", extra=extra_log_data)

        raw_search_results_accumulator: List[Dict[str, Any]] = []
        error_log = list(state.error_log or [])

        search_strategy = state.search_strategy
        if not search_strategy or not isinstance(search_strategy, dict) or not search_strategy.get("queries"):
            msg = "Search strategy or queries missing in state. Skipping search execution."
            logger.warning(msg, extra=extra_log_data)
            error_log.append({"stage": node_name, "error": msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            return {
                "raw_search_results": [],
                "current_stage": "n05_report_generation",
                "error_log": error_log
            }

        queries_from_n03: List[str] = search_strategy.get("queries", [])
        selected_tools: List[str] = search_strategy.get("selected_tools", ["GoogleCSE_WebSearch"])
        writer_concept: Dict[str, Any] = search_strategy.get("writer_concept", {})
        base_search_params: Dict[str, Any] = search_strategy.get("parameters", {"max_results_per_query": 5})

        logger.info(f"Search plan: {len(queries_from_n03)} queries using tools: {selected_tools}", extra=extra_log_data)

        for query_text in queries_from_n03:
            query_extra_log = {**extra_log_data, "current_query": query_text}
            if not query_text or not isinstance(query_text, str) or query_text.strip().upper() == "N/A":
                logger.warning(f"Skipping invalid query: '{query_text}'", extra=query_extra_log)
                continue

            search_tasks_for_current_query = []

            for tool_name_iter in selected_tools:
                if tool_name_iter == self.youtube_tool_identifier: # self.youtube_tool_identifier 사용
                    continue

                tool_extra_log_iter = {**query_extra_log, "search_tool": tool_name_iter}
                current_tool_params = base_search_params.copy()
                target_sites_for_tool = self._determine_target_sites(writer_concept, config, tool_name_iter, query_text)
                advanced_options_for_tool = self._prepare_advanced_search_options(writer_concept, tool_name_iter, query_text)
                current_tool_params.update(advanced_options_for_tool)
                max_results_for_tool = current_tool_params.pop("max_results_per_query", 5)
                api_call_coroutine = None

                if target_sites_for_tool:
                    api_call_coroutine = self.search_tool.search_specific_sites_via_cse(
                        keyword=query_text, sites=target_sites_for_tool, max_results=max_results_for_tool,
                        trace_id=trace_id, **current_tool_params
                    )
                elif tool_name_iter == "GoogleCSE_WebSearch":
                    api_call_coroutine = self.search_tool.search_web_via_cse(
                        keyword=query_text, max_results=max_results_for_tool, trace_id=trace_id, **current_tool_params
                    )
                elif tool_name_iter == "GoogleCSE_NewsSearch":
                    api_call_coroutine = self.search_tool.search_news_via_cse(
                        keyword=query_text, max_results=max_results_for_tool, trace_id=trace_id, **current_tool_params
                    )
                elif tool_name_iter == "GoogleCSE_CommunitySearch":
                     api_call_coroutine = self.search_tool.search_communities_via_cse(
                        keyword=query_text, max_results=max_results_for_tool, trace_id=trace_id, **current_tool_params
                    )

                if api_call_coroutine:
                    async def search_wrapper(coroutine_to_await, tool_name_val, query_text_val, log_ctx_val): # Renamed log_ctx
                        task_error_log_wrapper = []
                        try:
                            logger.info(f"Executing {tool_name_val} for query '{query_text_val}'", extra=log_ctx_val)
                            results = await coroutine_to_await
                            processed_items = []
                            if results:
                                for res_item in results:
                                    if isinstance(res_item, dict):
                                        res_item['query_source'] = query_text_val
                                        res_item['tool_used'] = tool_name_val
                                        res_item['retrieved_at'] = datetime.now(timezone.utc).isoformat()
                                        res_item.setdefault('source_domain', urlparse(res_item.get("url", "")).netloc)
                                        processed_items.append(res_item)
                                logger.info(f"Found {len(processed_items)} results using {tool_name_val} for '{query_text_val}'.", extra=log_ctx_val)
                            else:
                                logger.info(f"No results found using {tool_name_val} for '{query_text_val}'.", extra=log_ctx_val)
                            return processed_items # Return results list
                        except Exception as e:
                            error_msg = f"Search execution failed for query '{query_text_val}' using {tool_name_val}: {e}"
                            logger.exception(error_msg, extra=log_ctx_val)
                            task_error_log_wrapper.append({
                                "stage": f"{node_name}.{tool_name_val}", "query": query_text_val, "error": str(e),
                                "detail": traceback.format_exc(), "timestamp": datetime.now(timezone.utc).isoformat()
                            })
                            return task_error_log_wrapper # Return error log list

                    search_tasks_for_current_query.append(search_wrapper(api_call_coroutine, tool_name_iter, query_text, tool_extra_log_iter))

            if self.youtube_tool_identifier in selected_tools: # self.youtube_tool_identifier 사용
                async def Youtube_and_transcript_task_runner(current_query_text, yt_trace_id):
                    yt_task_results = []
                    yt_task_error_log = []
                    yt_tool_name = self.youtube_tool_identifier # self.youtube_tool_identifier 사용
                    yt_log_ctx = {**query_extra_log, "search_tool": yt_tool_name}

                    try:
                        yt_params = base_search_params.copy()
                        yt_advanced = self._prepare_advanced_search_options(writer_concept, yt_tool_name, current_query_text)
                        yt_params.update(yt_advanced)
                        yt_max_videos = yt_params.pop("max_results_per_query", config.get("youtube_max_videos", 3))

                        logger.info(f"Executing YouTube Video Search for query '{current_query_text}' (max: {yt_max_videos})", extra=yt_log_ctx)
                        videos = await self.search_tool.search_youtube_videos(
                            keyword=current_query_text, max_results=yt_max_videos, trace_id=yt_trace_id, **yt_params
                        )

                        if videos:
                            logger.info(f"Found {len(videos)} YouTube videos for '{current_query_text}'. Fetching transcripts...", extra=yt_log_ctx)
                            transcript_fetch_coros = []
                            for video_info_item in videos:
                                video_id_val = video_info_item.get("video_id")
                                if video_id_val:
                                    transcript_langs = config.get("youtube_transcript_languages", ['en', 'ko'])
                                    translate_to_lang = config.get("youtube_transcript_translate_to", 'en')
                                    transcript_fetch_coros.append(
                                        self.search_tool.get_youtube_transcript(
                                            video_id_val, languages=transcript_langs,
                                            translate_to_language=translate_to_lang, trace_id=yt_trace_id
                                        )
                                    )
                                else:
                                    async def _dummy_none_coro(): return None
                                    transcript_fetch_coros.append(_dummy_none_coro())

                            transcripts_or_errors = await asyncio.gather(*transcript_fetch_coros, return_exceptions=True)

                            for idx, video_data_item in enumerate(videos):
                                full_video_item = {**video_data_item}
                                full_video_item['query_source'] = current_query_text
                                full_video_item['tool_used'] = yt_tool_name
                                full_video_item['retrieved_at'] = datetime.now(timezone.utc).isoformat()
                                full_video_item.setdefault('source', 'YouTube')
                                full_video_item.setdefault('source_domain', 'youtube.com') # 일관성을 위해 유지 또는 실제 도메인

                                transcript_content = transcripts_or_errors[idx]
                                if isinstance(transcript_content, dict) and transcript_content.get("text") is not None: # text가 None이 아닌지 명시적 확인
                                    full_video_item['transcript'] = transcript_content["text"]
                                    full_video_item['transcript_language'] = transcript_content["language"]
                                    logger.debug(f"Transcript added for video {video_data_item.get('video_id')}", extra=yt_log_ctx)
                                elif isinstance(transcript_content, Exception):
                                    logger.warning(f"Failed to fetch transcript for video {video_data_item.get('video_id')}: {transcript_content}", extra=yt_log_ctx)
                                    full_video_item['transcript_error'] = str(transcript_content)
                                elif transcript_content is None:
                                    logger.debug(f"No transcript found or fetch returned None for video {video_data_item.get('video_id')}", extra=yt_log_ctx)
                                    full_video_item['transcript'] = "" # 또는 None, 일관성 유지
                                    full_video_item['transcript_language'] = None

                                yt_task_results.append(full_video_item)
                        else:
                             logger.info(f"No YouTube videos found for '{current_query_text}'.", extra=yt_log_ctx)

                    except Exception as e_main_yt:
                        error_msg_yt_task = f"Youtube & transcript task failed for query '{current_query_text}': {e_main_yt}"
                        logger.exception(error_msg_yt_task, extra=yt_log_ctx)
                        yt_task_error_log.append({
                            "stage": f"{node_name}.{yt_tool_name}_Task", "query": current_query_text, "error": str(e_main_yt),
                            "detail": traceback.format_exc(), "timestamp": datetime.now(timezone.utc).isoformat()
                        })

                    if yt_task_error_log:
                        return yt_task_error_log # Return error log list
                    return yt_task_results # Return results list

                search_tasks_for_current_query.append(Youtube_and_transcript_task_runner(query_text, trace_id))

            if search_tasks_for_current_query:
                logger.info(f"Executing {len(search_tasks_for_current_query)} search tasks in parallel for query: '{query_text}'", extra=query_extra_log)
                results_from_all_tasks = await asyncio.gather(*search_tasks_for_current_query, return_exceptions=True)

                for single_task_outcome in results_from_all_tasks:
                    if isinstance(single_task_outcome, list):
                        # 결과 리스트이거나 오류 로그 리스트일 수 있음
                        if single_task_outcome and isinstance(single_task_outcome[0], dict) and "stage" in single_task_outcome[0] and "error" in single_task_outcome[0]:
                            error_log.extend(single_task_outcome) # 오류 로그 리스트인 경우
                        else: # 실제 검색 결과인 경우
                            raw_search_results_accumulator.extend(item for item in single_task_outcome if isinstance(item, dict))
                    elif isinstance(single_task_outcome, Exception):
                        logger.error(f"A top-level task aggregation failed for query '{query_text}': {single_task_outcome}", extra=query_extra_log)
                        error_log.append({
                            "stage": f"{node_name}.gather_exception", "query": query_text, "error": str(single_task_outcome),
                            "detail": traceback.format_exc(), "timestamp": datetime.now(timezone.utc).isoformat()
                        })
            else:
                logger.info(f"No search tasks to execute for query: '{query_text}'", extra=query_extra_log)

        unique_results = []
        seen_identifiers = set()

        for item in raw_search_results_accumulator:
            identifier = None
            # self.youtube_tool_identifier를 여기서 사용 (오류 1 수정)
            if item.get('tool_used') == self.youtube_tool_identifier and item.get('video_id'):
                identifier = f"youtube_{item['video_id']}"
            elif item.get('url'):
                identifier = item['url']

            if identifier and identifier not in seen_identifiers:
                unique_results.append(item)
                seen_identifiers.add(identifier)
            elif not identifier:
                unique_results.append(item)

        logger.info(
            f"Total raw search results collected across all queries: {len(raw_search_results_accumulator)}, Unique results after final processing: {len(unique_results)}",
            extra=extra_log_data)

        update_dict = {
            "raw_search_results": unique_results,
            "current_stage": "n05_report_generation",
            "error_log": error_log
        }

        # 로깅용 요약 정보 (오류 2 수정)
        update_dict_summary_for_log = {}
        for k, v in update_dict.items():
            if k == "raw_search_results":
                update_dict_summary_for_log['raw_search_results_count'] = len(v) if isinstance(v, list) else 0
            else:
                update_dict_summary_for_log[k] = v

        logger.info(
            f"Exiting node. Search execution complete. Output Update Summary: {summarize_for_logging(update_dict_summary_for_log, fields_to_show=['current_stage', 'raw_search_results_count', 'error_log'])}", # 수정된 요약 사용
            extra=extra_log_data)

        return update_dict