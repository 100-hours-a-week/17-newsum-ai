# ai/app/nodes/n04_execute_search_node.py
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from urllib.parse import urlparse  # 도메인 추출용

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.tools.search.Google_Search_tool import GoogleSearchTool  # 핵심 의존성

logger = get_logger(__name__)


class N04ExecuteSearchNode:
    """
    N03에서 수립된 검색 전략에 따라 실제 검색을 수행하고 결과를 수집하는 노드.
    (업그레이드됨) 사용자 지정 사이트 설정 및 고급 검색 옵션 동적 활용.
    """

    def __init__(self, search_tool: GoogleSearchTool):
        self.search_tool = search_tool
        # 노드 자체의 기본/Fallback 사이트 선호도 (설정 파일 등으로 관리 가능)
        self.default_site_preferences = {
            "tech_expert_3": {  # 예시 writer_id 기반
                "code_related": ["stackoverflow.com", "github.com"],
                "research_paper": ["arxiv.org", "semanticscholar.org"],
            },
            "community_search_defaults": ["reddit.com"]  # 커뮤니티 검색 기본
        }
        # 쿼리 키워드와 사이트 카테고리 매핑 (예시, 더 정교화 가능)
        self.query_keyword_to_site_category_map = {
            "code_related": ["code", "python", "javascript", "error", "github", "stack overflow", "programming",
                             "develop"],
            "research_paper": ["paper", "research", "study", "arxiv", "publication", "citation", "preprint"],
            "deep_dive_tech": ["deep dive", "technical analysis", "tutorial", "how it works", "internals"],
            "community": ["review", "opinion", "forum", "discussion", "vs", "recommendation", "advice"],
            "news": ["news", "latest update", "announcement"]
        }

    def _get_relevant_user_sites(self, user_preferences: Dict[str, List[str]], query_text: str) -> List[str]:
        """사용자 선호 사이트 중 현재 쿼리와 관련된 사이트 목록 반환"""
        relevant_sites = []
        query_lower = query_text.lower()
        for category, keywords in self.query_keyword_to_site_category_map.items():
            if any(kw in query_lower for kw in keywords):
                sites_in_category = user_preferences.get(category)
                if isinstance(sites_in_category, list):
                    relevant_sites.extend(sites_in_category)
        return list(set(relevant_sites))  # 중복 제거

    def _determine_target_sites(
            self, writer_concept: dict, config: dict, tool_name: str, query_text: str
    ) -> Optional[List[str]]:
        """(업그레이드됨) 사용자 지정 사이트 설정을 우선 사용하고, 없으면 동적 결정"""
        sites = []
        writer_id = config.get("writer_id", "default_writer")

        # --- 1. 사용자 지정 사이트 설정 확인 (`user_site_preferences` 키 사용) ---
        user_preferences = config.get("user_site_preferences")
        if isinstance(user_preferences, dict):
            logger.debug(f"Using user site preferences: {user_preferences}")
            user_relevant_sites = self._get_relevant_user_sites(user_preferences, query_text)
            if user_relevant_sites:
                logger.info(f"Applying user-defined sites for query '{query_text}': {user_relevant_sites}")
                return user_relevant_sites  # 사용자 설정이 있고 매칭되면 그것만 사용
            else:
                logger.debug("User site preferences found but no matching category for this query. Falling back.")

        # --- 2. Fallback: 노드 기본 설정 또는 동적 로직 ---
        # writer_id 기반 선호도 적용
        if writer_id in self.default_site_preferences:
            default_prefs = self.default_site_preferences[writer_id]
            query_lower = query_text.lower()  # 여기서 다시 소문자 변환
            # 카테고리 매핑 활용
            for category, keywords in self.query_keyword_to_site_category_map.items():
                if category in default_prefs and any(kw in query_lower for kw in keywords):
                    sites.extend(default_prefs[category])

        # 도구별 기본 사이트 추가 (쿼리에 site: 없으면)
        if tool_name == "GoogleCSE_CommunitySearch" and "site:" not in query_text.lower():
            sites.extend(self.default_site_preferences.get("community_search_defaults", []))

        # 기타 audience 기반 로직 추가 가능

        final_sites = list(set(sites)) if sites else None
        if final_sites:
            logger.debug(f"Applying fallback/default sites for query '{query_text}': {final_sites}")
        return final_sites

    def _prepare_advanced_search_options(
            self, writer_concept: dict, tool_name: str, query_text: str
    ) -> Dict[str, Any]:
        """작가 성향, 도구, 쿼리를 바탕으로 고급 검색 옵션 준비"""
        options: Dict[str, Any] = {}
        depth = writer_concept.get("depth", "medium")
        trend_sensitivity = writer_concept.get("trend_sensitivity", "medium")

        # 1. 날짜 제한
        if tool_name == "GoogleCSE_NewsSearch":  # 뉴스 검색 시에는 거의 항상 날짜 제한 고려
            if trend_sensitivity == "high":
                options["dateRestrict"] = "m1"  # 최근 1개월
            elif trend_sensitivity == "medium":
                options["dateRestrict"] = "m3"  # 최근 3개월
            # low는 제한 없음 (기본값)
        elif trend_sensitivity == "high" and "latest" in query_text.lower():  # 일반 웹/커뮤니티 등에서도 최신 정보 중요시
            options["dateRestrict"] = "m3"  # 예: 지난 3개월

        # 2. 파일 형식
        if depth == "high" and any(
                kw in query_text.lower() for kw in self.query_keyword_to_site_category_map.get("research_paper", [])):
            # 연구/논문 관련 키워드가 있고 depth가 높으면 PDF 우선 검색 시도
            options["fileType"] = "pdf"

        # 3. 기타 옵션 (필요시 추가)
        # 예: options["safe"] = "active"

        if options:
            logger.debug(f"Prepared advanced search options: {options}")
        return options

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')

        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        logger.info(f"Entering node. Executing searches based on strategy.", extra=extra_log_data)

        raw_search_results: List[Dict[str, Any]] = []
        error_log = list(state.error_log or [])

        search_strategy = state.search_strategy
        # --- 입력 유효성 검사 ---
        if not search_strategy or not isinstance(search_strategy, dict) or not search_strategy.get("queries"):
            msg = "Search strategy or queries missing in state. Skipping search execution."
            logger.warning(msg, extra=extra_log_data)
            error_log.append({"stage": node_name, "error": msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            return {
                "raw_search_results": [],
                "current_stage": "n05_report_generation",  # 다음 노드로 이동은 하되 결과는 없음
                "error_log": error_log
            }
        # ------------------------

        queries_from_n03: List[str] = search_strategy.get("queries", [])
        selected_tools: List[str] = search_strategy.get("selected_tools", ["GoogleCSE_WebSearch"])
        writer_concept: Dict[str, Any] = search_strategy.get("writer_concept", {})
        base_search_params: Dict[str, Any] = search_strategy.get("parameters", {"max_results_per_query": 5})

        logger.info(f"Search plan: {len(queries_from_n03)} queries using tools: {selected_tools}", extra=extra_log_data)
        logger.debug(f"Writer concept for search: {writer_concept}", extra=extra_log_data)
        logger.debug(f"Base search params: {base_search_params}", extra=extra_log_data)

        for query_text in queries_from_n03:
            query_extra_log = {**extra_log_data, "current_query": query_text}  # 개별 쿼리 로깅용
            if not query_text or not isinstance(query_text, str) or query_text.strip().upper() == "N/A":
                logger.warning(f"Skipping invalid query: '{query_text}'", extra=query_extra_log)
                continue

            for tool_name in selected_tools:
                tool_extra_log = {**query_extra_log, "search_tool": tool_name}
                search_params = base_search_params.copy()

                # 1. 동적 대상 사이트 결정
                target_sites = self._determine_target_sites(writer_concept, config, tool_name, query_text)

                # 2. 고급 검색 옵션 준비
                advanced_options = self._prepare_advanced_search_options(writer_concept, tool_name, query_text)
                search_params.update(advanced_options)

                # GoogleSearchTool에 전달할 최종 파라미터 정리
                max_results = search_params.pop("max_results_per_query", 5)  # max_results는 별도 인자
                # GoogleSearchTool의 **kwargs에 필요 없는 파라미터 제거 (예: base_search_params에만 있던 것)
                # 현재는 search_params에 CSE API 직접 파라미터만 남음 (dateRestrict, fileType 등)

                try:
                    results: Optional[List[Dict[str, Any]]] = None
                    log_suffix = f"for query '{query_text}' with options {search_params}"
                    if target_sites: log_suffix += f" on sites {target_sites}"

                    logger.info(f"Executing {tool_name} {log_suffix}", extra=tool_extra_log)

                    # 검색 도구 호출 분기
                    if target_sites:
                        # 특정 사이트 검색 시에는 search_specific_sites_via_cse 사용
                        results = await self.search_tool.search_specific_sites_via_cse(
                            keyword=query_text,
                            sites=target_sites,
                            max_results=max_results,
                            trace_id=trace_id,
                            **search_params  # dateRestrict, fileType 등 전달
                        )
                    elif tool_name == "GoogleCSE_WebSearch":
                        results = await self.search_tool.search_web_via_cse(
                            keyword=query_text, max_results=max_results, trace_id=trace_id, **search_params
                        )
                    elif tool_name == "GoogleCSE_NewsSearch":
                        results = await self.search_tool.search_news_via_cse(
                            keyword=query_text, max_results=max_results, trace_id=trace_id, **search_params
                        )
                    elif tool_name == "GoogleCSE_CommunitySearch":
                        # 커뮤니티 검색은 target_sites가 결정되어야 유의미함 (위 target_sites 블록에서 처리됨)
                        # 만약 target_sites가 결정 안됐으면 아래는 실행되지 않음 (또는 에러 처리)
                        # 이 분기는 target_sites가 None일 때의 Fallback 처리가 필요하면 추가
                        logger.warning(
                            f"CommunitySearch called without specific target sites derived for query '{query_text}'. Search might be ineffective.",
                            extra=tool_extra_log)
                        # Fallback으로 일반 웹 검색 수행 가능
                        # results = await self.search_tool.search_web_via_cse(...)
                    else:
                        logger.warning(f"Unsupported search tool configured: {tool_name}", extra=tool_extra_log)
                        error_log.append({
                            "stage": f"{node_name}.{tool_name}",
                            "error": f"Unsupported search tool: {tool_name}",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        continue  # 다음 도구 또는 쿼리로

                    # 결과 처리 및 메타데이터 추가
                    if results:
                        logger.info(f"Found {len(results)} results using {tool_name} for query '{query_text}'.",
                                    extra=tool_extra_log)
                        for res_item in results:
                            if isinstance(res_item, dict):  # 결과 항목 타입 체크
                                res_item['query_source'] = query_text
                                res_item['tool_used'] = tool_name
                                res_item['retrieved_at'] = datetime.now(timezone.utc).isoformat()
                                res_item['source_domain'] = urlparse(res_item.get("url", "")).netloc  # 출처 도메인 추가
                            else:
                                logger.warning(f"Unexpected item type in search results: {type(res_item)}",
                                               extra=tool_extra_log)
                        raw_search_results.extend([item for item in results if isinstance(item, dict)])  # Dict 타입만 추가
                    else:
                        logger.info(f"No results found using {tool_name} for query '{query_text}'.",
                                    extra=tool_extra_log)

                except Exception as e:
                    error_msg = f"Search execution failed for query '{query_text}' using {tool_name}: {e}"
                    logger.exception(error_msg, extra=tool_extra_log)
                    error_log.append({
                        "stage": f"{node_name}.{tool_name}",
                        "query": query_text,
                        "error": str(e),
                        "detail": traceback.format_exc(),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

        # --- 최종 결과 정리 (중복 제거 등) ---
        unique_results = []
        seen_links = set()
        for item in raw_search_results:
            link = item.get('url')
            # 유효한 link가 있고, 아직 추가되지 않았으면 추가
            if link and link not in seen_links:
                unique_results.append(item)
                seen_links.add(link)
            elif not link:
                # 링크 없는 결과는 일단 포함 (예: 일부 API 응답 형식)
                unique_results.append(item)

        logger.info(
            f"Total raw search results collected: {len(raw_search_results)}, Unique results by link: {len(unique_results)}",
            extra=extra_log_data)

        # --- 상태 업데이트 사전 준비 ---
        update_dict = {
            "raw_search_results": unique_results,
            "current_stage": "n05_report_generation",  # 다음 스테이지
            "error_log": error_log  # 누적된 오류 로그
        }

        # 로깅용 요약 정보 추가
        update_dict_summary = update_dict.copy()
        update_dict_summary['raw_search_results_count'] = len(unique_results)
        logger.info(
            f"Exiting node. Search execution complete. Output Update Summary: {summarize_for_logging(update_dict_summary, fields_to_show=['current_stage', 'raw_search_results_count'])}",
            extra=extra_log_data)

        return update_dict