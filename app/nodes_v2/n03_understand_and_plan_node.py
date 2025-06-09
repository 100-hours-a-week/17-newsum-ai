# ai/app/nodes_v2/n03_understand_and_plan_node.py
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from app.workflows.state_v2 import WorkflowState, QuerySection as QuerySectionModel, \
    SearchSection as SearchSectionModel, ConfigSection as ConfigSectionModel
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json
from app.services.postgresql_service import PostgreSQLService

logger = get_logger(__name__)

NODE_ORDER = 3
MAX_KEY_ASPECTS_FOR_LLM = 5
MAX_FINAL_GENERAL_QUERIES = 3
MIN_DOMAINS_TO_FETCH = 30  # 도메인 최소 조회 개수
DEFAULT_KOREAN_CATEGORIES_FOR_FALLBACK = ['mainstream_news', 'broadcast_news', 'news_agency']


class N03UnderstandAndPlanNode:
    def __init__(self, llm_service: LLMService, postgresql_service: PostgreSQLService):
        self.llm_service = llm_service
        self.postgresql_service = postgresql_service

    def _build_system_prompt(self, audience: str) -> str:
        schema = {
            "refined_intent": "<A concise, actionable rephrasing of the user's core information need, in English.>",
            "key_aspects_to_search": [f"<English keyword phrase 1 to be used for targeted searches>",
                                      "<English keyword phrase 2>",
                                      f"<Up to {MAX_KEY_ASPECTS_FOR_LLM} key aspects. Each aspect should be a specific search query.>"],
            "resolutions_for_ambiguities": {
                "<ambiguous_term_in_query_english>": "<Suggested clarification or resolution in English. If no resolution, explain why.>"},
            "unresolved_ambiguities": [
                "<English description of any ambiguity that could not be resolved and critically needs user clarification. If none, this should be an empty list.>"],
            "clarification_error": "<If direct clarification from the user is absolutely necessary to proceed, provide a concise error message in English here (e.g., 'Query too vague, please specify X'). Otherwise, this field MUST be null.>",
            "planning_reasoning_korean": {
                "intent_refinement_rationale_korean": "<한글 설명: 사용자의 원본 질의를 어떻게 분석하여 'refined_intent'를 도출했는지, 그 과정과 핵심 근거를 상세히 기술합니다.>",
                "search_aspects_derivation_rationale_korean": f"<한글 설명: '{MAX_KEY_ASPECTS_FOR_LLM}개 이내의 'key_aspects_to_search'를 선정한 기준과 각 항목이 사용자의 정보 요구를 해결하는 데 어떻게 기여하는지 설명합니다.>",
                "ambiguity_handling_rationale_korean": "<한글 설명: 질의에서 발견된 모호성이 있었다면 이를 어떻게 처리했는지 설명합니다.>"
            }
        }
        json_schema_example_str = json.dumps(schema, ensure_ascii=False, indent=2)

        return f"""
You are an expert planning agent for a satirical 4-panel comic based on news. Your primary role is to deeply understand the user's query, refine it into an actionable intent, identify key search aspects (which will become search queries), and address ambiguities. You must also provide a detailed rationale for your planning decisions in Korean.

The target audience for the final comic is: {audience}. Keep this in mind when assessing intent and information needs, but your direct output for intent, keywords, and ambiguities should be factual and in English. Only the 'planning_reasoning_korean' field should be in Korean.

Return ONLY a single, valid JSON object that strictly adheres to the schema provided below. Do NOT include any extraneous text, thoughts, or comments outside this JSON structure.

JSON Schema:
{json_schema_example_str}
"""

    def _build_user_prompt(self, state: WorkflowState) -> str:
        query_sec = state.query
        original_query_str = query_sec.original_query or ""
        initial_context_snippets = query_sec.initial_context_results or []

        snippets_text_for_prompt = "\n\n".join([
            f"Snippet {idx + 1}:\n- Source: {r.get('source', r.get('source_domain', 'Unknown'))}\n- URL: {r.get('url', 'N/A')}\n- Title: {r.get('title', 'N/A')}\n- Snippet Text: {r.get('snippet', '')[:400]}..."
            for idx, r in enumerate(initial_context_snippets[:5])
        ])
        if not initial_context_snippets:
            snippets_text_for_prompt = "No initial context snippets were available from the pre-search step (N02)."

        return f"""
# User's Original Query:
{original_query_str}

# Initial Context Snippets (from N02 pre-search, use these to understand the query better):
{snippets_text_for_prompt}

Please analyze the query and the provided context snippets to generate the structured plan and your detailed reasoning, strictly following the JSON schema provided in the system prompt.
Ensure that 'refined_intent' and 'key_aspects_to_search' are in English and are precise enough to be used as effective search queries.
The 'planning_reasoning_korean' field must contain your justifications in Korean.
If no ambiguities are found, 'resolutions_for_ambiguities' should be an empty object and 'unresolved_ambiguities' an empty list.
'clarification_error' must be null unless user clarification is absolutely essential to proceed.
"""

    async def _fetch_domains_from_db_generic(
            self,
            work_id_log: str,
            country_code: Optional[str] = None,
            categories: Optional[List[str]] = None,
            keywords: Optional[List[str]] = None,
            limit: int = MIN_DOMAINS_TO_FETCH
    ) -> List[str]:
        if not self.postgresql_service:
            logger.error("PostgreSQLService not injected. Cannot fetch domains from DB.",
                         extra={"work_id": work_id_log})
            return []

        query_parts = []
        params: list[Any] = []

        if country_code:
            query_parts.append("country = %s")
            params.append(country_code.upper())

        if categories:
            valid_categories = [cat for cat in categories if
                                cat and cat.strip() and cat.lower() not in ['other', 'unknown', 'ambiguous']]
            if valid_categories:
                query_parts.append("category = ANY(%s)")
                params.append(valid_categories)


        sql_where_clause = " AND ".join(query_parts) if query_parts else "TRUE"
        sql_query = f"""
            SELECT domain 
            FROM ai_test_seed_domains 
            WHERE {sql_where_clause};
        """
        current_params = tuple(params + [limit])

        try:
            logger.debug(f"Executing generic domain DB query: {sql_query} | Params: {current_params}",
                         extra={"work_id": work_id_log})
            results = await self.postgresql_service.fetch_all(sql_query, current_params)
            domains = [row[0] for row in results if row and row[0]]
            if domains:
                logger.info(f"Fetched {len(domains)} domains from DB: {domains}", extra={"work_id": work_id_log})
            else:
                logger.info("No domains found in DB for given criteria. Returning empty list.",
                            extra={"work_id": work_id_log})
            return domains
        except Exception as e:
            logger.error(f"Error fetching domains from DB: {e}. Returning empty list.",
                         extra={"work_id": work_id_log, "sql_query": sql_query})
            return []

    async def _get_default_korean_domains_from_db(self, work_id_log: str) -> List[str]:
        logger.info("Fetching default Korean domains from DB.", extra={"work_id": work_id_log})
        return await self._fetch_domains_from_db_generic(
            work_id_log=work_id_log,
            country_code='KR',
            categories=DEFAULT_KOREAN_CATEGORIES_FOR_FALLBACK,
            limit=MIN_DOMAINS_TO_FETCH
        )

    async def _get_relevant_seed_domains(self, category: Optional[str], keywords: List[str], work_id_log: str) -> List[
        str]:
        if not self.postgresql_service:
            logger.error("PostgreSQLService not available in _get_relevant_seed_domains. Cannot fetch domains.",
                         extra={"work_id": work_id_log})
            return []

        if not category and not keywords:
            logger.debug("No category or keywords for seed domain search. Using default Korean domains from DB.",
                         extra={"work_id": work_id_log})
            return await self._get_default_korean_domains_from_db(work_id_log)

        logger.debug("Attempting to fetch Korean seed domains first.", extra={"work_id": work_id_log})
        korean_domains = await self._fetch_domains_from_db_generic(
            work_id_log=work_id_log,
            country_code='KR',
            categories=[category] if category else None,
            keywords=keywords,
            limit=MIN_DOMAINS_TO_FETCH
        )
        if korean_domains:
            logger.info(f"Found {len(korean_domains)} Korean seed domains: {korean_domains}",
                        extra={"work_id": work_id_log})
            return korean_domains

        logger.debug("Korean seed domains not found or empty. Attempting general seed domain search.",
                     extra={"work_id": work_id_log})
        general_domains = await self._fetch_domains_from_db_generic(
            work_id_log=work_id_log,
            country_code=None,
            categories=[category] if category else None,
            keywords=keywords,
            limit=MIN_DOMAINS_TO_FETCH
        )
        if general_domains:
            logger.info(f"Found {len(general_domains)} general seed domains: {general_domains}",
                        extra={"work_id": work_id_log})
            return general_domains

        logger.info("No specific seed domains found. Falling back to default Korean domains from DB.",
                    extra={"work_id": work_id_log})
        return await self._get_default_korean_domains_from_db(work_id_log)

    def _adjust_audience_by_category(self, query_category: Optional[str], current_config: Dict[str, Any]) -> str:
        user_specified_audience = current_config.get("target_audience")
        if user_specified_audience and user_specified_audience.lower() not in [None, "general_public", "", "none"]:
            return user_specified_audience

        category_to_audience_map = {
            "it_tech": "tech-savvy individuals and developers",
            "startup": "entrepreneurs, startup employees, and venture capitalists",
            "economy_news": "business professionals, economists, and investors",
            "business": "business leaders, managers, and corporate strategists",
            "science": "researchers, scientists, and science enthusiasts",
            "academic_research": "academics, scholars, and university students",
            "politics": "politically engaged citizens, policymakers, and analysts",
            "mainstream_news": "general public with interest in current affairs",
            "community": "members of the specific online community or general internet users",
            "entertainment": "general audience interested in arts and culture",
            "sports": "sports fans and enthusiasts", "lifestyle": "general consumers interested in lifestyle topics",
            "design": "designers, artists, and creative professionals",
            "investigative": "citizens interested in in-depth journalism and accountability",
            "local_news": "residents of the specific local area"
        }
        return category_to_audience_map.get(query_category, "general_public") if query_category else "general_public"

    def _determine_writer_concept(self, refined_intent: str, query_type: Optional[str], config: Dict[str, Any],
                                  work_id_log: str) -> Dict[str, Any]:
        writer_id = config.get('writer_id', 'default_writer')
        target_audience = config.get('target_audience', 'general_public')
        logger.debug(
            f"[N03 WriterConcept] Input - WriterID: {writer_id}, Target Audience (adjusted): {target_audience}",
            extra={"work_id": work_id_log})

        concept = {"style": "neutral", "depth": "medium", "audience": target_audience, "trend_sensitivity": "medium"}

        if "tech_expert" in writer_id:
            concept.update({"depth": "high", "style": "analytical"})
        elif "story_teller" in writer_id:
            concept.update({"style": "narrative", "trend_sensitivity": "low"})

        intent_lower = refined_intent.lower()
        query_type_str = str(query_type).lower() if query_type else ""

        if "how-to" in query_type_str or "how to" in intent_lower or "방법" in refined_intent:
            concept.update({"style": "instructional", "depth": "medium"})
        if "comparison" in query_type_str or "vs" in intent_lower or "비교" in refined_intent:
            concept.update({"style": "analytical", "depth": "high"})
        if "latest" in intent_lower or "trend" in intent_lower or "최신" in refined_intent or "동향" in refined_intent:
            concept["trend_sensitivity"] = "high"
        if len(refined_intent) > 70 and concept["depth"] == "medium":
            concept["depth"] = "high"

        logger.info(f"[N03 WriterConcept] 결정된 작가 컨셉: {concept}", extra={"work_id": work_id_log})
        return concept

    def _select_search_tools(self, writer_concept: Dict[str, Any], key_aspects: List[str], tone: str,
                             work_id_log: str, has_target_seed_domains: bool) -> List[str]:
        tools = set()
        # 일반 웹 검색 무조건 포함
        tools.add("search_web_via_cse")

        # 트렌드 민감도 높으면 뉴스 검색 포함
        if writer_concept.get("trend_sensitivity") == "high":
            tools.add("search_news_via_cse")

        # 시드 도메인 있으면 site-specific 검색 포함
        if has_target_seed_domains:
            tools.add("search_specific_sites_via_cse")

        # 풍자 톤에 따라 커뮤니티/블로그/유튜브 가중
        tone_lower = tone.lower() if isinstance(tone, str) else ""
        # 날카로운 풍자(sarcasm)일 때는 커뮤니티 반응 우선
        if "sarcasm" in tone_lower or "satire" in tone_lower:
            tools.add("search_communities_via_cse")
        else:
            # 가벼운 유머·풍자(irony)일 때 블로그도 포함
            tools.add("search_blogs_via_cse")

        # 유튜브 영상은 풍자 톤 상관없이 포함
        tools.add("search_youtube_videos")

        logger.info(f"[N03 SearchTools] 선택된 주요 검색 도구: {list(tools)} (시드 도메인 사용 여부: {has_target_seed_domains})",
                    extra={"work_id": work_id_log})
        return list(tools)

    def _generate_final_search_queries(self, issue_summary: str, resolved_ambiguities: Dict[str, str],
                                       key_aspects: List[str], satire_targets: List[str], work_id_log: str) -> List[str]:
        final_queries: List[str] = []
        # 이슈 요약을 우선
        if issue_summary and issue_summary.strip():
            final_queries.append(issue_summary.strip())

        # 기존 키워드
        if key_aspects:
            for aspect_query in key_aspects:
                if aspect_query and isinstance(aspect_query, str) and aspect_query.strip():
                    final_queries.append(aspect_query.strip())

        # 풍자 대상과 이슈 결합 키워드 추가
        for target in satire_targets or []:
            combined = f"{target} + {issue_summary}" if issue_summary else target
            final_queries.append(combined)

        # 모호성 해결이 없다면 issue_summary 사용
        if not final_queries and issue_summary:
            final_queries.append(issue_summary.strip())

        deduplicated_queries = list(dict.fromkeys(final_queries))
        logger.info(f"[N03 FinalQueries] 생성된 일반 검색 쿼리: {deduplicated_queries[:MAX_FINAL_GENERAL_QUERIES]}",
                    extra={"work_id": work_id_log})
        return deduplicated_queries[:MAX_FINAL_GENERAL_QUERIES]

    def _define_search_parameters(self, writer_concept: Dict[str, Any], tone: str, work_id_log: str) -> Dict[str, Any]:
        # 채널별 고급 검색 파라미터 기본 템플릿
        tool_parameters = {
            "search_web_via_cse": {"max_results": 5, "dateRestrict": "m6", "safe": "off"},
            "search_news_via_cse": {"max_results": 5, "dateRestrict": "d7"},
            "search_specific_sites_via_cse": {"max_results": 5},
            "search_blogs_via_cse": {"max_results": 5},
            "search_communities_via_cse": {"max_results": 5, "dateRestrict": "y1"},
            "search_youtube_videos": {"max_results": 5, "publishedAfter": "2024-01-01T00:00:00Z"}
        }

        # 풍자 톤이 ‘유머러스’면 블로그 검색 시 “유머” 키워드 추가용 옵션
        tone_lower = tone.lower() if isinstance(tone, str) else ""
        if "irony" in tone_lower or "humor" in tone_lower:
            tool_parameters["search_blogs_via_cse"]["q_suffix"] = " 유머"
        # 풍자 톤이 ‘냉소적’이면 커뮤니티 필터 강화
        if "sarcasm" in tone_lower or "satire" in tone_lower:
            tool_parameters["search_communities_via_cse"]["dateRestrict"] = "m1"

        logger.info(f"[N03 SearchParams] 정의된 채널별 파라미터 템플릿: {tool_parameters}", extra={"work_id": work_id_log})
        return tool_parameters

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        query_sec = state.query
        config_sec_dict = state.config.config or {}

        work_id = meta_sec.work_id
        extra_log = {"work_id": work_id, "node_name": self.__class__.__name__, "node_order": NODE_ORDER}

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info("N03 노드 진입: 사용자 질의 이해, 시드 도메인 조회 및 검색 계획 수립 시작.", extra=extra_log)

        if not query_sec.original_query or query_sec.query_context is None:
            msg = "N03 실행을 위한 필수 입력(original_query 또는 query_context)이 누락되었습니다."
            logger.error(msg, extra=extra_log)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"meta": meta_sec.model_dump()}

        current_config_copy = dict(config_sec_dict)
        query_category_from_n02 = query_sec.query_context.get("query_category")
        adjusted_audience = self._adjust_audience_by_category(query_category_from_n02, current_config_copy)
        current_config_copy["target_audience"] = adjusted_audience
        logger.info(f"조정된 타겟 독자: '{adjusted_audience}' (원본 카테고리: '{query_category_from_n02}')", extra=extra_log)

        system_prompt_for_llm = self._build_system_prompt(audience=adjusted_audience)
        user_prompt_for_llm = self._build_user_prompt(state)

        request_id_for_llm = f"{work_id}_N03_UnderstandAndPlan"
        llm_messages = [{"role": "system", "content": system_prompt_for_llm},
                        {"role": "user", "content": user_prompt_for_llm}]

        logger.debug("질의 이해 및 계획 LLM 호출 시작...", extra=extra_log)
        llm_response = await self.llm_service.generate_text(
            messages=llm_messages, request_id=request_id_for_llm, temperature=0.05, max_tokens=2500
        )

        cleaned_llm_output = llm_response.get("generated_text", "")
        think_content = llm_response.get("think_content")

        if think_content:
            meta_sec.llm_think_traces.append({
                "node_name": self.__class__.__name__, "request_id": request_id_for_llm,
                "timestamp": datetime.now(timezone.utc).isoformat(), "log_content": think_content
            })
            logger.debug(f"LLM <think> 내용 저장됨 (Request ID: {request_id_for_llm})", extra=extra_log)

        if not cleaned_llm_output or llm_response.get("error"):
            error_detail = llm_response.get("error", "LLM이 계획 수립에 실패했거나 빈 응답을 반환했습니다.")
            logger.error(f"질의 이해 및 계획 LLM 호출 실패: {error_detail}", extra=extra_log)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"meta": meta_sec.model_dump(),
                    "config": ConfigSectionModel(config=current_config_copy).model_dump()}

        logger.debug(f"질의 이해 및 계획 LLM 원본 응답 (정리 후): {summarize_for_logging(cleaned_llm_output, 400)}", extra=extra_log)

        try:
            parsed_llm_json = extract_json(cleaned_llm_output)
            if not isinstance(parsed_llm_json, dict):
                raise ValueError(f"extract_json이 계획 LLM 응답에 대해 딕셔너리를 반환하지 않았습니다. 출력: {parsed_llm_json}")
            logger.info("질의 이해 및 계획 LLM JSON 파싱 성공.", extra=extra_log)
        except Exception as e:
            logger.error(f"질의 이해 및 계획 LLM JSON 파싱 실패: {e}. 원본(정리후): {cleaned_llm_output}", extra=extra_log)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"meta": meta_sec.model_dump(),
                    "config": ConfigSectionModel(config=current_config_copy).model_dump()}

        # N02에서 넘어온 이슈 요약, 풍자 대상, 톤을 가져옴
        issue_summary = parsed_llm_json.get("issue_summary_for_comic", "")
        satire_targets = parsed_llm_json.get("satire_target", []) or []
        tone_suggestion = parsed_llm_json.get("tone_suggestion", "")

        # refined_intent 결정: 우선 이슈 요약, 없으면 기존 refined_intent
        final_refined_intent = issue_summary.strip() if issue_summary else parsed_llm_json.get("refined_intent", query_sec.original_query or "")

        # 기존 키워드
        key_aspects = parsed_llm_json.get("key_aspects_to_search", []) or []

        # 풍자 대상 결합 키워드 추가
        for target in satire_targets:
            composite = f"{target} + {issue_summary}" if issue_summary else target
            key_aspects.append(composite)

        # 모호성 해소 내용
        resolved_ambiguities = parsed_llm_json.get("resolutions_for_ambiguities", {}) or {}
        query_type = parsed_llm_json.get("query_type")

        # planning_reasoning_korean 구성
        planning_reasoning_dict = parsed_llm_json.get("planning_reasoning_korean", {})
        if isinstance(planning_reasoning_dict, dict):
            korean_searchplan_details_str = "\n\n".join([
                f"의도 정제 근거: {planning_reasoning_dict.get('intent_refinement_rationale_korean', '제공되지 않음')}",
                f"검색 핵심 요소 선정 근거: {planning_reasoning_dict.get('search_aspects_derivation_rationale_korean', '제공되지 않음')}",
                f"모호성 처리 근거: {planning_reasoning_dict.get('ambiguity_handling_rationale_korean', '제공되지 않음')}"
            ])
        else:
            korean_searchplan_details_str = "N03 LLM으로부터 검색 계획 근거(한글)를 수신하지 못했거나 내용이 없습니다."

        logger.info(f"N03 LLM 검색 계획 근거(한글): {summarize_for_logging(korean_searchplan_details_str, 200)}...", extra=extra_log)

        updated_query_context = dict(query_sec.query_context)
        updated_query_context.update({
            "refined_intent": final_refined_intent,
            "key_aspects_to_search": key_aspects,
            "resolutions_for_ambiguities": resolved_ambiguities,
            "unresolved_ambiguities": parsed_llm_json.get("unresolved_ambiguities", []),
            "clarification_error_from_n03_llm": parsed_llm_json.get("clarification_error"),
            "issue_summary_for_comic": issue_summary,
            "satire_target": satire_targets,
            "tone_suggestion": tone_suggestion
        })

        category_for_db_search = updated_query_context.get("query_category")
        keywords_for_db_search = key_aspects

        target_seed_domains = await self._get_relevant_seed_domains(
            category_for_db_search, keywords_for_db_search, work_id
        )

        writer_concept = self._determine_writer_concept(
            final_refined_intent, query_type, current_config_copy, work_id
        )
        selected_tools = self._select_search_tools(
            writer_concept, key_aspects, tone_suggestion, work_id, bool(target_seed_domains)
        )
        tool_parameters = self._define_search_parameters(writer_concept, tone_suggestion, work_id)
        general_search_queries = self._generate_final_search_queries(
            issue_summary, resolved_ambiguities, key_aspects, satire_targets, work_id
        )

        calculated_search_strategy = {
            "writer_concept": writer_concept,
            "selected_tools": selected_tools,
            "queries": general_search_queries,
            "target_seed_domains": target_seed_domains,
            "tool_parameters": tool_parameters
        }
        logger.info(
            f"검색 전략 생성 완료. 일반 쿼리: {general_search_queries}, 시드 도메인 수: {len(target_seed_domains)}, 도구: {selected_tools}",
            extra=extra_log
        )
        if target_seed_domains:
            logger.debug(f"사용될 시드 도메인 목록 (최대 5개 표시): {target_seed_domains[:5]}", extra=extra_log)

        meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"
        logger.info("N03 이해, 시드 도메인 조회 및 검색 계획 수립 완료. 다음 단계로 이동합니다.", extra=extra_log)

        return {
            "meta": meta_sec.model_dump(),
            "query": QuerySectionModel(
                original_query=query_sec.original_query,
                search_target_site_domain=target_seed_domains,
                initial_context_results=query_sec.initial_context_results,
                query_context=updated_query_context,
                llm_analysis_details_korean=query_sec.llm_analysis_details_korean,
                llm_analysis_details_searchplan_korean=korean_searchplan_details_str
            ).model_dump(),
            "search": SearchSectionModel(search_strategy=calculated_search_strategy).model_dump(),
            "config": ConfigSectionModel(config=current_config_copy).model_dump()
        }
