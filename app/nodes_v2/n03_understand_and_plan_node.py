# ai/app/nodes_v2/n03_understand_and_plan_node.py
import json
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)

class N03UnderstandAndPlanNode:
    """
    (업그레이드됨) 단일 LLM 호출을 통해 사용자 질의의 진의를 파악하고,
    검색 전략 및 모호성 해소를 위한 구조화된 JSON 응답을 파싱하여 상태에 반영합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_system_prompt(self, audience: str) -> str:
        # System-level instructions and JSON schema only
        schema = {
            "refined_intent": "<string>",
            "key_aspects_to_search": ["<string>"],
            "resolutions_for_ambiguities": {"<term>": "<string>"},
            "unresolved_ambiguities": ["<string>"],
            "clarification_error": "<string|null>"
        }
        schema_example = json.dumps(schema, ensure_ascii=False, indent=2)
        return f"""
You are an expert planning agent for a satirical 4-panel comic based on news.
Your target audience is {audience}.
Return ONLY valid JSON matching the schema below, NO think tags, NO explanation, NO comments.

JSON Schema:
{schema_example}
"""

    def _build_user_prompt(self, state: WorkflowState) -> str:
        # User message contains original query and context snippets
        query_sec = state.query
        audience = state.config.config.get('target_audience', 'general_public')
        original = query_sec.original_query or ""
        snippets = query_sec.initial_context_results or []
        snippet_text = "\n".join([
            f"- {r.get('source','?')}: {r.get('snippet','')[:100]}..."
            for r in snippets
        ])
        return f"""
User Original Query:
{original}

Target Audience: {audience}

Initial Context Snippets:
{snippet_text}
"""

    def _adjust_audience_by_category(self, category: str) -> str:
        mapping = {
            "IT": "tech_industry",
            "Economy": "financial_analysts",
            "Politics": "policy_makers",
            # "Leisure": "general_public",
            # "Science": "academic_researchers",
            "Other": "general_public"
        }
        return mapping.get(category, "general_public")

    def _determine_writer_concept(self, refined_intent: str, query_type: Optional[str], config: dict) -> Dict[
        str, Any]:  # config 추가
        writer_id = config.get('writer_id', 'default_writer')  # writer_id 직접 활용 가능
        target_audience = config.get('target_audience', 'general_public')

        concept = {"style": "neutral", "depth": "medium", "audience": target_audience, "trend_sensitivity": "medium"}

        # writer_id에 따른 기본 컨셉 조정 (예시)
        if "tech_expert" in writer_id:
            concept.update({"depth": "high", "style": "analytical"})
        elif "story_teller" in writer_id:
            concept.update({"style": "narrative"})

        intent_lower = refined_intent.lower();
        query_type_str = str(query_type).lower()
        if "how-to" in query_type_str or "how to" in intent_lower or "방법" in refined_intent: concept.update(
            {"style": "instructional"})
        if "comparison" in query_type_str or "vs" in intent_lower or "비교" in refined_intent: concept.update(
            {"style": "analytical", "depth": "high"})
        if "최신" in refined_intent or "동향" in refined_intent or "trend" in intent_lower or "latest" in intent_lower: concept["trend_sensitivity"] = "high"
        if len(refined_intent) > 80 and concept["depth"] != "high": concept[
            "depth"] = "high"  # tech_expert가 아니어도 high로 설정 가능

        return concept
    
    def _select_search_tools(self, concept: dict, key_aspects: list[str]) -> List[str]:
        # (기존 로직 유지)
        tools = set(["GoogleCSE_WebSearch"])
        if concept.get("depth") == "high": tools.add("GoogleCSE_NewsSearch")
        if concept.get("trend_sensitivity") == "high": tools.add("GoogleCSE_NewsSearch")
        if any(
                "opinion" in aspect.lower() or "review" in aspect.lower() or "후기" in aspect or "리뷰" in aspect or "평판" in aspect
                for aspect in key_aspects):
            tools.add("GoogleCSE_CommunitySearch")
        return list(tools)

    def _generate_final_search_queries(self, refined_intent: str, resolved_ambiguities_dict: dict,
                                       key_aspects: list[str], concept: dict, query_context_full: dict,
                                       selected_tools: list[str]) -> List[str]:
        # (이전 응답에서 복사, 로깅 개선 부분 포함)
        final_queries = []
        replacements = {}
        for k_with_quotes, v_resolution in resolved_ambiguities_dict.items():
            if isinstance(v_resolution, str) and \
                    "Clarification needed:" not in v_resolution and \
                    "No specific clue" not in v_resolution and \
                    "sLLM call failed" not in v_resolution:
                term_key = k_with_quotes.strip("'")
                # 만약 resolution이 너무 길면, 검색어 치환에는 부적합할 수 있음 (여기서는 그대로 사용)
                replacements[term_key] = v_resolution.strip()

        logger.debug(f"Applying ambiguity resolutions for query generation: {replacements}",
                     extra=query_context_full.get('_extra_log_data', {}))

        processed_aspects = set()
        for aspect in key_aspects[:5]:
            query = aspect
            for ambiguous_term, replacement_phrase in replacements.items():
                # 원본 모호성 용어가 aspect에 있고, replacement가 너무 길지 않을 때만 치환
                if ambiguous_term in query and len(replacement_phrase) < (
                        len(ambiguous_term) + 30):  # 예: 해결책이 원본보다 30자 이상 길면 치환 안함
                    query = query.replace(ambiguous_term, replacement_phrase)

            query_suffix = ""
            negative_keywords = query_context_full.get("negative_keywords", [])
            if negative_keywords:
                query_suffix += " " + " ".join([f"-{neg_kw.strip()}" for neg_kw in negative_keywords if neg_kw.strip()])

            final_query = (query + query_suffix).strip()
            if final_query and final_query not in processed_aspects:
                final_queries.append(final_query)
                processed_aspects.add(final_query)

        if not final_queries and refined_intent:
            final_queries.append(refined_intent)

        return final_queries[:5]

    def _define_search_parameters(self, concept: dict) -> dict:
        # (기존 로직 유지)
        params = {"max_results_per_query": 5}
        if concept.get("depth") == "high":
            params["max_results_per_query"] = 7
        elif concept.get("depth") == "shallow":
            params["max_results_per_query"] = 3
        return params
    
    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta = state.meta
        query_sec = state.query
        search_sec = state.search
        extra = {"trace_id": meta.trace_id, "node": self.__class__.__name__}

        if not query_sec.original_query or query_sec.initial_context_results is None:
            msg = "Missing inputs for N03"
            logger.error(msg, extra=extra)
            meta.current_stage = "ERROR"
            meta.error_message = msg
            return {"meta": meta.model_dump()}

        # 1) 청중 설정: category 기반
        query_category = query_sec.query_context.get("query_category", "Other")
        target_audience = self._adjust_audience_by_category(query_category)
        cfg = dict(state.config.config)
        cfg["target_audience"] = target_audience
        state.config.config = cfg

        # 2) Prepare prompts
        system_prompt = self._build_system_prompt(target_audience)
        user_prompt = self._build_user_prompt(state)

        # messages 리스트 생성
        messages_for_llm = [
            {"role": "system", "content": system_prompt},  # prompt를 system 역할로
            {"role": "user", "content": user_prompt}  # original_query를 user 역할로
        ]

        # 수정된 방식으로 LLMService 호출
        llm_resp = await self.llm_service.generate_text(
            messages=messages_for_llm,
            temperature=0.2,
            max_tokens=1200
        )
        raw_txt = llm_resp.get("generated_text", "")
        logger.debug("Raw N03 LLM response: %s", raw_txt, extra=extra)
        logger.debug("Target Audience: %s", target_audience, extra=extra)

        # 3) Parse JSON output
        try:
            parsed = extract_json(raw_txt)
            logger.debug("Parsed N03 JSON: %s", json.dumps(parsed, ensure_ascii=False), extra=extra)
        except Exception as e:
            logger.error("JSON parsing failed in N03: %s", str(e), extra=extra)
            raise

        # 4) Update state.query.query_context with parsed values
        ctx = query_sec.query_context or {}
        ctx["refined_intent"] = parsed.get("refined_intent")
        ctx["key_aspects_to_search"] = parsed.get("key_aspects_to_search", [])
        ctx["resolutions_for_ambiguities"] = parsed.get("resolutions_for_ambiguities", {})
        ctx["unresolved_ambiguities"] = parsed.get("unresolved_ambiguities", [])
        ctx["clarification_error"] = parsed.get("clarification_error")
        query_sec.query_context = ctx

        # 5) 기존 로직: 검색 전략 생성
        writer_concept = self._determine_writer_concept(
            ctx.get("refined_intent", query_sec.original_query),
            ctx.get("query_type"), state.config.config
        )
        selected_tools = self._select_search_tools(
            writer_concept, ctx.get("key_aspects_to_search", [])
        )
        final_queries = self._generate_final_search_queries(
            ctx.get("refined_intent", query_sec.original_query),
            ctx.get("resolutions_for_ambiguities", {}),
            ctx.get("key_aspects_to_search", [query_sec.original_query]),
            writer_concept,
            ctx,
            selected_tools
        )
        search_sec.search_strategy = {
            "writer_concept": writer_concept,
            "selected_tools": selected_tools,
            "queries": final_queries,
            "parameters": self._define_search_parameters(writer_concept)
        }

        meta.current_stage = "n04_execute_search"
        logger.info(f"N03 completed: queries={final_queries}", extra=extra)
        return {
            "query": query_sec.model_dump(),
            "search": search_sec.model_dump(),
            "meta": meta.model_dump()
        }

    # _determine_writer_concept, _select_search_tools,
    # _generate_final_search_queries, _define_search_parameters 등 기존 로직 유지
