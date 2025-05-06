# ai/app/nodes/n03_understand_and_plan_node.py
import json
import traceback
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService

logger = get_logger(__name__)


class N03UnderstandAndPlanNode:
    """
    (업그레이드됨) Node 2 결과와 sLLM과의 여러 단계의 간단한 프롬프팅(config 활용, 모호성 해결 개선)을 통해
    사용자 질의의 진의를 파악하고, 검색 전략을 수립합니다.
    """
    MAX_CLARIFICATION_TURNS = 1  # 현재 순차적 방식에서는 이 값의 의미가 크지 않음

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def _get_refined_intent_sllm(
            self, original_query: str, n02_keywords: list[str], n02_query_type: str,
            initial_snippets: list[dict], config: dict,  # config 추가
            trace_id: str, extra_log_data: dict
    ) -> Optional[str]:
        snippets_summary = chr(10).join(
            [f"- {res.get('source', 'Unknown')}: {str(res.get('snippet', ''))[:100]}..." for res in initial_snippets])

        # --- config에서 writer_id 및 target_audience 가져오기 ---
        writer_id = config.get('writer_id', 'default_writer')
        target_audience = config.get('target_audience', 'general_public')  # N01에서 설정된 예시 config

        # --- 프롬프트 수정: config (writer_id, target_audience) 활용 ---
        prompt = f"""[System] You are an expert analyst. Your current persona is '{writer_id}' and you are addressing a '{target_audience}' audience.
Based on the user's original query, extracted keywords, predicted query type, and initial context snippets, reformulate the user's core information need into a single, clear sentence.
This sentence should represent their most likely true intent, staying faithful to the original query's core topic, and reflecting your persona and target audience.
Respond ONLY with the single reformulated sentence. Do NOT add any explanations or introductory phrases.

[User Inputs]
Original Query: "{original_query}"
Extracted Keywords: {n02_keywords}
Predicted Query Type: {n02_query_type}
Initial Context Snippets (Summarized):
{snippets_summary}

[Persona & Audience Context]
Writer Persona: "{writer_id}"
Target Audience: "{target_audience}"

[Task]
Reformulated Core Intent (single sentence, considering persona and audience):"""
        logger.debug(
            f"Attempting to get refined_intent string from sLLM (persona: {writer_id}, audience: {target_audience})...",
            extra=extra_log_data)
        result = await self.llm_service.generate_text(prompt=prompt, max_tokens=180, temperature=0.25)  # 온도 약간 더 높여 유연성

        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM refined_intent generation failed: {result.get('error', 'No text generated')}",
                         extra=extra_log_data)
            return None

        refined_intent_str = result["generated_text"].strip()
        refined_intent_str = refined_intent_str.removeprefix('"').removesuffix('"').strip()
        logger.debug(f"sLLM generated refined_intent string: '{refined_intent_str}'", extra=extra_log_data)
        return refined_intent_str if refined_intent_str else None

    async def _get_key_aspects_sllm(
            self, refined_intent: str, config: dict,  # config 추가
            trace_id: str, extra_log_data: dict
    ) -> List[str]:
        writer_id = config.get('writer_id', 'default_writer')
        target_audience = config.get('target_audience', 'general_public')

        # --- 프롬프트 수정: config (writer_id, target_audience) 활용 ---
        prompt = f"""[System] Your current persona is '{writer_id}' and the target audience is '{target_audience}'.
Based on the provided refined user intent, list up to 5 specific search queries or key aspects that would help answer this intent.
Each aspect should be a concrete search term or phrase, suitable for the persona and audience.
Respond ONLY with the list of aspects, one aspect per line. Do NOT add any numbering, bullet points, explanations, or introductory phrases.

Refined User Intent: "{refined_intent}"

[Persona & Audience Context]
Writer Persona: "{writer_id}"
Target Audience: "{target_audience}"

[Task]
Key Search Aspects (one per line, max 5, considering persona and audience):"""
        logger.debug(
            f"Attempting to get key_aspects_to_search from sLLM for intent: '{refined_intent}' (persona: {writer_id}, audience: {target_audience})...",
            extra=extra_log_data)
        result = await self.llm_service.generate_text(prompt=prompt, max_tokens=250, temperature=0.25)

        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM key_aspects generation failed: {result.get('error', 'No text generated')}",
                         extra=extra_log_data)
            return []

        aspects_text = result["generated_text"].strip()
        key_aspects_list = [line.strip() for line in aspects_text.splitlines() if line.strip()]
        cleaned_aspects = []
        for aspect in key_aspects_list[:5]:
            aspect = aspect.removeprefix('"').removesuffix('"')
            aspect = aspect.removeprefix('- ').removeprefix('* ').strip()
            if aspect:
                cleaned_aspects.append(aspect)
        logger.debug(f"sLLM generated key_aspects: {cleaned_aspects}", extra=extra_log_data)
        return cleaned_aspects

    async def _resolve_one_ambiguity_sllm(
            self, refined_intent: str, ambiguity_term: str, initial_snippets: list[dict], config: dict,  # config 추가
            trace_id: str, extra_log_data: dict
    ) -> str:
        snippets_summary = chr(10).join(
            [f"- {res.get('source', 'Unknown')}: {str(res.get('snippet', ''))[:80]}..." for res in initial_snippets])
        writer_id = config.get('writer_id', 'default_writer')

        # --- 프롬프트 수정: 검색어에 직접 활용 가능한 간결한 해결책 요청 ---
        prompt = f"""[System] Your current persona is '{writer_id}'.
For the given 'Refined User Intent', the term '{ambiguity_term}' has been identified as ambiguous.
Based on the intent and the 'Initial Context Snippets', provide a CONCISE clarification for '{ambiguity_term}'.
This clarification should ideally be a short phrase or a few keywords that could be used to make a search query more specific.
If the context is insufficient or the term is too vague to resolve with a concise phrase, respond with "Clarification needed: [briefly explain why]".
Respond ONLY with the concise clarification phrase/keywords OR the "Clarification needed..." statement. Do NOT add extra explanations.

Refined User Intent: "{refined_intent}"
Ambiguous Term: "{ambiguity_term}"
Initial Context Snippets (Summarized):
{snippets_summary}

[Persona Context]
Writer Persona: "{writer_id}"

[Task]
Concise Clarification for '{ambiguity_term}' (for search query use, or "Clarification needed: ..."):"""
        logger.debug(
            f"Attempting to resolve ambiguity '{ambiguity_term}' from sLLM (persona: {writer_id}, aiming for conciseness)...",
            extra=extra_log_data)
        result = await self.llm_service.generate_text(prompt=prompt, max_tokens=80, temperature=0.15)  # max_tokens 줄임

        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM resolution for '{ambiguity_term}' failed: {result.get('error', 'No text generated')}",
                         extra=extra_log_data)
            return f"Clarification needed: sLLM call failed for '{ambiguity_term}'"

        resolution_text = result["generated_text"].strip()
        resolution_text = resolution_text.removeprefix('"').removesuffix('"').strip()
        logger.debug(f"sLLM resolution for '{ambiguity_term}': '{resolution_text}'", extra=extra_log_data)
        return resolution_text

    async def _clarify_intent_via_sequential_prompting(
            self, original_query: str, n02_context: dict, initial_results: list[dict],
            config: dict,  # config 추가
            trace_id: str, extra_log_data: dict
    ) -> Dict[str, Any]:
        final_understanding = {
            "refined_intent": original_query,
            "resolutions_for_ambiguities": {},
            "key_aspects_to_search": [original_query],
            "unresolved_ambiguities": n02_context.get('detected_ambiguities', []),
            "clarification_error": None
        }
        sllm_errors = []

        # 1. 정제된 의도 생성
        try:
            n02_keywords = n02_context.get('extracted_keywords', [original_query])
            n02_query_type = n02_context.get('query_type', 'Unknown')

            refined_intent_str = await self._get_refined_intent_sllm(
                original_query, n02_keywords, n02_query_type, initial_results, config, trace_id, extra_log_data
            )
            if refined_intent_str:
                final_understanding["refined_intent"] = refined_intent_str
            else:
                sllm_errors.append("Failed to generate refined_intent.")
        except Exception as e:
            logger.error(f"Error in _get_refined_intent_sllm: {e}", extra=extra_log_data, exc_info=True)
            sllm_errors.append(f"Refined intent step errored: {e}")

        current_intent_for_aspects = final_understanding["refined_intent"]

        # 2. 검색할 핵심 주제 생성
        try:
            key_aspects_list = await self._get_key_aspects_sllm(
                current_intent_for_aspects, config, trace_id, extra_log_data
            )
            if key_aspects_list:
                final_understanding["key_aspects_to_search"] = key_aspects_list
            else:
                final_understanding["key_aspects_to_search"] = [current_intent_for_aspects]
                sllm_errors.append("Failed to generate key_aspects_to_search.")
        except Exception as e:
            logger.error(f"Error in _get_key_aspects_sllm: {e}", extra=extra_log_data, exc_info=True)
            sllm_errors.append(f"Key aspects step errored: {e}")

        # 3. 모호성 해결
        detected_ambiguities = n02_context.get('detected_ambiguities', [])
        resolutions = {}
        unresolved = []
        if detected_ambiguities:
            logger.debug(
                f"Attempting to resolve {len(detected_ambiguities)} detected ambiguities: {detected_ambiguities}",
                extra=extra_log_data)
            for term in detected_ambiguities:
                try:
                    resolution = await self._resolve_one_ambiguity_sllm(
                        final_understanding["refined_intent"], term, initial_results, config, trace_id, extra_log_data
                    )
                    resolutions[f"'{term}'"] = resolution
                    if "Clarification needed:" in resolution or "sLLM call failed" in resolution:
                        unresolved.append(term)
                except Exception as e:
                    logger.error(f"Error resolving ambiguity for '{term}': {e}", extra=extra_log_data, exc_info=True)
                    sllm_errors.append(f"Resolution for '{term}' errored: {e}")
                    resolutions[f"'{term}'"] = f"Clarification needed: Error during processing '{term}'"
                    unresolved.append(term)
            final_understanding["resolutions_for_ambiguities"] = resolutions
            final_understanding["unresolved_ambiguities"] = unresolved
        else:
            final_understanding["unresolved_ambiguities"] = []

        if sllm_errors:
            final_understanding["clarification_error"] = "; ".join(sllm_errors)
            logger.warning(f"sLLM processing in N03 encountered errors: {final_understanding['clarification_error']}",
                           extra=extra_log_data)

        logger.info(
            f"Final understanding after sequential prompting (upgraded N03): {summarize_for_logging(final_understanding)}",
            extra=extra_log_data)
        return final_understanding

    # _determine_writer_concept, _select_search_tools 등은 이전과 동일하게 유지
    # ... (이전 응답에서 복사) ...
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
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        # config는 N01에서 초기화되어 state에 존재한다고 가정
        config = state.config or {}  # config 가져오기
        writer_id = config.get('writer_id', 'default_writer')  # run 메서드 내에서 사용할 writer_id

        error_log = list(state.error_log or [])
        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name,
                 'retry_count': state.retry_count or 0}

        logger.info(
            f"Entering node. Input State Summary: {summarize_for_logging(state.model_dump(exclude_none=True), fields_to_show=['original_query', 'query_context', 'initial_context_results', 'current_stage', 'config'])}",
            extra=extra)

        if not state.original_query or not isinstance(state.query_context,
                                                      dict) or state.initial_context_results is None:
            error_msg = "Required inputs are missing for N03."
            logger.error(error_msg, extra=extra)
            error_log.append(
                {"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            return {"error_log": error_log, "current_stage": "ERROR", "error_message": error_msg}

        try:
            # state.query_context에 _extra_log_data 임시 저장 (하위 함수에서 사용 가능하도록)
            # 이는 상태 객체 스키마에 정의되어 있어야 하거나, 일시적 사용 후 제거해야 함
            # 더 나은 방법은 extra_log_data를 명시적으로 전달하는 것 (이미 하고 있음)

            clarified_understanding_dict = await self._clarify_intent_via_sequential_prompting(
                state.original_query,
                state.query_context,
                state.initial_context_results,
                config,  # config 전달
                trace_id,
                extra
            )

            if clarified_understanding_dict.get("clarification_error"):
                error_log.append({
                    "stage": f"{node_name}._clarify_sequential",
                    "error": clarified_understanding_dict["clarification_error"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            updated_query_context = {**state.query_context, **clarified_understanding_dict}

            refined_intent = updated_query_context.get("refined_intent", state.original_query)
            query_type = updated_query_context.get("query_type")

            # _determine_writer_concept에 config 전달
            writer_concept = self._determine_writer_concept(refined_intent, query_type, config)
            logger.info(f"Determined Writer Concept: {writer_concept}", extra=extra)

            key_aspects = updated_query_context.get("key_aspects_to_search", [])
            if not key_aspects or key_aspects == [state.original_query]:  # 또는 refined_intent가 fallback 된 경우
                key_aspects = [refined_intent] if refined_intent and refined_intent != state.original_query else [
                    state.original_query]

            selected_tools = self._select_search_tools(writer_concept, key_aspects)
            logger.info(f"Selected Search Tools: {selected_tools}", extra=extra)

            resolved_ambiguities = updated_query_context.get("resolutions_for_ambiguities", {})
            # _generate_final_search_queries에 config를 직접 전달할 필요는 없으나,
            # query_context_full을 통해 _extra_log_data 등 간접 활용 가능
            updated_query_context['_extra_log_data'] = extra  # 임시로 추가하여 로깅에 사용
            final_queries = self._generate_final_search_queries(
                refined_intent, resolved_ambiguities, key_aspects, writer_concept, updated_query_context, selected_tools
            )
            updated_query_context.pop('_extra_log_data', None)  # 사용 후 제거

            if not final_queries:
                final_queries = [state.original_query]
            logger.info(f"Generated Final Search Queries ({len(final_queries)}): {final_queries}", extra=extra)

            search_params = self._define_search_parameters(writer_concept)
            logger.info(f"Defined Search Parameters: {search_params}", extra=extra)

            search_strategy = {
                "writer_concept": writer_concept,
                "selected_tools": selected_tools,
                "queries": final_queries,
                "parameters": search_params
            }
            next_stage = "n04_execute_search"

            update_dict = {
                "query_context": updated_query_context,
                "search_strategy": search_strategy,
                "current_stage": next_stage,
                "error_log": error_log
            }

            logger.info(
                f"Exiting node. Search Strategy Created. Output Update Summary: {summarize_for_logging(update_dict, fields_to_show=['current_stage', 'search_strategy.queries'])}",
                extra=extra)
            return update_dict

        except Exception as e:
            error_msg = f"Unexpected error in N03 node execution: {e}"
            logger.exception(error_msg, extra=extra)
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            return {
                "error_log": error_log,
                "current_stage": "ERROR",
                "error_message": f"N03 Exception: {error_msg}"
            }