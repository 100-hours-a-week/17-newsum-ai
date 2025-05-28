# ai/app/nodes_v2/n02_analyze_query_node.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)

class N02AnalyzeQueryNode:
    """
    (업그레이드됨) 단일 LLM 호출을 통해 쿼리 분석 결과를 JSON으로 반환받고,
    think_parser_tool로 파싱하여 state.query.query_context에 저장합니다.
    또한 기존 얕은 검색(_fetch_initial_context_revised)을 통합하여
    state.query.initial_context_results에 할당합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_analysis_prompt(self, query: str, context: str) -> str:
        schema_example = {
            "extracted_keywords": ["<키워드1>", "<키워드2>"],
            "query_type": "<Information Seeking|Comparison|How-to|Problem Solving|Opinion Seeking|Ambiguous|Other>",
            "query_category": "<IT|Economy|Politics|Other>",
            "detected_ambiguities": ["<모호어1>", "<모호어2>"],
            "error": None
        }
        # Leisure, Science 제거 안하면 이렇게
        #"query_category": "<IT|Economy|Politics|Leisure|Science|Other>",
        json_example = json.dumps(schema_example, ensure_ascii=False, indent=2)
        return f"""
You are an expert query analysis agent.
Use the context below to analyze the user's original query.

# Context Snippets:
{context}

Return ONLY valid JSON matching the schema below, NO think tags, NO explanation, NO comments.

# JSON Schema Example:
{json_example}

Original Query: "{query}"
"""

    async def _fetch_initial_context_revised(
        self, original_query: str, extracted_terms: List[str], trace_id: str, extra_log: dict
    ) -> List[Dict[str, Any]]:
        # 기존 n02의 얕은 검색 로직 유지
        search_results: List[Dict[str, Any]] = []
        search_errors: List[Dict[str, Any]] = []
        queries = [original_query]
        if extracted_terms:
            for term in extracted_terms[:2]:
                if term and term.lower() != original_query.lower():
                    queries.append(term)
        for q in dict.fromkeys(queries):
            try:
                logger.debug(f"Searching web via CSE for: '{q}'", extra=extra_log)
                res = await self.search_tool.search_web_via_cse(keyword=q, max_results=3, trace_id=trace_id)
                if res:
                    search_results.extend(res)
            except Exception as e:
                search_errors.append({"tool": "GoogleCSE_WebSearch", "query": q, "error": str(e)})
        if search_errors:
            extra_log['search_errors'] = search_errors
        logger.debug(f"Fetched {len(search_results)} initial context items", extra=extra_log)
        return search_results

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node = self.__class__.__name__
        meta, query_sec, config = state.meta, state.query, state.config.config
        trace_id = meta.trace_id
        comic_id = meta.comic_id
        extra = {"trace_id": trace_id, "comic_id": comic_id, "node": node}

        original_query = query_sec.original_query or config.get("original_query", "")
        query_sec.original_query = original_query
        logger.info("Entering N02AnalyzeQueryNode: %s", summarize_for_logging(original_query), extra=extra)

        try:
            # 1) 얕은 검색으로 컨텍스트 확보
            context = await self._fetch_initial_context_revised(
                original_query, [], trace_id, extra
            )
            snippets = "\n".join([item.get("snippet", item.get("title", "")) for item in context])

            # 2) 프롬프트 생성 및 LLM 호출
            prompt = self._build_analysis_prompt(original_query, snippets)

            # messages 리스트 생성
            messages_for_llm = [
                {"role": "system", "content": prompt},  # prompt를 system 역할로
                {"role": "user", "content": original_query}  # original_query를 user 역할로
            ]

            # 수정된 방식으로 LLMService 호출
            llm_resp = await self.llm_service.generate_text(
                messages=messages_for_llm,
                temperature=0.3,
                max_tokens=800,
            )
            raw_txt = llm_resp.get("generated_text", "")
            logger.debug("Raw LLM response: %s", raw_txt, extra=extra)

            # 3) JSON 파싱
            try:
                parsed = extract_json(raw_txt)
                logger.debug("Parsed N02 JSON: %s", json.dumps(parsed, ensure_ascii=False), extra=extra)
            except Exception as e:
                logger.error("JSON parsing failed in N03: %s", str(e), extra=extra)
                raise
            

            # 4) state 업데이트 (기존 n02와 동일한 구조 유지)
            query_sec.query_context = parsed
            # 5) 초기 컨텍스트 재검색 (키워드 기반)
            extracted = parsed.get("extracted_keywords", [])
            initial_context = await self._fetch_initial_context_revised(
                original_query, extracted, trace_id, extra
            )
            query_sec.initial_context_results = initial_context

            # 메타 업데이트
            meta.current_stage = "n03_understand_and_plan"
            meta.error_message = parsed.get("error")
            meta.error_log = list(meta.error_log)

            logger.info(
                "Exiting N02AnalyzeQueryNode: stage=%s", meta.current_stage,
                extra=extra
            )
            return {"query": query_sec.model_dump(), "meta": meta.model_dump()}

        except Exception as e:
            err = f"N02AnalyzeQueryNode failed: {e}"
            logger.exception(err, extra=extra)
            meta.error_log.append({
                "stage": node,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return {"meta": {"current_stage": "ERROR", "error_log": meta.error_log, "error_message": err}}
