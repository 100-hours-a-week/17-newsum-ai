# ai/app/nodes/n02_analyze_query_node.py
from __future__ import annotations

import json  # JSON 파싱은 이제 사용하지 않지만, 혹시 모를 상황 대비 또는 로깅용으로 남겨둘 수 있음
import traceback
from typing import Dict, Any, List, Optional  # Optional 추가
from datetime import datetime, timezone

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.search.Google_Search_tool import GoogleSearchTool

logger = get_logger(__name__)


class N02AnalyzeQueryNode:
    """
    (업그레이드됨 - 다회 호출) 원본 쿼리에 대해 순차적 sLLM 호출을 통해
    핵심 용어/개체, 쿼리 유형, 모호성을 추출하고, 초기 컨텍스트 검색을 수행합니다.
    """

    def __init__(self, llm_service: LLMService, search_tool: GoogleSearchTool):
        self.llm_service = llm_service
        self.search_tool = search_tool

    async def _get_keywords_entities_sllm(
            self, query: str, trace_id: str, extra_log_data: dict
    ) -> List[str]:
        """sLLM을 사용하여 키워드 및 명명된 개체 추출 (쉼표 구분 문자열 요청)"""
        prompt = f"""[System] You are a text analysis expert. Your task is to extract key information from the user's query.

From the 'Original Query' below, identify and list:
- All important keywords and noun phrases
- Specific named entities (such as model names, companies, technologies, products, locations, persons)
- Key technical terms

Strictly extract only terms **explicitly mentioned** in the query. Do not generate new words, synonyms, or paraphrases.
Respond **only** with a comma-separated list. Do not include explanations or formatting. Do not number the items. Do not add categories.
If nothing relevant is found, reply with exactly: `N/A`

Original Query: "{query}"

[Task]
Extracted Keywords/Entities (comma-separated):"""
        logger.debug("Attempting to get keywords/entities from sLLM...", extra=extra_log_data)
        result = await self.llm_service.generate_text(prompt=prompt, max_tokens=150, temperature=0.05)  # 정확도 위해 온도 낮춤

        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM keyword/entity extraction failed: {result.get('error', 'No text generated')}",
                         extra=extra_log_data)
            return []  # 실패 시 빈 리스트

        text_output = result["generated_text"].strip()
        if text_output.upper() == "N/A":
            logger.info("sLLM reported N/A for keywords/entities.", extra=extra_log_data)
            return []

        # 쉼표로 구분된 문자열을 리스트로 변환, 각 항목 공백 제거, 빈 항목 제거
        keywords = [kw.strip() for kw in text_output.split(',') if kw.strip()]
        # 간단한 후처리 (따옴표 제거 등)
        keywords = [kw.removeprefix('"').removesuffix('"').strip() for kw in keywords if kw]
        logger.debug(f"sLLM generated keywords/entities: {keywords}", extra=extra_log_data)
        return keywords

    async def _get_query_type_sllm(
            self, query: str, keywords: List[str], trace_id: str, extra_log_data: dict
    ) -> Optional[str]:
        """sLLM을 사용하여 쿼리 유형 분류 (단일 문자열 요청)"""
        keywords_str = ", ".join(keywords) if keywords else "N/A"
        # 제공할 쿼리 유형 목록 (모델이 이 중에서 선택하도록 유도)
        query_type_options = ["Information Seeking", "Comparison", "How-to", "Problem Solving", "Opinion Seeking",
                              "Ambiguous", "Other"]

        prompt = f"""[System] You are a query classification expert.
Based on the 'Original Query' and its 'Extracted Keywords', classify the user's query into ONE of the following types:
{', '.join(query_type_options)}.
Respond ONLY with the single most appropriate query type from this list. If unsure or it doesn't fit, respond with "Unknown".

Original Query: "{query}"
Extracted Keywords: "{keywords_str}"

[Task]
Predicted Query Type (choose one from the list):"""
        logger.debug("Attempting to get query_type from sLLM...", extra=extra_log_data)
        result = await self.llm_service.generate_text(prompt=prompt, max_tokens=50, temperature=0.1)

        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM query_type classification failed: {result.get('error', 'No text generated')}",
                         extra=extra_log_data)
            return None  # 실패 시 None

        query_type = result["generated_text"].strip()
        query_type = query_type.removeprefix('"').removesuffix('"').strip()

        # 반환된 타입이 옵션에 없으면 "Unknown" 처리 (보다 엄격한 검증)
        if query_type not in query_type_options and query_type != "Unknown":
            logger.warning(f"sLLM returned an unexpected query_type: '{query_type}'. Defaulting to 'Unknown'.",
                           extra=extra_log_data)
            query_type = "Unknown"
        elif not query_type:  # 빈 문자열 응답 시
            query_type = "Unknown"

        logger.debug(f"sLLM generated query_type: {query_type}", extra=extra_log_data)
        return query_type
    
    async def _get_query_category_sllm(
        self, query: str, trace_id: str, extra_log_data: dict
    ) -> Optional[str]:
        """sLLM을 사용하여 분야(category) 분류"""
        category_options = ["IT", "Economy", "Politics", "Leisure", "Science", "Other"]
        prompt = f"""[System] You are a topic classification expert.
    Classify the following user query into ONE of the predefined high-level categories:
    {', '.join(category_options)}.
    Respond ONLY with the category name.

    Original Query: "{query}"

    [Task]
    Predicted Query Category (choose one):"""

        logger.debug("Attempting to get query_category from sLLM...", extra=extra_log_data)
        result = await self.llm_service.generate_text(prompt=prompt, max_tokens=30, temperature=0.1)

        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM query_category classification failed: {result.get('error', 'No text generated')}",
                        extra=extra_log_data)
            return None

        category = result["generated_text"].strip().removeprefix('"').removesuffix('"').strip()
        if category not in category_options:
            logger.warning(f"Unexpected category received: {category}", extra=extra_log_data)
            return "Other"
        return category


    async def _get_detected_ambiguities_sllm(
            self, query: str, keywords: List[str], query_type: Optional[str],
            trace_id: str, extra_log_data: dict
    ) -> List[str]:
        """sLLM을 사용하여 모호한 용어/구문 감지 (쉼표 구분 문자열 요청)"""
        keywords_str = ", ".join(keywords) if keywords else "N/A"
        context_query_type = query_type if query_type and query_type != "Unknown" else "Not yet determined"

        prompt = f"""[System] You are a linguistic analysis expert.
From the 'Original Query' below, identify any terms or phrases that are potentially ambiguous or might require further clarification to fully understand the user's intent. These must be *extracted directly* from the query.
Consider the context of keywords and query type if available.
Respond ONLY with the ambiguous items, separated by commas. If no ambiguities are found, respond with "N/A".

Original Query: "{query}"
Context (Optional):
- Extracted Keywords: "{keywords_str}"
- Predicted Query Type: "{context_query_type}"

[Task]
Detected Ambiguities (comma-separated, from original query):"""
        logger.debug("Attempting to get detected_ambiguities from sLLM...", extra=extra_log_data)
        result = await self.llm_service.generate_text(prompt=prompt, max_tokens=100, temperature=0.1)

        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM ambiguity detection failed: {result.get('error', 'No text generated')}",
                         extra=extra_log_data)
            return []

        text_output = result["generated_text"].strip()
        if text_output.upper() == "N/A":
            logger.info("sLLM reported N/A for ambiguities.", extra=extra_log_data)
            return []

        ambiguities = [amb.strip() for amb in text_output.split(',') if amb.strip()]
        ambiguities = [amb.removeprefix('"').removesuffix('"').strip() for amb in ambiguities if amb]
        logger.debug(f"sLLM generated detected_ambiguities: {ambiguities}", extra=extra_log_data)
        return ambiguities

    async def _analyze_query_sequentially(  # 이름 변경: _extract_query_elements_revised -> _analyze_query_sequentially
            self, query: str, trace_id: str, extra_log_data: dict
    ) -> Dict[str, Any]:
        """순차적 sLLM 호출을 통해 쿼리 분석 수행"""
        analysis_results = {}
        sllm_errors = []

        # 1. 키워드 및 개체 추출
        extracted_keywords = await self._get_keywords_entities_sllm(query, trace_id, extra_log_data)
        if not extracted_keywords:  # sLLM 호출 실패 또는 "N/A" 반환 시
            sllm_errors.append("Keyword/entity extraction failed or returned N/A.")
            analysis_results["extracted_keywords"] = [query]  # Fallback: 원본 쿼리 전체를 키워드로
        else:
            analysis_results["extracted_keywords"] = extracted_keywords

        # 2. 쿼리 유형 분류
        # 이전 단계의 키워드를 컨텍스트로 활용
        query_type = await self._get_query_type_sllm(query, analysis_results["extracted_keywords"], trace_id,
                                                     extra_log_data)
        if not query_type or query_type == "Unknown":
            sllm_errors.append(f"Query type classification failed or returned '{query_type}'.")
            analysis_results["query_type"] = "Unknown"  # Fallback
        else:
            analysis_results["query_type"] = query_type

        # 3. 분야 분류
        query_category = await self._get_query_category_sllm(query, trace_id, extra_log_data)
        if not query_category:
            sllm_errors.append("Query category classification failed or returned N/A.")
            analysis_results["query_category"] = "Unknown"  # Fallback
        else:
            analysis_results["query_category"] = query_category 

        # 4. 모호성 감지
        # 이전 단계의 키워드 및 쿼리 유형을 컨텍스트로 활용
        detected_ambiguities = await self._get_detected_ambiguities_sllm(
            query, analysis_results["extracted_keywords"], analysis_results["query_type"],
            trace_id, extra_log_data
        )
        # 모호성 감지 실패는 오류로 간주하지 않고 빈 리스트로 처리 가능
        analysis_results["detected_ambiguities"] = detected_ambiguities

        if sllm_errors:
            analysis_results["error"] = "; ".join(sllm_errors)
            logger.warning(f"sLLM processing in N02 encountered errors: {analysis_results['error']}",
                           extra=extra_log_data)
        else:
            analysis_results["error"] = None  # 명시적으로 에러 없음 표시

        logger.debug(f"Sequential query analysis complete: {analysis_results}", extra=extra_log_data)
        return analysis_results

    async def _fetch_initial_context_revised(self, original_query: str, extracted_terms: List[str], trace_id: str,
                                             extra_log_data: dict) -> List[Dict[str, Any]]:
        # (이 함수는 이전과 동일하게 유지)
        search_results = []
        search_errors = []
        # 원본 쿼리와, (존재하고 원본 쿼리와 다를 경우) 추출된 첫번째 핵심 용어로 검색 (최대 2개 쿼리)
        queries_to_search = [original_query]
        if extracted_terms and extracted_terms[0] and extracted_terms[0].lower() != original_query.lower():
            queries_to_search.append(extracted_terms[0])
        else:  # 첫번째 키워드가 원본과 같거나 없으면, 두번째 키워드가 있는지 확인 (있고 원본과 다르면)
            if len(extracted_terms) > 1 and extracted_terms[1] and extracted_terms[1].lower() != original_query.lower():
                queries_to_search.append(extracted_terms[1])

        queries_to_search = list(dict.fromkeys(queries_to_search))[:2]  # 중복제거 및 최대 2개

        for query_term in queries_to_search:
            if not query_term or query_term.upper() == "N/A": continue
            try:
                logger.debug(f"Fetching initial context via Google CSE Web Search for query: '{query_term}'",
                             extra=extra_log_data)
                web_res = await self.search_tool.search_web_via_cse(keyword=query_term, max_results=3,
                                                                    trace_id=trace_id)
                if web_res: search_results.extend(web_res)
            except Exception as e:
                msg = f"Google CSE Web Search failed for '{query_term}': {e}"
                logger.warning(msg, extra=extra_log_data, exc_info=False)
                search_errors.append({"tool": "GoogleCSE_WebSearch", "query": query_term, "error": str(e)})

        wiki_query_term = None
        if extracted_terms:  # 가장 첫번째로 추출된 유의미한 용어 사용
            for term in extracted_terms:
                if term and term.upper() != "N/A" and len(term.split()) <= 3:  # 너무 긴 구문은 제외
                    wiki_query_term = term
                    break

        if wiki_query_term:
            try:
                logger.debug(f"Fetching initial context via Wikipedia Search (CSE) for term: '{wiki_query_term}'",
                             extra=extra_log_data)
                wiki_res = await self.search_tool.search_specific_sites_via_cse(keyword=wiki_query_term,
                                                                                sites=["ko.wikipedia.org",
                                                                                       "en.wikipedia.org"],
                                                                                max_results=2, trace_id=trace_id)
                if wiki_res:
                    for res in wiki_res: res['source'] = "Wikipedia_CSE"
                    search_results.extend(wiki_res)
            except Exception as e:
                msg = f"Wikipedia Search (CSE) failed for '{wiki_query_term}': {e}"
                logger.warning(msg, extra=extra_log_data, exc_info=False)
                search_errors.append({"tool": "Wikipedia_CSE", "query": wiki_query_term, "error": str(e)})
        if search_errors: extra_log_data['search_errors'] = search_errors
        logger.debug(f"Total initial context results fetched (raw): {len(search_results)}", extra=extra_log_data)
        return search_results

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        meta, query_sec, config_sec = state.meta, state.query, state.config.config

        trace_id = meta.trace_id
        comic_id = meta.comic_id
        retry_count = meta.retry_count
        error_log = list(meta.error_log)

        extra = {
            "trace_id": trace_id,
            "comic_id": comic_id,
            "node_name": node_name,
            "retry_count": retry_count,
        }
        logger.info(
            "Entering node (section‑aware). Input summary: "
            f"{summarize_for_logging(state.model_dump(exclude_none=True), fields_to_show=['query.original_query'])}",
            extra=extra,
        )

        original_query = query_sec.original_query or ""
        # original_query가 비어 있으면 config에서 가져오거나 meta에서 가져오기 (초기 진입 시)
        if not original_query:
            original_query = config_sec.get("original_query") or getattr(meta, "original_query", "")
            query_sec.original_query = original_query

        try:
            # 1. 쿼리 분석 (키워드, 유형, 카테고리, 모호성)
            query_analysis_results = await self._analyze_query_sequentially(original_query, trace_id, extra)
            # 2. 초기 컨텍스트 검색
            initial_context_results = await self._fetch_initial_context_revised(
                original_query,
                query_analysis_results.get("extracted_keywords", [original_query]),
                trace_id,
                extra
            )

            # 3. 결과를 state_v2 구조에 맞게 Section 객체에 직접 할당
            query_sec.query_context = query_analysis_results
            query_sec.initial_context_results = initial_context_results
            meta.current_stage = "n03_understand_and_plan"
            meta.error_log = error_log
            meta.error_message = query_analysis_results.get("error")

            logger.info(
                "Exiting node. Output Update Summary: "
                f"{summarize_for_logging({'meta': meta.model_dump()}, fields_to_show=['meta.current_stage'])}",
                extra=extra,
            )
            return {
                "query": query_sec.model_dump(),
                "meta": meta.model_dump(),
            }

        except Exception as e:
            err_msg = f"N02 unexpected error: {e}"
            logger.exception(err_msg, extra=extra)
            error_log.append({
                "stage": node_name,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return {
                "meta": {
                    "current_stage": "ERROR",
                    "error_log": error_log,
                    "error_message": err_msg,
                }
            }
