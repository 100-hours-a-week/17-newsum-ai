# ai/app/nodes_v2/n02_analyze_query_node.py
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.services.postgresql_service import PostgreSQLService

logger = get_logger(__name__)

NODE_ORDER = 2
MAX_SEED_DOMAINS_TO_USE = 5
MAX_KEYWORDS_FOR_LLM = 5


class N02AnalyzeQueryNode:
    def __init__(
        self,
        llm_service: LLMService,
        search_tool: GoogleSearchTool,
        postgresql_service: PostgreSQLService
    ):
        self.llm_service = llm_service
        self.search_tool = search_tool
        self.postgresql_service = postgresql_service

        # ===== DB 접근 불가 시 사용할 최소한의 도메인 목록 =====
        self.ULTIMATE_FALLBACK_DOMAINS = ["naver.com", "mbc.co.kr", "chosun.com"]

    def _build_analysis_prompt(self, query: str, context_snippets: str) -> str:
        """
        변경: ‘만화 생성 목적’을 포함하여 JSON 스키마에 이슈 요약·풍자 대상·톤 제안을 추가
        """
        current_year = datetime.now().year
        is_future_query = self._detect_future_event(query)
        temporal_warning = ""
        if is_future_query:
            temporal_warning = (
                f"⚠️ IMPORTANT: Current year is {current_year}. "
                f"If this query refers to future events (like {current_year + 1} or later), "
                f"please lower confidence and flag accordingly.\n"
            )

        categories = [
            "mainstream_news",
            "broadcast_news",
            "economy",
            "it_tech",
            "science",
            "academic",
            "culture_entertainment",
            "extra_misc",
            "community",
            "regional_news",
            "magazine",
            "finance_business",
            "sports",
            "entertainment",
        ]
        category_options_str = "|".join(categories)

        schema_example = {
            "extracted_keywords": [
                f"<keyword1_english_only>",
                "<keyword2_english_only>",
                f"<up_to_{MAX_KEYWORDS_FOR_LLM}_keywords_english_only>"
            ],
            "query_type": "<Information Seeking|Comparison|How-to|Problem Solving|Opinion Seeking|Ambiguous|Other>",
            "query_category": f"<{category_options_str}>",
            "detected_ambiguities": [
                "<ambiguity1_description_english>",
                "<ambiguity2_description_english>"
            ],
            "issue_summary_for_comic": "<A brief English summary of the core issue to guide web search>",
            "satire_target": [
                "<who_or_what_to_satirize (e.g., government, corporation, public opinion)>"
            ],
            "tone_suggestion": "<English suggestion on comic tone (e.g., 'sharp_sarcasm', 'playful_irony')>",
            "analysis_reasoning_korean": {
                "overall_assessment_korean": "<한글 설명: 사용자의 주요 의도 및 요청에 대한 전반적인 평가>",
                "category_selection_rationale_korean": "<한글 설명: 왜 이 카테고리를 선택했는지에 대한 상세 근거>",
                "keyword_extraction_rationale_korean": f"<한글 설명: {MAX_KEYWORDS_FOR_LLM}개의 주요 영어 키워드를 어떻게, 왜 추출했는지 설명>",
                "ambiguity_analysis_korean": "<한글 설명: 감지된 모호성과 초기 해결 제안 또는 추가 정보 필요성>",
                "temporal_context_korean": "<한글 설명: 쿼리의 시간적 맥락 분석 및 검색 전략 권장사항>",
                "issue_summary_rationale_korean": "<한글 설명: 왜 이 짧은 요약이 검색에 도움이 되는지 설명>",
                "satire_target_rationale_korean": "<한글 설명: 어떤 대상을 풍자해야 하는지에 대한 근거>",
                "tone_suggestion_rationale_korean": "<한글 설명: 톤 제안이 검색/분석에 왜 적합한지>"
            }
        }
        json_example_str = json.dumps(schema_example, ensure_ascii=False, indent=2)

        return (
            f"You are an expert query analysis agent specialized in satirical 4-panel comic creation based on social or 뉴스 이슈.\n"
            f"{temporal_warning}"
            "Your task:\n"
            f"1. Extract up to {MAX_KEYWORDS_FOR_LLM} relevant English keywords (consider Korean context).\n"
            "2. Determine query type (e.g., Comparison, How-to, Opinion Seeking, etc.).\n"
            f"3. Assign a query category from: {category_options_str}.\n"
            "4. Identify any ambiguities (including temporal issues).\n"
            "5. Summarize the core issue for a 4-panel satirical comic (issue_summary_for_comic).\n"
            "6. Suggest who or what to satirize (satire_target).\n"
            "7. Recommend a tone for the comic (tone_suggestion).\n"
            "8. Provide detailed Korean reasoning for each decision.\n\n"
            f"# Context Snippets:\n{context_snippets}\n\n"
            f"# User's Original Query:\n\"{query}\"\n\n"
            "Return ONLY a single, valid JSON object that strictly adheres to the schema below:\n"
            f"{json_example_str}"
        )

    async def _fetch_initial_context_revised(
        self,
        original_query: str,
        extracted_terms: List[str],
        work_id: str,
        extra_log: Dict[str, Any],
        search_attempt_label: str,
        target_domains: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        GoogleSearchTool을 통해 초기 컨텍스트 스니펫을 수집합니다.
        - target_domains가 주어지면 해당 도메인 전부(제한 없이) 우선 검색
        - 주어지지 않으면 일반 웹 검색
        """
        results: List[Dict[str, Any]] = []
        max_results = 3

        if target_domains:
            # 모든 target_domains 리스트를 그대로 넘겨서, 도메인 수에 제한 없이 검색
            for term in extracted_terms or [original_query]:
                try:
                    fetched = await self.search_tool.search_specific_sites_via_cse(
                        keyword=term,
                        sites=target_domains,           # **변경**: slicing 제거
                        max_results=max_results,
                        trace_id=work_id
                    )
                    if fetched:
                        results.extend(fetched)
                except Exception as e:
                    logger.error(
                        f"[{search_attempt_label}] seed domain 검색 중 오류: {e}",
                        extra=extra_log, exc_info=True
                    )
        else:
            # 일반 웹 검색 (도메인 제한 없음)
            try:
                fetched = await self.search_tool.search_web_via_cse(
                    keyword=original_query,
                    max_results=max_results,
                    trace_id=work_id
                )
                if fetched:
                    results.extend(fetched)
            except Exception as e:
                logger.error(
                    f"[{search_attempt_label}] 웹 검색 중 오류: {e}",
                    extra=extra_log, exc_info=True
                )

        # 결과 중복 URL 제거
        unique = {}
        for item in results:
            url = item.get("url", "")
            if url and url not in unique:
                unique[url] = item
            elif not url:
                unique[f"no_url_{len(unique)}"] = item

        return list(unique.values())

    async def _get_trusted_domains(
        self,
        category: Optional[str],
        keywords: List[str],
        work_id: str,
        extra_log: Dict[str, Any]
    ) -> List[str]:
        """
        카테고리(category)만으로 ai_test_seed_domains 테이블에서
        조건에 맞는 모든 도메인을 조회합니다.
        (tags 조건은 제거되었습니다)
        """
        if not self.postgresql_service:
            logger.error(
                "[TrustedDomains] PostgreSQLService가 없습니다.",
                extra={"work_id": work_id}
            )
            return []

        query_parts: List[str] = []
        params: List[Any] = []

        # 1) 카테고리 필터
        if category:
            query_parts.append(f"category = ${len(params) + 1}")
            params.append(category)

        # 2) WHERE 절 구성 (category가 없으면 TRUE → 모든 도메인 조회)
        where_clause = " AND ".join(query_parts) if query_parts else "TRUE"

        # 3) LIMIT 절 제거: 조건에 맞는 모든 도메인을 반환
        sql = f"""
            SELECT domain
            FROM ai_test_seed_domains
            WHERE {where_clause};
        """
        params_tuple = tuple(params)  # 예: ('it_tech',) 또는 빈 튜플

        try:
            # 실제 날아가는 SQL과 파라미터를 로그로 찍어 확인
            logger.debug(
                f"[TrustedDomains] Executing SQL: {sql} | Params: {params_tuple}",
                extra=extra_log
            )
            # 파라미터를 언패킹해서 전달 ($1 등 플레이스홀더에 맞게)
            rows = await self.postgresql_service.fetch_all(sql, *params_tuple)
            # asyncpg.Record 또는 dict 형태로 반환될 수 있으므로, 컬럼명 'domain'으로 값 조회
            domains = []
            for row in rows:
                # Record 객체면 row["domain"], dict이면 row.get("domain")
                try:
                    dom = row["domain"]
                except (KeyError, TypeError):
                    dom = row.get("domain") if isinstance(row, dict) else None
                if dom:
                    domains.append(dom)
            if domains:
                logger.info(
                    f"[TrustedDomains] 조회된 도메인 개수: {len(domains)}",
                    extra=extra_log
                )
            else:
                logger.info(
                    "[TrustedDomains] 조건에 맞는 도메인이 없습니다.",
                    extra=extra_log
                )
            return domains

        except Exception as e:
            logger.error(
                f"[TrustedDomains] DB 조회 중 오류: {e}",
                extra={**extra_log, "sql": sql, "params": params_tuple},
                exc_info=True
            )
            return []

    def _detect_future_event(self, query: str) -> bool:
        """
        쿼리 내에 향후 이벤트를 나타내는 패턴이 있으면 True 반환.
        """
        current_year = datetime.now().year
        future_patterns = [r'202[5-9]년', r'202[5-9]', r'내년', r'다음.*년', r'향후', r'예정']
        for pattern in future_patterns:
            if re.search(pattern, query):
                return True
        year_matches = re.findall(r'(20\d{2})', query)
        for year_str in year_matches:
            if int(year_str) > current_year:
                return True
        return False

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        query_sec = state.query

        original_query = query_sec.original_query or ""
        work_id = meta_sec.work_id or ""
        extra_log_base = {
            "work_id": work_id,
            "node_name": self.__class__.__name__,
            "node_order": NODE_ORDER
        }

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info("N02 노드 진입: 쿼리 분석 및 초기 컨텍스트 수집 시작.", extra=extra_log_base)

        # ※ original_query 누락 시 예외
        if not original_query:
            error_msg = "original_query가 누락되었습니다."
            logger.error(error_msg, extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            meta_sec.error_message = error_msg
            return {
                "query": query_sec.model_dump(),
                "meta": meta_sec.model_dump()
            }

        # 1단계: 초기 컨텍스트 수집 (일반 웹)
        try:
            logger.info(
                "1단계: LLM 분석용 초기 컨텍스트 수집 (일반 웹 검색)", extra=extra_log_base
            )
            initial_stage_context = await self._fetch_initial_context_revised(
                original_query=original_query,
                extracted_terms=[],
                work_id=work_id,
                extra_log={**extra_log_base, "sub_process": "fetch_for_llm_prompt_context"},
                search_attempt_label="LLM_Context_Search"
            )
            context_snippets_str = "\n".join(
                f"- Title: {item.get('title', 'N/A')}\n  Snippet: {item.get('snippet', 'N/A')}"
                for item in initial_stage_context
            ).strip() or "No initial context snippets found."
            query_sec.initial_context_results = initial_stage_context
            logger.info(
                f"LLM 분석용 초기 컨텍스트 스니펫 {len(initial_stage_context)}개 준비 완료.",
                extra=extra_log_base
            )
        except Exception as e:
            error_msg = f"초기 컨텍스트 수집 중 오류: {e}"
            logger.error(error_msg, extra=extra_log_base, exc_info=True)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            meta_sec.error_message = error_msg
            return {
                "query": query_sec.model_dump(),
                "meta": meta_sec.model_dump()
            }

        # 2단계: LLM 쿼리 분석 요청
        try:
            logger.info("2단계: LLM 쿼리 분석 시작", extra=extra_log_base)
            analysis_prompt = self._build_analysis_prompt(
                original_query, context_snippets_str
            )
            request_id_for_llm = f"{work_id}_N02_QueryAnalysis"
            llm_messages = [
                {"role": "system", "content": "You are a sophisticated query analysis expert agent."},
                {"role": "user", "content": analysis_prompt}
            ]
            llm_response = await self.llm_service.generate_text(
                messages=llm_messages,
                request_id=request_id_for_llm,
                temperature=0.1,
                max_tokens=2048
            )

            cleaned_output = llm_response.get("generated_text", "")
            think_content = llm_response.get("think_content")
            if think_content:
                meta_sec.llm_think_traces.append({
                    "node_name": self.__class__.__name__,
                    "request_id": request_id_for_llm,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "log_content": think_content
                })
                logger.debug(
                    f"LLM <think> 저장됨 (Request ID: {request_id_for_llm})",
                    extra=extra_log_base
                )

            # 2-1) JSON 파싱 결과 유효성 검사
            parsed = extract_json(cleaned_output)
            if not isinstance(parsed, dict):
                logger.error(
                    f"[LLM Parsing Error] extract_json 결과가 dict가 아닙니다: {parsed}",
                    extra=extra_log_base
                )
                raise ValueError("LLM 응답에서 유효한 JSON 객체를 추출하지 못했습니다.")

            # 2-2) 필수 필드 누락 검사
            required_fields = [
                "extracted_keywords", "query_type", "query_category",
                "issue_summary_for_comic", "satire_target", "tone_suggestion"
            ]
            missing = [fld for fld in required_fields if fld not in parsed]
            if missing:
                logger.error(
                    f"[LLM Missing Fields] 다음 필드가 누락되었습니다: {missing}",
                    extra=extra_log_base
                )
                raise ValueError(f"LLM 응답에 다음 필드가 없습니다: {missing}")

            logger.info(
                f"LLM 쿼리 분석 완료. 카테고리: '{parsed.get('query_category')}', "
                f"키워드: {parsed.get('extracted_keywords')}, 이슈 요약: '{parsed.get('issue_summary_for_comic')}'",
                extra=extra_log_base
            )
        except Exception as e:
            error_msg = f"LLM 분석 중 오류 또는 검증 실패: {e}"
            logger.error(error_msg, extra=extra_log_base, exc_info=True)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            meta_sec.error_message = error_msg
            return {
                "query": query_sec.model_dump(),
                "meta": meta_sec.model_dump()
            }

        # 3단계: 결과를 query_context에 저장 및 한글 근거 갱신
        try:
            query_sec.query_context = parsed

            reasoning = parsed.get("analysis_reasoning_korean", {})
            if isinstance(reasoning, dict):
                reasoning_texts = [
                    f"종합 평가: {reasoning.get('overall_assessment_korean', '제공되지 않음')}",
                    f"카테고리 선택 근거: {reasoning.get('category_selection_rationale_korean', '제공되지 않음')}",
                    f"키워드 추출 근거: {reasoning.get('keyword_extraction_rationale_korean', '제공되지 않음')}",
                    f"모호성 분석: {reasoning.get('ambiguity_analysis_korean', '제공되지 않음')}",
                    f"시간적 맥락: {reasoning.get('temporal_context_korean', '제공되지 않음')}",
                    f"이슈 요약 근거: {reasoning.get('issue_summary_rationale_korean', '제공되지 않음')}",
                    f"풍자 대상 근거: {reasoning.get('satire_target_rationale_korean', '제공되지 않음')}",
                    f"톤 제안 근거: {reasoning.get('tone_suggestion_rationale_korean', '제공되지 않음')}"
                ]
                query_sec.llm_analysis_details_korean = "\n\n".join(reasoning_texts)
            else:
                query_sec.llm_analysis_details_korean = (
                    "LLM으로부터 상세 분석 근거(한글)가 수신되지 않거나 형식이 올바르지 않습니다."
                )

            logger.info("LLM 분석 상세(한글) 저장됨.", extra=extra_log_base)
        except Exception as e:
            error_msg = f"query_context 저장 또는 한글 근거 갱신 중 오류: {e}"
            logger.error(error_msg, extra=extra_log_base, exc_info=True)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            meta_sec.error_message = error_msg
            return {
                "query": query_sec.model_dump(),
                "meta": meta_sec.model_dump()
            }

        # 4단계: 신뢰하는 시드 도메인 조회
        try:
            logger.info("3단계: 신뢰 시드 도메인 조회 시작", extra=extra_log_base)
            category = parsed.get("query_category")
            keywords = parsed.get("extracted_keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            trusted_domains = []
            if category or keywords:
                trusted_domains = await self._get_trusted_domains(
                    category=category,
                    keywords=keywords,
                    work_id=work_id,
                    extra_log={**extra_log_base, "sub_process": "fetch_trusted_domains"}
                )
        except Exception as e:
            logger.error(
                f"신뢰 도메인 조회 중 오류: {e}", extra=extra_log_base, exc_info=True
            )
            trusted_domains = []

        # 5단계: 최종 초기 컨텍스트 수집 (신뢰 도메인 우선)
        try:
            logger.info("4단계: 최종 초기 컨텍스트 수집 시작 (신뢰 도메인 우선)", extra=extra_log_base)
            effective_domains = trusted_domains if trusted_domains else self.ULTIMATE_FALLBACK_DOMAINS
            final_context = await self._fetch_initial_context_revised(
                original_query=original_query,
                extracted_terms=keywords,
                work_id=work_id,
                extra_log={**extra_log_base, "sub_process": "final_context_fetch"},
                search_attempt_label="Final_Context_Search",
                target_domains=effective_domains
            )

            # 1단계와 5단계 결과 합치기
            merged_context = []
            if initial_stage_context:
                merged_context.extend(initial_stage_context)
            if final_context:
                merged_context.extend(final_context)

            # URL 기준으로 중복 제거
            unique_by_url: Dict[str, Dict[str, Any]] = {}
            for item in merged_context:
                url = item.get("url", f"nokey_{len(unique_by_url)}")
                if url not in unique_by_url:
                    unique_by_url[url] = item
            query_sec.initial_context_results = list(unique_by_url.values())

            logger.info(f"초기 컨텍스트(합산) 총 {len(query_sec.initial_context_results)}개 확보.", extra=extra_log_base)
        except Exception as e:
            error_msg = f"최종 초기 컨텍스트 수집 중 오류: {e}"
            logger.error(error_msg, extra=extra_log_base, exc_info=True)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            meta_sec.error_message = error_msg
            return {
                "query": query_sec.model_dump(),
                "meta": meta_sec.model_dump()
            }

        meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"
        logger.info("N02 노드 종료: 쿼리 분석 및 초기 컨텍스트 수집 완료.", extra=extra_log_base)
        return {
            "query": query_sec.model_dump(),
            "meta": meta_sec.model_dump()
        }
