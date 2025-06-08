# ai/app/nodes_v2/n06_issue_analysis_node.py
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.workflows.state_v2 import WorkflowState, IdeaSection
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)
NODE_ORDER = 6  # 이 노드의 순서


class N06IssueAnalysisNode:
    """
    (N06) 입력된 사회 이슈 보고서의 핵심을 분석하여 구조화된 데이터로 출력합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _strip_html_for_analysis(self, html_content: str) -> str:
        # ... (이전 N06IssueAnalysisAndSatireNode의 _strip_html_for_analysis 내용과 동일) ...
        if not html_content: return ""
        text = re.sub(r'<(style|script)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _build_issue_analysis_prompt(self, report_plain_text: str) -> str:
        # ... (이전 N06IssueAnalysisAndSatireNode의 _build_issue_analysis_prompt 내용과 동일) ...
        analysis_schema = {
            "core_problem_definition": "<In English: A concise definition of the core problem...>",
            "key_actors_targets": "<In English: Identify key individuals, groups...>",
            "social_context_background": "<In English: Briefly describe the social, cultural...>",
            "potential_satire_points": ["<In English: Identify specific absurdities... List 2-4 distinct points.>",
                                        "..."],
            "extracted_keywords": ["<English keyword 1>", "<Up to 5 most relevant keywords>"],
            "dominant_public_emotion": "<In English: Infer the likely dominant public emotion(s)...>"
        }
        json_schema_str = json.dumps(analysis_schema, indent=2)
        return f"""
You are an expert socio-political analyst... (이하 동일) ...
# JSON Schema:
{json_schema_str}
"""

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        report_sec = state.report
        idea_sec = state.idea  # IdeaSection에 결과 저장

        work_id = meta_sec.work_id
        node_name = self.__class__.__name__
        extra_log_base = {"work_id": work_id, "node_name": node_name, "node_order": NODE_ORDER}

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info(f"{node_name} 진입: 보고서 심층 분석 시작.", extra=extra_log_base)

        report_html_content = report_sec.report_content
        if not report_html_content:
            logger.error("분석할 보고서 내용(HTML)이 없습니다.", extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"meta": meta_sec.model_dump(), "idea": idea_sec.model_dump(exclude_unset=True)}

        report_plain_text = self._strip_html_for_analysis(report_html_content)
        if not report_plain_text.strip():
            logger.error("HTML 보고서에서 추출된 텍스트 내용이 없습니다.", extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"meta": meta_sec.model_dump(), "idea": idea_sec.model_dump(exclude_unset=True)}

        try:
            analysis_prompt = self._build_issue_analysis_prompt(report_plain_text)
            request_id_analysis = f"{work_id}_{node_name}_IssueAnalysis"

            llm_response = await self.llm_service.generate_text(
                messages=[{"role": "system", "content": "You are an expert socio-political analyst."},
                          {"role": "user", "content": analysis_prompt}],
                request_id=request_id_analysis, temperature=0.1, max_tokens=1500
            )

            cleaned_output = llm_response.get("generated_text", "")
            think_content = llm_response.get("think_content")

            if think_content:
                meta_sec.llm_think_traces.append({
                    "node_name": node_name, "request_id": request_id_analysis,
                    "timestamp": datetime.now(timezone.utc).isoformat(), "log_content": think_content
                })

            if not cleaned_output or llm_response.get("error"):
                raise ValueError(f"이슈 분석 LLM 실패: {llm_response.get('error', 'Empty output')}")

            parsed_analysis = extract_json(cleaned_output)
            if not isinstance(parsed_analysis, dict):
                raise ValueError(f"이슈 분석 결과가 JSON 객체 형식이 아닙니다. 파싱된 내용: {parsed_analysis}")

            idea_sec.structured_issue_analysis = parsed_analysis  # 상태에 저장
            logger.info(
                f"보고서 심층 분석 성공. 풍자 포인트: {summarize_for_logging(str(parsed_analysis.get('potential_satire_points')), 100)}",
                extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"

        except Exception as e:
            logger.error(f"{node_name} 실행 중 오류: {e}", extra=extra_log_base, exc_info=True)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"

        return {"meta": meta_sec.model_dump(), "idea": idea_sec.model_dump(exclude_unset=True)}