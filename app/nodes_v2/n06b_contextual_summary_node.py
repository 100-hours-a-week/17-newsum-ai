# ai/app/nodes_v2/n06b_contextual_summary_node.py
import json
import re
import traceback
from typing import Dict, Any, List
from datetime import datetime, timezone

from app.utils.logger import get_logger, summarize_for_logging
from app.workflows.state_v2 import WorkflowState
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)

class N06BContextualSummaryNode:
    """
    (업그레이드됨) 보고서 내용을 refined_intent, category 및 audience에 맞춰 요약하여
    구조화된 JSON으로 반환하고, 이를 다시 markdown bullet list 형식으로 저장하는 노드.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _strip_html(self, html_content: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', html_content)
        return re.sub(r'\s+', ' ', text).strip()

    def _build_summary_prompt(
        self,
        report_text: str,
        refined_intent: str,
        category: str,
        audience: str
    ) -> str:
        # JSON 스키마 예시
        schema_example = {
            "contextual_summary": [
                "<핵심 인사이트 1>",
                "<핵심 인사이트 2>",
                "<핵심 인사이트 3>"
            ]
        }
        json_schema = json.dumps(schema_example, ensure_ascii=False, indent=2)
        # 프롬프트 구성
        return f"""
You are an expert summarization assistant.
Generate a contextual summary of the report in valid JSON only.
Write in clear Korean (한국어로 작성하세요). Use bullet points converted into a JSON array.

# Context
- Refined Intent: {refined_intent}
- Category: {category}
- Audience: {audience}

# Task
Summarize the following report into 3~5 core insights.
Each insight should be a concise sentence or short paragraph (20~40 words).

# Report Content
{report_text[:8000]}

# Output Format
Return ONLY valid JSON matching the schema and key "contextual_summary" below, NO think tags, NO explanation, NO comments.:
{json_schema}
"""

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta = state.meta
        report_sec = state.report
        query_sec = state.query
        config_sec = state.config
        node_name = self.__class__.__name__
        trace_id = meta.trace_id
        comic_id = meta.comic_id
        extra = {"trace_id": trace_id, "comic_id": comic_id, "node_name": node_name}

        logger.info(
            f"Entering {node_name}. Generating contextual summary.",
            extra=extra
        )
        error_log = list(meta.error_log or [])

        try:
            if not report_sec.report_content:
                raise ValueError("report_content is missing")

            # HTML 제거
            report_text = self._strip_html(report_sec.report_content)
            refined_intent = query_sec.query_context.get("refined_intent", query_sec.original_query)
            category = query_sec.query_context.get("query_category", "Other")
            audience = config_sec.config.get("target_audience", "general_public")

            # 프롬프트 생성 및 LLM 호출
            prompt = self._build_summary_prompt(report_text, refined_intent, category, audience)
            llm_resp = await self.llm_service.generate_text(
                system_prompt_content=prompt,
                prompt=refined_intent,
                max_tokens=800,
                temperature=0.3
            )
            raw_txt = llm_resp.get("generated_text", "")
            logger.debug(f"Raw summary JSON: {raw_txt}", extra=extra)

            # JSON 파싱
            parsed = extract_json(raw_txt)
            summary_items = parsed.get("contextual_summary", [])
            if not isinstance(summary_items, list):
                raise ValueError("Parsed JSON 'contextual_summary' is not a list")

            # Markdown bullet list 형태로 변환
            summary_text = "\n".join([f"- {item}" for item in summary_items])
            report_sec.contextual_summary = summary_text

            # 메타 업데이트
            meta.current_stage = "n07_comic_ideation"
            meta.error_log = error_log
            logger.info(
                f"Contextual summary generated: {summary_items[:3]}...",
                extra=extra
            )

            return {"report": report_sec.model_dump(), "meta": meta.model_dump()}

        except Exception as e:
            error_msg = f"Error in {node_name}: {e}"
            logger.exception(error_msg, extra=extra)
            error_log.append({
                "stage": node_name,
                "error": str(e),
                "detail": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            meta.current_stage = "ERROR"
            meta.error_log = error_log
            meta.error_message = error_msg
            return {"report": report_sec.model_dump(), "meta": meta.model_dump()}
