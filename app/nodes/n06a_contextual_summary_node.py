import re
import traceback
from typing import Dict, Any
from datetime import datetime, timezone

from app.utils.logger import get_logger, summarize_for_logging
from app.workflows.state_v2 import WorkflowState
from app.services.llm_service import LLMService

logger = get_logger(__name__)


class N06AContextualSummaryNode:
    """
    보고서 내용을 refined_intent와 category 기준으로 요약하여 후속 노드(n07, n08 등)에서 사용할 수 있도록
    핵심 요약(contextual_summary)을 생성합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _strip_html(self, html: str) -> str:
        text = re.sub(r'<[^>]+>', ' ', html)
        return re.sub(r'\s+', ' ', text).strip()

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        """
        보고서 내용을 요약하여 report Section에 저장하고, stage/error 등은 meta Section에 저장합니다.
        반환값은 state_v2 구조에 맞게 {"report": ..., "meta": ...} 형태로 반환합니다.
        """
        meta = state.meta
        report_sec = state.report
        query_sec = state.query
        config_sec = state.config
        node_name = self.__class__.__name__
        trace_id = meta.trace_id
        comic_id = meta.comic_id
        retry_count = meta.retry_count or 0
        extra = {
            "trace_id": trace_id,
            "comic_id": comic_id,
            "node_name": node_name,
            "retry_count": retry_count
        }
        logger.info(f"Entering node. Input State Summary: {summarize_for_logging(state, fields_to_show=['report.report_content', 'query.query_context'])}", extra=extra)
        error_log = list(meta.error_log or [])
        try:
            if not report_sec.report_content:
                raise ValueError("report_content is missing")
            # 입력값 추출
            report_text = self._strip_html(report_sec.report_content)
            refined_intent = query_sec.query_context.get("refined_intent", query_sec.original_query)
            category = query_sec.query_context.get("query_category", "Other")
            audience = config_sec.config.get("target_audience", "general_public")
            # 프롬프트 구성
            system_msg = "You are a helpful summarization assistant."
            user_msg = f"""
Summarize the following report with the following rules:
- Prioritize information related to: \"{refined_intent}\"
- Focus on category: {category}, audience: {audience}
- Output 3~5 core insights in concise bullet points or short paragraphs

Report:
{report_text[:8000]}
            """
            result = await self.llm_service.generate_text(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            summary_text = result.get("generated_text", "").strip()
            report_sec.contextual_summary = summary_text
            meta.current_stage = "n07_comic_ideation"
            meta.error_log = error_log
            logger.info(f"Generated contextual summary (truncated): {summary_text[:200]}...", extra=extra)
            return {
                "report": report_sec.model_dump(),
                "meta": meta.model_dump(),
            }
        except Exception as e:
            error_msg = f"Error in N06AContextualSummaryNode: {e}"
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
            return {
                "report": report_sec.model_dump(),
                "meta": meta.model_dump(),
            }
