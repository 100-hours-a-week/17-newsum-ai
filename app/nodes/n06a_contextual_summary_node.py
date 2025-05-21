import re
import traceback
from typing import Dict, Any
from datetime import datetime, timezone

from app.utils.logger import get_logger, summarize_for_logging
from app.workflows.state import WorkflowState
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
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        retry_count = state.retry_count or 0

        extra = {
            "trace_id": trace_id,
            "comic_id": comic_id,
            "node_name": node_name,
            "retry_count": retry_count
        }

        logger.info(f"Entering node. Input State Summary: {summarize_for_logging(state, fields_to_show=['report_content', 'query_context'])}", extra=extra)

        error_log = list(state.error_log or [])

        try:
            if not state.report_content:
                raise ValueError("report_content is missing")

            # 입력값 추출
            report_text = self._strip_html(state.report_content)
            refined_intent = state.query_context.get("refined_intent", state.original_query)
            category = state.query_context.get("query_category", "Other")
            audience = state.config.get("target_audience", "general_public")

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
            state.contextual_summary = summary_text

            logger.info(f"Generated contextual summary (truncated): {summary_text[:200]}...", extra=extra)
            return {
                "contextual_summary": summary_text,
                "current_stage": "n07_comic_ideation",
                "error_log": error_log
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
            return {
                "current_stage": "ERROR",
                "error_message": error_msg,
                "error_log": error_log
            }
