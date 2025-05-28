# ai/app/nodes_v2/n07_comic_ideation_node.py
import re
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)

# 최대 처리할 보고서 요약 길이
MAX_REPORT_CHARS = 4000
MAX_INSIGHTS = 5
MAX_IDEAS = 3

class N07ComicIdeationNode:
    """
    (업그레이드됨) N06A가 생성한 contextual_summary 또는 보고서 본문을 사용해,
    핵심 인사이트와 만화 아이디어를 JSON으로 생성하는 노드.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _strip_html(self, html_content: str) -> str:
        # 간단한 HTML 태그 제거
        text = re.sub(r'<[^>]+>', ' ', html_content)
        return re.sub(r'\s+', ' ', text).strip()

    async def _get_report_text(self, report_html: str, contextual: Optional[str], trace_id: str, extra: Dict) -> str:
        """
        contextual_summary가 있으면 사용, 그렇지 않으면 HTML을 텍스트로 변환하여 반환.
        적절히 길이를 제한.
        """
        if contextual:
            logger.info("Using contextual_summary from N06A.", extra=extra)
            text = contextual.strip()
        else:
            text = self._strip_html(report_html)
        # 길이 제한
        if len(text) > MAX_REPORT_CHARS:
            text = text[:MAX_REPORT_CHARS] + '...'
        return text

    async def _extract_key_insights(self,
            summary: str, refined_intent: str, category: str, audience: str,
            trace_id: str, extra: Dict[str, Any]
    ) -> List[str]:
        """
        핵심 인사이트를 JSON 배열로 요청하고, 파싱하여 Python 리스트로 반환.
        """
        system_prompt = (
            "You are an analytical assistant. Extract the most critical insights "
            "directly addressing the user's refined question."
        )
        # 프롬프트에 JSON 스키마 명시
        schema = {"insights": ["<string>"]}
        prompt = f"""
System Prompt:
{system_prompt}

Generate 3 to {MAX_INSIGHTS} key insights in JSON only.
Write in clear Korean (한국어로 작성하세요).

Return ONLY valid JSON matching the schema below, NO think tags, NO explanation, NO comments.

Output format:
{schema}

User Context:
- Refined Question: {refined_intent}
- Category: {category}
- Audience: {audience}

Report Summary:
{summary}
"""
        # messages 리스트 생성
        messages_for_llm = [
            {"role": "system", "content": prompt},  # prompt를 system 역할로
            {"role": "user", "content": refined_intent}  # original_query를 user 역할로
        ]

        # 수정된 방식으로 LLMService 호출
        response = await self.llm_service.generate_text(
            messages=messages_for_llm,
            temperature=0.3,
            max_tokens=1000
        )
        raw = response.get("generated_text", "")
        logger.debug(f"Raw insights JSON: {raw}", extra=extra)
        try:
            parsed = extract_json(raw)
            insights = parsed.get("insights", [])
            if isinstance(insights, list):
                return insights[:MAX_INSIGHTS]
        except Exception as e:
            logger.error(f"Failed to parse insights JSON: {e}", extra=extra)
        # fallback: 빈 리스트 반환
        return []

    async def _generate_comic_ideas(
            self,
            insights: List[str], refined_intent: str,
            original_query: str, category: str,
            audience: str, trace_id: str, extra: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        인사이트를 바탕으로 JSON 형태의 만화 아이디어 리스트를 생성.
        """
        system_prompt = (
            "You are a creative comic-idea generator. "
            "Generate ideas formatted as valid JSON only."
        )
        # JSON 스키마 예시
        schema = {"ideas": [{
            "title": "<string>",
            "logline": "<string>",
            "genre": "<string>",
            "target_emotion": "<string>",
            "key_elements": ["<string>"]
        }]}
        # 인사이트 포맷 문자열
        insights_text = '\n'.join([f"- {i}" for i in insights])
        prompt = f"""
System Prompt:
{system_prompt}

Generate exactly {MAX_IDEAS} unique comic ideas following this JSON schema only:
{schema}

User Context:
- Original Query: {original_query}
- Refined Question: {refined_intent}
- Category: {category}
- Audience: {audience}

Key Insights:
{insights_text}
"""
        # messages 리스트 생성
        messages_for_llm = [
            {"role": "system", "content": prompt},  # prompt를 system 역할로
            {"role": "user", "content": refined_intent}  # original_query를 user 역할로
        ]

        # 수정된 방식으로 LLMService 호출
        response = await self.llm_service.generate_text(
            messages=messages_for_llm,
            temperature=0.7,
            max_tokens=1500
        )
        raw = response.get("generated_text", "")
        logger.debug(f"Raw ideas JSON: {raw}", extra=extra)
        try:
            parsed = extract_json(raw)
            ideas = parsed.get("ideas", [])
            if isinstance(ideas, list):
                return ideas[:MAX_IDEAS]
        except Exception as e:
            logger.error(f"Failed to parse ideas JSON: {e}", extra=extra)
        # fallback: 빈 리스트 반환
        return []

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta = state.meta
        idea_sec = state.idea
        report_sec = state.report
        query_sec = state.query
        config = state.config.config or {}
        node_name = self.__class__.__name__
        extra = {'trace_id': meta.trace_id, 'comic_id': meta.comic_id, 'node_name': node_name}
        logger.info(f"Entering {node_name}.", extra=extra)
        error_log = list(meta.error_log or [])

        # 1) 보고서 텍스트 준비
        report_text = await self._get_report_text(
            report_sec.report_content,
            getattr(report_sec, 'contextual_summary', None),
            meta.trace_id, extra
        )
        if not report_text:
            error_msg = "Report text is empty."
            logger.error(error_msg, extra=extra)
            error_log.append({'stage': node_name, 'error': error_msg, 'timestamp': datetime.now(timezone.utc).isoformat()})
            meta.current_stage = 'ERROR'
            meta.error_log = error_log
            return {'idea': idea_sec.model_dump(), 'meta': meta.model_dump()}

        # 2) 핵심 인사이트 추출
        refined_intent = query_sec.query_context.get('refined_intent', query_sec.original_query)
        category = query_sec.query_context.get('query_category', 'Other')
        audience = config.get('target_audience', 'general_public')
        insights = await self._extract_key_insights(
            report_text, refined_intent, category, audience,
            meta.trace_id, extra
        )

        # 3) 만화 아이디어 생성
        ideas = await self._generate_comic_ideas(
            insights, refined_intent,
            query_sec.original_query, category,
            audience, meta.trace_id, extra
        )

        idea_sec.comic_ideas = ideas
        meta.current_stage = 'n08_scenario_generation'
        meta.error_log = error_log
        logger.info(f"Generated {len(ideas)} comic ideas.", extra=extra)
        return {'idea': idea_sec.model_dump(), 'meta': meta.model_dump()}
