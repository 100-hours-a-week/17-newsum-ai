# ai/app/nodes/n07_comic_ideation_node.py
import re, traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService

logger = get_logger(__name__)

MAX_REPORT_CONTENT_CHARS_FOR_IDEATION = 4_000
MAX_IDEAS_TO_GENERATE                = 3
MAX_RETRY_ATTEMPTS                   = 2


class N07ComicIdeationNode:
    """
    N06A가 만든 contextual_summary(있을 경우) + 보고서 원문을 이용해
    refined_intent에 부합하는 핵심 Insight를 뽑고 만화 아이디어(<idea> XML) 생성.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    # ──────────────────── 0. 유틸 ────────────────────
    def _is_valid_idea_xml(self, xml: str) -> bool:
        if not isinstance(xml, str):
            return False
        required_tags = [
            "title", "logline", "genre",
            "target_emotion", "key_elements_from_report"
        ]
        for tag in required_tags:
            if re.search(fr"<{tag}>.*?</{tag}>", xml, flags=re.I | re.S) is None:
                return False
        return True

    async def _summarize_report_with_sllm(
        self, text: str, max_len: int, trace_id: str, extra: dict, attempt: int = 1
    ) -> str:
        approx_words = max_len // 5
        system_msg   = "You are a helpful assistant that summarizes texts concisely."
        user_msg = (
            f"Summarize the key findings below (~{approx_words} words, ≤{max_len} characters):\n\n{text}"
        )
        if attempt > 1:
            user_msg = f"[Retry {attempt}/{MAX_RETRY_ATTEMPTS+1}] " + user_msg

        res = await self.llm_service.generate_text(
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user",   "content": user_msg}],
            max_tokens=1000,
            temperature=0.3,
        )
        out = res.get("generated_text", "").strip()
        if out:
            return out
        if attempt <= MAX_RETRY_ATTEMPTS:
            return await self._summarize_report_with_sllm(text, max_len, trace_id, extra, attempt + 1)
        logger.warning("Summarizer fallback (truncate).", extra=extra)
        return text[:max_len] + "..."

    async def _preprocess_report_content(
        self, html: str, trace_id: str, extra: dict, contextual: Optional[str] = None
    ) -> str:
        if contextual:
            logger.info("Using contextual_summary from N06A.", extra=extra)
            return contextual.strip()

        # HTML → text
        txt = re.sub(r"<style[^>]*?>.*?</style>", "", html, flags=re.S | re.I)
        txt = re.sub(r"<script[^>]*?>.*?</script>", "", txt,  flags=re.S | re.I)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()

        if len(txt) > MAX_REPORT_CONTENT_CHARS_FOR_IDEATION:
            txt = await self._summarize_report_with_sllm(
                txt, MAX_REPORT_CONTENT_CHARS_FOR_IDEATION, trace_id, extra
            )
        return txt

    # ──────────────────── 1. Insight 추출 ────────────────────
    async def _extract_key_insights_sllm(
        self,
        summary: str,
        refined_intent: str,
        category: str,
        audience: str,
        trace_id: str,
        extra: dict,
        attempt: int = 1,
    ) -> Optional[str]:

        system_msg = (
            "You are an analytical assistant. List the 3-5 most critical insights that directly answer the "
            "refined question within the news context."
        )
        user_msg = f"""
[Refined Question] {refined_intent}
[News Category] {category}     [Target Audience] {audience}

[Report Summary]
{summary}

• Provide 3-5 bullet (or numbered) insights.
• Each must clearly relate to the refined question.
"""
        if attempt > 1:
            user_msg = f"[Retry {attempt}/{MAX_RETRY_ATTEMPTS+1}] " + user_msg

        res  = await self.llm_service.generate_text(
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user",   "content": user_msg}],
            max_tokens=400,
            temperature=0.25,
        )
        insights = res.get("generated_text", "").strip()
        if (insights.count("\n") < 2 or len(insights) < 40) and attempt <= MAX_RETRY_ATTEMPTS:
            logger.warning("Insight extraction weak, retry.", extra=extra)
            return await self._extract_key_insights_sllm(
                summary, refined_intent, category, audience, trace_id, extra, attempt + 1
            )
        return insights or None

    # ──────────────────── 2. Comic Idea 생성 ────────────────────
    async def _generate_comic_ideas_sllm(
        self,
        summary: str,
        insights: Optional[str],
        refined_intent: str,
        original_query: str,
        category: str,
        num_ideas: int,
        config: dict,
        trace_id: str,
        extra: dict,
        attempt: int = 1,
    ) -> List[Dict[str, Any]]:

        writer_persona = config.get("writer_id", "general_storyteller")
        audience_lbl   = config.get("target_audience", "general_audience")

        # 완전한 XML 스키마 & 규칙
        schema_block = f"""
<idea>
  <title> … </title>
  <logline> … </logline>
  <genre> … </genre>
  <target_emotion> … </target_emotion>
  <key_elements_from_report> … </key_elements_from_report>
</idea>
"""

        system_msg = (
            f"You are a creative comic-idea generator (persona: '{writer_persona}', audience: '{audience_lbl}').\n"
            f"News Category Context: '{category}'.\n"
            "Generate exactly "
            f"{num_ideas} UNIQUE ideas strictly following the XML schema below (no extra text):\n"
            f"{schema_block}\n"
            "Rules:\n"
            "1. Each <logline> must reflect at least one Key Insight.\n"
            "2. NO unrelated inventions.\n"
            "3. Output = concatenated <idea> blocks, nothing else."
        )

        user_msg = f"Report Summary:\n{summary}\n"
        if insights:
            user_msg += f"\nKey Insights:\n{insights}\n"
        user_msg += (
            f'\nOriginal Query: "{original_query}"\nRefined Question: "{refined_intent}"\n'
            "Generate ideas now:"
        )
        if attempt > 1:
            user_msg = f"[Retry {attempt}/{MAX_RETRY_ATTEMPTS+1}] " + user_msg

        res  = await self.llm_service.generate_text(
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user",   "content": user_msg}],
            max_tokens=2_500,
            temperature=0.75,
        )
        raw  = res.get("generated_text", "").strip()
        blocks = re.findall(r"<idea\b[^>]*>(.*?)</idea>", raw, flags=re.S | re.I)

        if len(blocks) < num_ideas and attempt <= MAX_RETRY_ATTEMPTS:
            logger.warning("XML idea blocks missing, retry.", extra=extra)
            return await self._generate_comic_ideas_sllm(
                summary, insights, refined_intent, original_query,
                category, num_ideas, config, trace_id, extra, attempt + 1
            )

        ideas: List[Dict[str, Any]] = []
        for blk in blocks:
            if not self._is_valid_idea_xml(blk):
                continue

            def _get(tag: str) -> str:
                m = re.search(fr"<{tag}>(.*?)</{tag}>", blk, flags=re.S | re.I)
                return m.group(1).strip() if m else ""

            ideas.append(
                {
                    "title": _get("title"),
                    "logline": _get("logline"),
                    "genre": _get("genre"),
                    "target_emotion": _get("target_emotion"),
                    "key_elements_from_report": [
                        e.strip() for e in _get("key_elements_from_report").split(",") if e.strip()
                    ],
                }
            )
        return ideas[:num_ideas]

    # ──────────────────── 3. run() ────────────────────
    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        extra = {"trace_id": state.trace_id, "comic_id": state.comic_id, "node_name": node_name}
        logger.info("Entering N07.", extra=extra)

        error_log = list(state.error_log or [])
        if not state.report_content:
            logger.error("report_content missing.", extra=extra)
            return {"comic_ideas": [], "current_stage": "ERROR", "error_log": error_log}

        refined_intent = state.query_context.get("refined_intent", state.original_query)
        category       = state.query_context.get("query_category", "Other")
        audience       = state.config.get("target_audience", "general_public")
        contextual_sum = getattr(state, "contextual_summary", None)

        # 1) Summary 확보
        summary_text = await self._preprocess_report_content(
            state.report_content, state.trace_id, extra, contextual_sum
        )
        if len(summary_text) < 50:
            logger.error("Pre-processed summary too short.", extra=extra)
            return {"comic_ideas": [], "current_stage": "ERROR", "error_log": error_log}

        # 2) Insight 추출
        insights = await self._extract_key_insights_sllm(
            summary_text, refined_intent, category, audience, state.trace_id, extra
        )

        # 3) Idea 생성
        ideas = await self._generate_comic_ideas_sllm(
            summary_text, insights, refined_intent, state.original_query,
            category, MAX_IDEAS_TO_GENERATE, state.config or {},
            state.trace_id, extra
        )

        logger.info(f"Generated {len(ideas)} comic ideas.", extra=extra)
        return {
            "comic_ideas": ideas,
            "current_stage": "n08_scenario_generation",
            "error_log": error_log,
        }
