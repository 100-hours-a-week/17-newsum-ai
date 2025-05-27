# ai/app/nodes_v2/n05_report_generation_node.py

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import jinja2
from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)


class N05ReportGenerationNode:
    """
    (업그레이드됨) Qwen3 기반 한 번의 LLM 호출로
    구조화된 보고서 JSON을 반환받고, Jinja2 템플릿으로 HTML을 렌더링하여
    state.report.report_content에 저장하는 노드.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FunctionLoader(self._load_template),
            autoescape=True
        )
        self.template_name = "deep_research_report_template.jinja2"

    def _load_template(self, name: str) -> Optional[str]:
        if name != self.template_name:
            return None
        # 복사해 온 기존 Jinja2 템플릿
        return """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>{{ title | e }}</title>
    <style>
        body { font-family: sans-serif; line-height: 1.6; margin: 20px; }
        h1, h2, h3 { color: #333; }
        .section { margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid #eee; }
        .sources li { margin-bottom: 5px; }
        .summary { font-style: italic; color: #555; margin-bottom:15px; padding:10px; background-color:#f9f9f9; border-left: 3px solid #007bff; }
    </style>
</head>
<body>
    <h1>{{ title | e }}</h1>
    <p class="summary"><strong>요청 쿼리:</strong> {{ original_query | e }}<br>
    <strong>정제된 핵심 질문:</strong> {{ refined_intent | e }}</p>

    <div class="section">
        <h2>1. 서론 (Introduction)</h2>
        <p>{{ introduction | replace('\\n', '<br>') | safe }}</p>
    </div>

    {% for sec in sections %}
    <div class="section">
        <h2>{{ loop.index + 1 }}. {{ sec.aspect_title | e }}</h2>
        <p>{{ sec.content | replace('\\n', '<br>') | safe }}</p>
    </div>
    {% endfor %}

    <div class="section">
        <h2>{{ sections|length + 2 }}. 결론 (Conclusion)</h2>
        <p>{{ conclusion | replace('\\n', '<br>') | safe }}</p>
    </div>

    {% if sources %}
    <div class="section sources">
        <h2>{{ sections|length + 3 }}. 참고 자료 (Sources)</h2>
        <ul>
            {% for src in sources %}
            <li><a href="{{ src.url }}" target="_blank">{{ src.title | e }}</a>
                (검색 엔진: {{ src.tool_used | e }})</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
    <hr>
    <p><small>본 보고서는 AI에 의해 {{ generation_timestamp }}에 생성되었습니다.
    Writer Persona: {{ writer_id }}</small></p>
</body>
</html>
"""

    def _build_system_prompt(self, original_query: str, refined_intent: str, snippets: List[Dict[str, Any]]) -> str:
        """
        Qwen3에게 JSON 스키마에 맞춰 보고서 내용을 생성하도록 지시하는 System Prompt.
        """
        schema_example = {
            "title": "<string>",
            "introduction": "<string>",
            "sections": [
                {"aspect_title": "<string>", "content": "<string>"}
            ],
            "conclusion": "<string>",
            "sources": [
                {"title": "<string>", "url": "<string>", "tool_used": "<string>"}
            ]
        }
        json_schema = json.dumps(schema_example, ensure_ascii=False, indent=2)
        # 간단한 스니펫 요약을 하나의 문자열로 병합
        snippets_text = "\n".join(
            f"- [{s.get('source_domain','N/A')}] {s.get('title','')} — {s.get('snippet','')} ({s.get('url','')})"
            for s in snippets
        )
        return f"""
You are an expert research report generation agent.
Using the provided search snippets, write a detailed and structured report in JSON format.
Write in natural Korean (한국어로 작성하세요), targeting a professional but non-specialist audience (e.g., tech readers, policymakers, advanced students).

Each 'section' in the JSON should consist of 3 to 5 well-developed paragraphs (~300+ words), clearly explaining the key aspect with relevant facts, analysis, and examples.
Avoid vague statements or generic text. Emphasize clarity and insight.

# Inputs
Original Query: "{original_query}"
Refined Core Question: "{refined_intent}"
Context Snippets (up to {len(snippets)} items):
{snippets_text}

Return ONLY valid JSON matching the schema below, NO think tags, NO explanation, NO comments.

# Output Format
{json_schema}
"""

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta = state.meta
        query_sec = state.query
        search_sec = state.search
        config_sec = state.config

        node_name = self.__class__.__name__
        trace_id = meta.trace_id
        comic_id = meta.comic_id
        writer_id = config_sec.config.get("writer_id", "default_writer")

        extra = {
            "trace_id": trace_id,
            "comic_id": comic_id,
            "node": node_name,
        }
        logger.info("Entering N05ReportGenerationNode", extra=extra)

        original_query = query_sec.original_query
        refined_intent = query_sec.query_context.get("refined_intent", original_query)
        raw_results = search_sec.raw_search_results or []

        # 스니펫 객체 준비 (최대 5개)
        snippets = []
        for item in raw_results[:5]:
            snippets.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("url", ""),
                "source_domain": item.get("source", item.get("source_domain", "N/A"))
            })

        # 1) 한 번의 LLM 호출로 JSON 생성
        system_prompt = self._build_system_prompt(original_query, refined_intent, snippets)
        # qwen3 사용 시 prompt 인자는 간단히 Refined Intent 정도로 활용
        llm_resp = await self.llm_service.generate_text(
            system_prompt_content=system_prompt,
            prompt=refined_intent,
            temperature=0.3,
            max_tokens=4096,
        )
        raw_txt = llm_resp.get("generated_text", "")
        logger.debug("Raw LLM response for report JSON: %s", raw_txt, extra=extra)

        # 2) JSON 파싱
        try:
            report_json = extract_json(raw_txt)
        except Exception as e:
            logger.error("Failed to parse report JSON: %s", e, extra=extra)
            meta.current_stage = "ERROR"
            meta.error_message = f"JSON parsing error in N05: {e}"
            return {"meta": meta.model_dump()}

        # 3) Jinja2로 HTML 렌더링
        report_data = {
            "title": report_json.get("title", ""),
            "original_query": original_query,
            "refined_intent": refined_intent,
            "introduction": report_json.get("introduction", ""),
            "sections": report_json.get("sections", []),
            "conclusion": report_json.get("conclusion", ""),
            "sources": report_json.get("sources", []),
            "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "writer_id": writer_id,
        }

        try:
            template = self.jinja_env.get_template(self.template_name)
            html = template.render(report_data)
            state.report.report_content = html
            meta.current_stage = "n06_finalize_report"
            logger.info("Report HTML successfully rendered.", extra=extra)
        except Exception as e:
            logger.error("Failed to render report template: %s", e, extra=extra)
            meta.current_stage = "ERROR"
            meta.error_message = f"Template rendering error in N05: {e}"
            state.report.report_content = ""

        return {
            "report": state.report.model_dump(),
            "meta": meta.model_dump(),
        }
