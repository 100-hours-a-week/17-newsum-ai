# ai/app/nodes/n05_report_generation_node.py

import jinja2  # Jinja2 임포트
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timezone

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService

logger = get_logger(__name__)

# Llama 3.2 (3B) 모델의 컨텍스트 길이 제한을 고려한 설정 (가정치)
# 예: 4096 토큰 모델, 80%는 약 3276 토큰.
# 프롬프트와 생성될 응답 공간을 제외하면, LLM에 한 번에 입력할 컨텍스트(스니펫 텍스트 등)의 길이.
# 1토큰을 약 3자로 가정 시, 3276 * 3 = ~9800자. 안전하게 7000~8000자.
# 각 LLM 호출 시 이 한도를 넘지 않도록 텍스트를 조절해야 함.
# 여기서는 각 LLM 호출에 전달할 '컨텍스트 텍스트'의 최대 문자 수를 정의.
MAX_CONTEXT_CHARS_PER_LLM_CALL = 7500
MAX_SNIPPETS_PER_ASPECT_FOR_LLM = 5  # 한 주제에 대해 LLM이 직접 처리할 스니펫 수
MAX_CHARS_PER_SNIPPET_FOR_LLM = 1500  # 각 스니펫에서 LLM에 전달할 최대 문자 수


class N05ReportGenerationNode:
    """
    N04에서 수집된 검색 결과를 바탕으로, 다회 프롬프팅과 Jinja2 템플릿을 사용하여
    구조화된 보고서를 생성하는 노드. (LLM 부담 줄이기 적용)
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.jinja_env = jinja2.Environment(loader=jinja2.FunctionLoader(self._load_template), autoescape=True)
        self.report_template_name = "deep_research_report_template.jinja2"
        self.MAX_CONTEXT_CHARS = MAX_CONTEXT_CHARS_PER_LLM_CALL  # LLM 호출 시 컨텍스트 최대 문자 수
        self.MAX_SNIPPETS_PER_ASPECT = MAX_SNIPPETS_PER_ASPECT_FOR_LLM
        self.MAX_CHARS_PER_SNIPPET = MAX_CHARS_PER_SNIPPET_FOR_LLM

    def _load_template(self, template_name: str) -> Optional[str]:
        """Jinja2 템플릿 로드 함수 (여기서는 코드 내에 직접 정의)"""
        if template_name == self.report_template_name:
            return """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
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

    {% for section_item in sections %}
    <div class="section">
        <h2>{{ loop.index + 1 }}. {{ section_item.aspect_title | e }}</h2>
        <p>{{ section_item.content | replace('\\n', '<br>') | safe }}</p>
    </div>
    {% endfor %}

    <div class="section">
        <h2>{{ sections|length + 2 }}. 결론 (Conclusion)</h2>
        <p>{{ conclusion | replace('\\n', '<br>') | safe }}</p>
    </div>

    {% if sources and sources|length > 0 %}
    <div class="section sources">
        <h2>{{ sections|length + 3 }}. 참고 자료 (Sources)</h2>
        <ul>
            {% for source in sources %}
            <li><a href="{{ source.url }}" target="_blank">{{ source.title | e }}</a> (검색 엔진: {{source.tool_used | e }})</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
    <hr>
    <p><small>본 보고서는 AI에 의해 {{ generation_timestamp }}에 생성되었습니다. Writer Persona: {{ writer_id }}</small></p>
</body>
</html>
"""
        return None

    def _truncate_text(self, text: str, max_length: int) -> str:
        """텍스트를 최대 길이로 자르고, 너무 길면 [...] 추가"""
        if len(text) > max_length:
            return text[:max_length - 3] + "..."
        return text

    async def _sllm_call(self, prompt: str, max_tokens: int, temperature: float, extra_log_data: dict) -> Optional[str]:
        """LLM 호출을 위한 공통 래퍼 함수"""
        result = await self.llm_service.generate_text(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM call failed: {result.get('error', 'No text generated')}", extra=extra_log_data)
            return None
        return result["generated_text"].strip()

    def _filter_and_prepare_snippets_for_aspect(
            self, key_aspect: str, all_raw_snippets: List[Dict], extra_log_data: dict
    ) -> List[Dict[str, Any]]:
        """특정 주제(aspect)와 관련된 스니펫을 필터링하고 LLM 입력용으로 준비"""
        relevant_snippets = []
        # 간단한 키워드 매칭 (개선 가능: TF-IDF, Embedding 유사도 등)
        aspect_keywords = set(key_aspect.lower().split())
        for snip in all_raw_snippets:
            title = snip.get("title", "").lower()
            snippet_text = snip.get("snippet", "").lower()
            content_text = title + " " + snippet_text
            if any(kw in content_text for kw in aspect_keywords):
                # URL에서 도메인 추출
                from urllib.parse import urlparse
                domain = urlparse(snip.get("url", "")).netloc

                relevant_snippets.append({
                    "title": snip.get("title", "N/A"),
                    "url": snip.get("url", "#"),
                    "snippet": self._truncate_text(snip.get("snippet", ""), self.MAX_CHARS_PER_SNIPPET),
                    # LLM 입력용 스니펫 길이 제한
                    "source_domain": domain
                })

        # 관련도 높은 순으로 정렬 (여기서는 단순 추가 순서 유지, 추후 개선 가능)
        # LLM에 한 번에 전달할 스니펫 수 제한
        logger.debug(
            f"For aspect '{key_aspect}', found {len(relevant_snippets)} relevant snippets. Taking up to {self.MAX_SNIPPETS_PER_ASPECT}.",
            extra=extra_log_data)
        return relevant_snippets[:self.MAX_SNIPPETS_PER_ASPECT]

    async def _generate_report_title_sllm(
            self, refined_intent: str, config: dict, trace_id: str, extra_log_data: dict
    ) -> str:
        writer_id = config.get('writer_id', 'default_writer')
        prompt = f"""[System] You are a report title generator with the persona '{writer_id}'.
Based on the following 'Refined Core Question', create a concise, informative, and engaging title for a research report.
Respond ONLY with the title.

Refined Core Question: "{refined_intent}"

[Task]
Report Title:"""
        title = await self._sllm_call(prompt, max_tokens=60, temperature=0.3, extra_log_data=extra_log_data)
        return title or f"Research Report on: {refined_intent}"  # Fallback

    async def _generate_introduction_sllm(
            self, original_query: str, refined_intent: str, config: dict, trace_id: str, extra_log_data: dict
    ) -> str:
        writer_id = config.get('writer_id', 'default_writer')
        target_audience = config.get('target_audience', 'general_public')
        prompt = f"""[System] You are an AI writer with the persona '{writer_id}', tasked with writing an introduction for a research report aimed at a '{target_audience}' audience.
The report addresses the user's 'Original Query' which has been clarified into a 'Refined Core Question'.
Write a brief introduction (2-3 paragraphs) that sets the context, states the report's purpose (based on the refined question), and briefly outlines what the report will cover.
Respond ONLY with the introduction text.

Original Query: "{original_query}"
Refined Core Question: "{refined_intent}"

[Task]
Report Introduction:"""
        intro = await self._sllm_call(prompt, max_tokens=400, temperature=0.4, extra_log_data=extra_log_data)
        return intro or "Introduction could not be generated."

    async def _summarize_snippets_for_aspect_sllm(
            self, key_aspect: str, snippets_to_summarize: List[Dict[str, Any]], refined_intent: str,
            config: dict, trace_id: str, extra_log_data: dict
    ) -> str:
        """여러 스니펫을 하나의 요약된 텍스트로 만듦 (컨텍스트 길이 초과 시 사용)"""
        writer_id = config.get('writer_id', 'default_writer')
        snippets_text = "\n\n".join(
            [f"Source Title: {s['title']}\nSnippet: {s['snippet']}" for s in snippets_to_summarize])

        # 전체 스니펫 텍스트가 너무 길면, 추가로 자르거나 다른 전략 필요 (여기서는 한번에 요약 시도)
        truncated_snippets_text = self._truncate_text(snippets_text, self.MAX_CONTEXT_CHARS - 500)  # 프롬프트 길이 고려하여 여유

        prompt = f"""[System] You are an AI research assistant with the persona '{writer_id}'.
Your task is to synthesize information from multiple text snippets related to a specific 'Key Aspect' of a larger research topic ('Overall Research Focus').
Summarize the key findings, facts, and arguments from the 'Provided Snippets' that are relevant to the 'Key Aspect'.
The summary should be a coherent paragraph or two.
Respond ONLY with the summary text.

Overall Research Focus: "{refined_intent}"
Key Aspect to Summarize: "{key_aspect}"
Provided Snippets:
{truncated_snippets_text}

[Task]
Synthesized Summary for the Key Aspect:"""
        summary = await self._sllm_call(prompt, max_tokens=500, temperature=0.3, extra_log_data=extra_log_data)
        return summary or f"Summary for '{key_aspect}' could not be generated from the provided snippets."

    async def _generate_section_content_sllm(
            self, key_aspect: str, relevant_snippets: List[Dict[str, Any]],
            refined_intent: str, config: dict, trace_id: str, extra_log_data: dict
    ) -> Tuple[str, List[Dict[str, Any]]]:  # 생성된 텍스트와 실제 사용된 스니펫 목록 반환
        """특정 주제(aspect)에 대한 본론 섹션 내용 생성. 필요 시 스니펫 요약 선행."""
        writer_id = config.get('writer_id', 'default_writer')
        target_audience = config.get('target_audience', 'general_public')

        # LLM에 전달할 스니펫 텍스트 준비
        # 모든 스니펫 텍스트를 합쳤을 때 self.MAX_CONTEXT_CHARS를 넘는지 확인
        current_snippets_text = "\n\n".join(
            [f"Source: {s.get('source_domain', 'N/A')}\nTitle: {s['title']}\nSnippet: {s['snippet']}" for s in
             relevant_snippets])

        context_for_llm = ""
        actually_used_snippets = relevant_snippets  # 기본적으로는 전달된 모든 스니펫이 사용되었다고 가정

        if len(current_snippets_text) > self.MAX_CONTEXT_CHARS:
            logger.warning(
                f"Snippets for aspect '{key_aspect}' (length {len(current_snippets_text)}) exceed MAX_CONTEXT_CHARS ({self.MAX_CONTEXT_CHARS}). Summarizing first.",
                extra=extra_log_data)
            # 스니펫 요약 (실제로는 이 요약 과정에서 어떤 스니펫이 기여했는지 추적 어려워짐)
            # 여기서는 요약된 내용을 컨텍스트로 사용하고, 원본 스니펫 목록을 그대로 반환 (개선 필요)
            context_for_llm = await self._summarize_snippets_for_aspect_sllm(
                key_aspect, relevant_snippets, refined_intent, config, trace_id, extra_log_data
            )
        else:
            context_for_llm = current_snippets_text

        # 컨텍스트가 비어있으면 (스니펫이 없거나 요약 실패) 기본 메시지 생성
        if not context_for_llm.strip():
            logger.warning(f"No content/context available for aspect '{key_aspect}' after processing snippets.",
                           extra=extra_log_data)
            return f"Information regarding '{key_aspect}' could not be sufficiently processed from the provided sources.", []

        prompt = f"""[System] You are an AI writer with the persona '{writer_id}', writing a section of a research report for a '{target_audience}' audience.
The overall report is about: "{refined_intent}".
This specific section focuses on the 'Key Aspect': "{key_aspect}".
Based on the 'Provided Context' (which consists of summarized or direct web snippets), write a comprehensive and informative section (3-5 paragraphs).
Analyze, synthesize, and present the information clearly. Highlight key findings or important points.
Respond ONLY with the text for this section.

Key Aspect for this Section: "{key_aspect}"
Provided Context:
{context_for_llm}

[Task]
Report Section Content for '{key_aspect}':"""
        section_content = await self._sllm_call(prompt, max_tokens=800, temperature=0.5,
                                                extra_log_data=extra_log_data)  # 내용 생성이므로 온도 약간 높임

        # 실제로는 context_for_llm 생성에 기여한 스니펫만 반환해야 함
        # 현재는 필터링된 relevant_snippets를 그대로 반환 (정확도 개선 필요)
        return section_content or f"Content for section '{key_aspect}' could not be generated.", actually_used_snippets

    async def _generate_conclusion_sllm(
            self, refined_intent: str, section_contents: List[str], config: dict, trace_id: str, extra_log_data: dict
    ) -> str:
        writer_id = config.get('writer_id', 'default_writer')
        target_audience = config.get('target_audience', 'general_public')

        # 섹션 내용이 너무 길면 요약하거나 일부만 사용
        summarized_sections = "\n\n---\n\n".join([self._truncate_text(s, 1000) for s in section_contents])  # 각 섹션 요약 길이
        final_sections_summary = self._truncate_text(summarized_sections, self.MAX_CONTEXT_CHARS - 500)

        prompt = f"""[System] You are an AI writer with the persona '{writer_id}', tasked with writing the conclusion for a research report aimed at a '{target_audience}' audience.
The report explored the 'Refined Core Question'. Key insights from the main sections are summarized in 'Summary of Report Sections'.
Write a concise conclusion (2-3 paragraphs) that summarizes the main findings of the report, offers final thoughts or insights, and perhaps suggests potential implications or future directions.
Do not introduce new information not covered in the report sections.
Respond ONLY with the conclusion text.

Refined Core Question of the Report: "{refined_intent}"
Summary of Report Sections:
{final_sections_summary}

[Task]
Report Conclusion:"""
        conclusion = await self._sllm_call(prompt, max_tokens=400, temperature=0.4, extra_log_data=extra_log_data)
        return conclusion or "Conclusion could not be generated."

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')

        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        logger.info(f"Entering node. Generating report.", extra=extra_log_data)

        report_data_for_template: Dict[str, Any] = {  # Jinja2 템플릿에 전달될 데이터
            "title": "Research Report (Title to be generated)",
            "original_query": state.original_query,
            "refined_intent": "N/A",
            "introduction": "N/A",
            "sections": [],  # List of {"aspect_title": str, "content": str, "supporting_snippets": List[Dict]}
            "conclusion": "N/A",
            "sources": [],  # List of {"title": str, "url": str, "tool_used": str}
            "generation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "writer_id": writer_id
        }
        error_log = list(state.error_log or [])

        query_context = state.query_context
        if not query_context or not state.raw_search_results:
            error_msg = "Query context or raw search results are missing. Cannot generate report."
            logger.error(error_msg, extra=extra_log_data)
            error_log.append(
                {"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            # 보고서 내용 없이 다음 단계로 (오류 상태로)
            return {
                "report_content": "Error: Missing required data for report generation.",
                "current_stage": "ERROR",  # 또는 다음 단계
                "error_message": error_msg,
                "error_log": error_log
            }

        refined_intent = query_context.get("refined_intent", state.original_query)
        report_data_for_template["refined_intent"] = refined_intent

        # 1. 보고서 제목 생성 (sLLM 호출)
        try:
            title = await self._generate_report_title_sllm(refined_intent, config, trace_id, extra_log_data)
            report_data_for_template["title"] = title
        except Exception as e:
            logger.error(f"Failed to generate report title: {e}", exc_info=True, extra=extra_log_data)
            error_log.append({"stage": f"{node_name}._generate_title", "error": str(e),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            # Fallback title은 이미 설정됨

        # 2. 서론 생성 (sLLM 호출)
        try:
            introduction = await self._generate_introduction_sllm(
                state.original_query, refined_intent, config, trace_id, extra_log_data
            )
            report_data_for_template["introduction"] = introduction
        except Exception as e:
            logger.error(f"Failed to generate introduction: {e}", exc_info=True, extra=extra_log_data)
            error_log.append({"stage": f"{node_name}._generate_introduction", "error": str(e),
                              "timestamp": datetime.now(timezone.utc).isoformat()})

        # 3. 각 주제(Aspect)별 본론 섹션 생성
        key_aspects = query_context.get("key_aspects_to_search", [])
        all_raw_snippets = state.raw_search_results or []

        processed_source_urls_for_main_list: Set[str] = set()  # 전체 보고서 참고자료 중복 방지용

        for aspect in key_aspects:
            section_extra_log = {**extra_log_data, "current_aspect": aspect}
            logger.info(f"Processing section for aspect: {aspect}", extra=section_extra_log)
            try:
                # 3a. 현재 주제와 관련된 스니펫 필터링 및 선택
                relevant_snippets_for_aspect = self._filter_and_prepare_snippets_for_aspect(
                    aspect, all_raw_snippets, section_extra_log
                )
                if not relevant_snippets_for_aspect:
                    logger.warning(f"No relevant snippets found for aspect: {aspect}", extra=section_extra_log)
                    report_data_for_template["sections"].append({
                        "aspect_title": aspect,
                        "content": f"No specific information could be gathered for '{aspect}' from the available sources.",
                        "supporting_snippets": []
                    })
                    continue

                # 3b. 선택된 스니펫을 바탕으로 섹션 내용 생성 (sLLM 호출, 내부적으로 요약 포함 가능)
                section_content_text, snippets_used_in_section = await self._generate_section_content_sllm(
                    aspect, relevant_snippets_for_aspect, refined_intent, config, trace_id, section_extra_log
                )
                report_data_for_template["sections"].append({
                    "aspect_title": aspect,
                    "content": section_content_text,
                    "supporting_snippets": snippets_used_in_section  # 이 섹션 생성에 직접 사용된 스니펫
                })

                # 전체 보고서 참고 자료 목록 업데이트 (중복 방지)
                for snip in snippets_used_in_section:
                    url = snip.get("url")
                    if url and url not in processed_source_urls_for_main_list:
                        report_data_for_template["sources"].append({
                            "title": snip.get("title", "Source"),
                            "url": url,
                            # N04에서 'tool_used'를 각 스니펫에 추가했다면 여기서 사용 가능
                            "tool_used": snip.get("tool_used", "N/A")
                        })
                        processed_source_urls_for_main_list.add(url)

            except Exception as e:
                logger.error(f"Failed to generate section for aspect '{aspect}': {e}", exc_info=True,
                             extra=section_extra_log)
                error_log.append({"stage": f"{node_name}._generate_section.{aspect}", "error": str(e),
                                  "timestamp": datetime.now(timezone.utc).isoformat()})
                report_data_for_template["sections"].append({
                    "aspect_title": aspect,
                    "content": f"Error generating content for '{aspect}'.",
                    "supporting_snippets": []
                })

        # 4. 결론 생성 (sLLM 호출)
        try:
            section_contents_for_conclusion = [sec["content"] for sec in report_data_for_template["sections"]]
            conclusion = await self._generate_conclusion_sllm(
                refined_intent, section_contents_for_conclusion, config, trace_id, extra_log_data
            )
            report_data_for_template["conclusion"] = conclusion
        except Exception as e:
            logger.error(f"Failed to generate conclusion: {e}", exc_info=True, extra=extra_log_data)
            error_log.append({"stage": f"{node_name}._generate_conclusion", "error": str(e),
                              "timestamp": datetime.now(timezone.utc).isoformat()})

        # 5. Jinja2 템플릿을 사용하여 최종 보고서 렌더링
        final_report_html = "Error: Report template could not be rendered."  # Fallback
        try:
            template = self.jinja_env.get_template(self.report_template_name)
            final_report_html = template.render(report_data_for_template)
            logger.info("Report content successfully rendered using Jinja2 template.", extra=extra_log_data)
        except Exception as e:
            logger.error(f"Failed to render Jinja2 template: {e}", exc_info=True, extra=extra_log_data)
            error_log.append({"stage": f"{node_name}.render_template", "error": str(e),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            # final_report_html은 이미 fallback 값으로 설정됨

        update_dict = {
            "report_content": final_report_html,  # 생성된 HTML 보고서 내용
            "current_stage": "n06_finalize_report",  # 다음 스테이지 (가칭)
            "error_log": error_log
        }
        logger.info(
            f"Exiting node. Report generation complete. Output Update Summary: {summarize_for_logging(update_dict, fields_to_show=['current_stage', 'report_content_length'])}",
            extra=extra_log_data)
        # 로깅 위해 report_content_length 추가
        if 'report_content_length' not in update_dict:
            update_dict_summary = update_dict.copy()
            update_dict_summary['report_content_length'] = len(final_report_html)
            logger.info(
                f"Exiting node. Report generation complete. Output Update Summary: {summarize_for_logging(update_dict_summary, fields_to_show=['current_stage', 'report_content_length'])}",
                extra=extra_log_data)

        return update_dict