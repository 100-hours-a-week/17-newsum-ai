# ai/app/nodes/n07_comic_ideation_node.py
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import re # 보고서 내용 전처리를 위해

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService # LLM 서비스 사용

logger = get_logger(__name__)

# LLM에 전달할 보고서 내용의 최대 문자 수 (토큰 제한 고려)
MAX_REPORT_CONTENT_CHARS_FOR_IDEATION = 4000 # 예시 값, 모델에 따라 조절
MAX_IDEAS_TO_GENERATE = 6 # 한 번에 생성할 아이디어 수

class N07ComicIdeationNode:
    """
    N05에서 생성된 보고서 내용을 바탕으로 여러 만화 아이디어를 생성합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _preprocess_report_content(self, html_content: str) -> str:
        """
        HTML 보고서 내용에서 텍스트만 추출하고 요약/정리합니다.
        (간단한 예시: 실제로는 BeautifulSoup 등 사용 권장)
        """
        # 주석: HTML 태그를 제거하는 더 견고한 방법(예: BeautifulSoup)을 사용하는 것이 좋습니다.
        # 이 코드는 간단한 정규식을 사용하며 모든 HTML 케이스를 처리하지 못할 수 있습니다.
        text_content = re.sub(r'<style[^>]*?>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE) # style 태그 제거
        text_content = re.sub(r'<script[^>]*?>.*?</script>', '', text_content, flags=re.DOTALL | re.IGNORECASE) # script 태그 제거
        text_content = re.sub(r'<[^>]+>', ' ', text_content) # 나머지 HTML 태그 제거
        text_content = re.sub(r'\s+', ' ', text_content).strip() # 다중 공백을 단일 공백으로 변경하고 양쪽 공백 제거

        if len(text_content) > MAX_REPORT_CONTENT_CHARS_FOR_IDEATION:
            # 주석: 내용이 너무 길 경우, 단순히 자르는 것보다 LLM을 사용한 요약이 더 좋을 수 있습니다.
            text_content = text_content[:MAX_REPORT_CONTENT_CHARS_FOR_IDEATION] + "..."
            logger.debug(f"Report content truncated to {MAX_REPORT_CONTENT_CHARS_FOR_IDEATION} chars for ideation.")
        return text_content

    async def _generate_comic_ideas_sllm(
        self, processed_report_content: str, original_query: str, refined_intent: str,
        num_ideas: int, config: dict, trace_id: str, extra_log_data: dict
    ) -> List[Dict[str, Any]]:
        """LLM을 사용하여 만화 아이디어 생성 (구조화된 응답 요청)"""
        writer_persona = config.get('writer_id', 'general_storyteller') # 작가 페르소나 활용
        target_audience = config.get('target_audience', 'general_audience')

        prompt = f"""[System] You are a creative comic idea generator with the persona '{writer_persona}', targeting a '{target_audience}' audience.
Based on the provided 'Report Summary', 'Original User Query', and 'Refined Core Question', generate {num_ideas} unique and engaging comic ideas.

IMPORTANT: Each idea MUST be directly inspired by and summarize key points from the 'Report Summary'. The logline (or summary) of each idea should clearly reflect the main findings or insights from the report. Do NOT invent ideas unrelated to the report content.

Each idea MUST strictly follow this XML-like format:
<idea>
  <title>Comic Title (concise and catchy)</title>
  <logline>A one or two-sentence core plot or concept, summarizing and reflecting the main points of the report</logline>
  <genre>Comic Genre (e.g., Sci-Fi, Fantasy, Drama, Comedy, Educational, Satire)</genre>
  <target_emotion>Primary emotion to evoke in the reader (e.g., Curiosity, Amusement, Empathy, Awareness, Thrill)</target_emotion>
  <key_elements_from_report>Key elements or keywords from the report that inspired this idea (comma-separated)</key_elements_from_report>
</idea>

Report Summary:
{processed_report_content}

Original User Query: "{original_query}"
Refined Core Question: "{refined_intent}"

[Task]
Generate {num_ideas} comic ideas adhering to the specified format. Enclose each idea within <idea>...</idea> tags. Do not include any text outside these tags or between them, other than the ideas themselves.
"""
        logger.debug("Attempting to generate comic ideas from sLLM...", extra=extra_log_data)
        # 주석: 이 프롬프트는 XML과 유사한 구조를 요청하므로, LLM이 잘 생성하도록 max_tokens를 충분히 주고, 후처리 로직이 중요합니다.
        result = await self.llm_service.generate_text(prompt=prompt, max_tokens=2000, temperature=0.75) # 창의성을 위해 온도 약간 높임

        generated_ideas = []
        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM comic idea generation failed: {result.get('error', 'No text generated')}", extra=extra_log_data)
            return []

        raw_text_output = result["generated_text"].strip()
        logger.debug(f"Raw sLLM output for ideas:\n{raw_text_output}", extra=extra_log_data)

        # 주석: <idea>...</idea> 블록 파싱. 더 복잡한 XML/HTML 파서 사용도 고려할 수 있으나, 여기서는 정규식을 사용합니다.
        idea_blocks = re.findall(r"<idea>(.*?)</idea>", raw_text_output, re.DOTALL)
        for block_content in idea_blocks:
            idea = {}
            # 주석: 각 태그 내부의 내용을 추출합니다. re.DOTALL 플래그는 개행 문자도 .에 포함시킵니다.
            title_match = re.search(r"<title>(.*?)</title>", block_content, re.DOTALL)
            logline_match = re.search(r"<logline>(.*?)</logline>", block_content, re.DOTALL)
            genre_match = re.search(r"<genre>(.*?)</genre>", block_content, re.DOTALL)
            emotion_match = re.search(r"<target_emotion>(.*?)</target_emotion>", block_content, re.DOTALL)
            elements_match = re.search(r"<key_elements_from_report>(.*?)</key_elements_from_report>", block_content, re.DOTALL)

            if title_match: idea['title'] = title_match.group(1).strip()
            if logline_match: idea['logline'] = logline_match.group(1).strip()
            if genre_match: idea['genre'] = genre_match.group(1).strip()
            if emotion_match: idea['target_emotion'] = emotion_match.group(1).strip()
            if elements_match: idea['key_elements_from_report'] = [el.strip() for el in elements_match.group(1).split(',') if el.strip()]
            else: idea['key_elements_from_report'] = []


            # 주석: 제목과 로그라인은 아이디어의 필수 요소로 간주합니다.
            if idea.get('title') and idea.get('logline'):
                generated_ideas.append(idea)
            else:
                logger.warning(f"Parsed idea block missing title or logline: {block_content[:100]}...", extra=extra_log_data)


        logger.info(f"Successfully parsed {len(generated_ideas)} comic ideas from LLM output.", extra=extra_log_data)
        return generated_ideas[:num_ideas] # 요청한 아이디어 수만큼만 반환

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')
        error_log = list(state.error_log or [])

        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        logger.info(
            f"Entering node. Input State Summary: {summarize_for_logging(state.model_dump(exclude_none=True), fields_to_show=['original_query', 'query_context.refined_intent', 'report_content_length', 'current_stage'])}",
            extra=extra
        )

        # 주석: state.report_content는 N05에서 생성된 HTML 문자열입니다.
        if not state.report_content:
            error_msg = "Report content is missing. Cannot generate comic ideas."
            logger.warning(error_msg, extra=extra)
            error_log.append({"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            # 주석: 아이디어 생성 없이 다음 노드로 진행하거나, 오류 상태로 변경할 수 있습니다.
            # 여기서는 빈 리스트로 다음 노드로 진행합니다.
            return {
                "comic_ideas": [],
                "current_stage": "n08_scenario_generation",
                "error_log": error_log
            }

        comic_ideas_list = []
        try:
            # 1. 보고서 내용 전처리 (HTML -> 텍스트, 길이 제한)
            processed_content = self._preprocess_report_content(state.report_content)

            if not processed_content.strip() or len(processed_content) < 50: # 너무 짧으면 의미 없음
                 error_msg = "Processed report content is too short or empty for meaningful idea generation."
                 logger.warning(error_msg, extra=extra)
                 # 주석: 이 경우에도 빈 아이디어 리스트로 다음 단계로 넘어갑니다.
                 # 심각한 오류로 처리하고 싶다면 예외를 발생시키거나 current_stage를 "ERROR"로 설정합니다.
                 return {
                    "comic_ideas": [],
                    "current_stage": "n08_scenario_generation",
                    "error_log": error_log # 이전 오류 로그 유지
                 }

            # 2. LLM을 사용하여 만화 아이디어 생성
            original_query = state.original_query or "N/A"
            # 주석: query_context가 존재하고 refined_intent가 있는지 확인합니다.
            refined_intent = state.query_context.get("refined_intent", original_query) if state.query_context else original_query

            comic_ideas_list = await self._generate_comic_ideas_sllm(
                processed_content,
                original_query,
                refined_intent,
                MAX_IDEAS_TO_GENERATE,
                config,
                trace_id,
                extra
            )

            if not comic_ideas_list:
                logger.warning("No comic ideas were generated by the LLM. Proceeding with empty list.", extra=extra)

            update_dict = {
                "comic_ideas": comic_ideas_list,
                "current_stage": "n08_scenario_generation", # 다음 노드로
                "error_log": error_log
            }

            # 주석: 로깅을 위해 comic_ideas의 개수를 별도 필드로 요약합니다.
            log_summary_update = update_dict.copy()
            log_summary_update['comic_ideas_count'] = len(comic_ideas_list)
            if 'comic_ideas' in log_summary_update : del log_summary_update['comic_ideas'] # 실제 데이터는 너무 길 수 있으므로 제거

            logger.info(
                f"Exiting node. Generated {len(comic_ideas_list)} comic ideas. Output Update Summary: {summarize_for_logging(log_summary_update, fields_to_show=['current_stage', 'comic_ideas_count'])}",
                extra=extra
            )
            return update_dict

        except Exception as e:
            error_msg = f"Unexpected error in {node_name} execution: {e}"
            logger.exception(error_msg, extra=extra) # 예외 정보 전체 로깅
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            return {
                "comic_ideas": [], # 실패 시 빈 리스트
                "error_log": error_log,
                "current_stage": "n08_scenario_generation", # 오류가 발생해도 다음 노드로 진행 (선택적) 또는 "ERROR"
                "error_message": f"{node_name} Exception: {error_msg}" # 상태에 최상위 에러 메시지 기록
            }