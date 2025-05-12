# ai/app/nodes/n07_comic_ideation_node.py
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import re

from app.workflows.state import WorkflowState  # (상대 경로로 수정 가능성 있음)
from app.utils.logger import get_logger, summarize_for_logging  # (상대 경로로 수정 가능성 있음)
# from app.services.llm_service import Llama3VLLMService # 실제 서비스 직접 임포트
from app.services.llm_service import LLMService  # 프로토콜로 타입 힌팅

logger = get_logger(__name__)

MAX_REPORT_CONTENT_CHARS_FOR_IDEATION = 4000
MAX_IDEAS_TO_GENERATE = 3
MAX_RETRY_ATTEMPTS = 2  # 재시도 최대 횟수


class N07ComicIdeationNode:
    """
    보고서 내용 기반 만화 아이디어 생성 (Llama 3 최적화, 재요청 준비)
    """

    def __init__(self, llm_service: LLMService):  # LLMService 프로토콜 사용
        self.llm_service = llm_service  # Llama3VLLMService 인스턴스가 주입될 것임

    def _is_valid_idea_xml(self, xml_block_content: str) -> bool:
        """생성된 아이디어 XML 블록의 기본 유효성 검사"""
        if not xml_block_content or not isinstance(xml_block_content, str):
            return False
        # 필수 태그 존재 여부 확인 (간단한 예시)
        required_tags = ["<title>", "</title>", "<logline>", "</logline>", "<genre>", "</genre>",
                         "<target_emotion>", "</target_emotion>", "<key_elements_from_report>",
                         "</key_elements_from_report>"]
        return all(tag in xml_block_content for tag in required_tags)

    async def _summarize_report_with_sllm(
            self, text_content: str, max_summary_length_chars: int, trace_id: str, extra_log_data: dict,
            attempt: int = 1
    ) -> str:
        approx_words = max_summary_length_chars // 5
        system_prompt = "You are a helpful assistant that summarizes texts concisely, focusing on critical findings and key information. Ensure the summary is well-structured and captures the essence of the original text."
        user_prompt = f"Concisely summarize the most critical findings and key information in the following text. The summary should be approximately {approx_words} words and must not exceed {max_summary_length_chars} characters: \n\nReport Content:\n{text_content}"
        if attempt > 1:  # 재시도 시 프롬프트에 추가 정보
            user_prompt = f"[Attempt {attempt}/{MAX_RETRY_ATTEMPTS + 1}] Previous summarization attempt might have been too long or missed key details. Please try again. Ensure the summary is concise, within {max_summary_length_chars} chars, and captures critical information.\n\n" + user_prompt

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        logger.debug(f"Attempting to summarize report (attempt {attempt}). Target chars: {max_summary_length_chars}",
                     extra=extra_log_data)

        result = await self.llm_service.generate_text(messages=messages, max_tokens=1000, temperature=0.3)

        if "error" in result or not result.get("generated_text"):
            logger.error(
                f"sLLM report summarization failed (attempt {attempt}): {result.get('error', 'No text generated')}",
                extra=extra_log_data)
            if attempt <= MAX_RETRY_ATTEMPTS:
                return await self._summarize_report_with_sllm(text_content, max_summary_length_chars, trace_id,
                                                              extra_log_data, attempt + 1)
            logger.warning(f"Max retries reached for summarization. Falling back to truncation.", extra=extra_log_data)
            return text_content[:max_summary_length_chars] + "..."

        summarized_content = result["generated_text"].strip()
        # (간단한 유효성 검사: 너무 짧거나 길지 않은지 - 여기서는 길이 제한은 프롬프트에 위임)
        logger.info(
            f"Report content summarized (attempt {attempt}). Original: {len(text_content)}, Summarized: {len(summarized_content)}",
            extra=extra_log_data)
        return summarized_content

    async def _preprocess_report_content(self, html_content: str, trace_id: str, extra_log_data: dict) -> str:
        text_content = re.sub(r'<style[^>]*?>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        text_content = re.sub(r'<script[^>]*?>.*?</script>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
        text_content = re.sub(r'<[^>]+>', ' ', text_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()

        if len(text_content) > MAX_REPORT_CONTENT_CHARS_FOR_IDEATION:
            logger.info(
                f"Report content (len: {len(text_content)}) > {MAX_REPORT_CONTENT_CHARS_FOR_IDEATION}. Starting SLM summarization.",
                extra=extra_log_data)
            text_content = await self._summarize_report_with_sllm(text_content, MAX_REPORT_CONTENT_CHARS_FOR_IDEATION,
                                                                  trace_id, extra_log_data)
        return text_content

    async def _extract_key_insights_sllm(
            self, processed_report_content: str, trace_id: str, extra_log_data: dict,
            attempt: int = 1
    ) -> Optional[str]:
        system_prompt = "You are an analytical assistant. Your task is to identify the 3 to 5 most critical conclusions or novel findings from the provided 'Report Summary'. List them concisely, as these will be the foundation for generating comic ideas. Each insight should be a distinct point."
        user_prompt = f"""From the 'Report Summary' provided below, please identify and list 3 to 5 most critical conclusions or novel findings.
Ensure each identified insight is distinct and directly supported by the report content.

Report Summary:
{processed_report_content}

List the critical conclusions/findings (e.g., as a numbered or bulleted list):"""
        if attempt > 1:
            user_prompt = f"[Attempt {attempt}/{MAX_RETRY_ATTEMPTS + 1}] Previous attempt to extract insights was not clear or was incomplete. Please ensure you list 3-5 distinct critical findings.\n\n" + user_prompt

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        logger.debug(f"Attempting to extract key insights (attempt {attempt})...", extra=extra_log_data)
        result = await self.llm_service.generate_text(messages=messages, max_tokens=500, temperature=0.2)

        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM key insight extraction failed (attempt {attempt}): {result.get('error', 'No text')}",
                         extra=extra_log_data)
            if attempt <= MAX_RETRY_ATTEMPTS:
                return await self._extract_key_insights_sllm(processed_report_content, trace_id, extra_log_data,
                                                             attempt + 1)
            return None  # Max retries reached

        extracted_insights = result["generated_text"].strip()
        # (간단한 유효성 검사: 내용이 있는지, 너무 짧지 않은지)
        if not extracted_insights or len(extracted_insights) < 10:
            logger.warning(
                f"Extracted insights seem too short or empty (attempt {attempt}): '{extracted_insights}'. Retrying if possible.",
                extra=extra_log_data)
            if attempt <= MAX_RETRY_ATTEMPTS:
                return await self._extract_key_insights_sllm(processed_report_content, trace_id, extra_log_data,
                                                             attempt + 1)
            return None

        logger.info(f"Extracted key insights (attempt {attempt}): {extracted_insights[:200]}...", extra=extra_log_data)
        return extracted_insights

    async def _generate_comic_ideas_sllm(
            self, processed_report_content: str, extracted_insights: Optional[str],
            original_query: str, refined_intent: str,
            num_ideas: int, config: dict, trace_id: str, extra_log_data: dict,
            attempt: int = 1
    ) -> List[Dict[str, Any]]:
        writer_persona = config.get('writer_id', 'general_storyteller')
        target_audience = config.get('target_audience', 'general_audience')

        system_prompt = f"""You are a creative comic idea generator with the persona '{writer_persona}', targeting a '{target_audience}' audience.
Your goal is to generate {num_ideas} unique and engaging comic ideas. Each idea MUST strictly follow this XML-like format:
<idea>
  <title>Comic Title (concise and catchy)</title>
  <logline>A one or two-sentence core plot or concept, reflecting main report points/insights. This logline should hint at a deeper connection or implication, sparking curiosity.</logline>
  <genre>Comic Genre (e.g. Comedy)</genre>
  <target_emotion>Primary emotion to evoke (e.g., Curiosity, Amusement, Empathy, Awareness, Thrill)</target_emotion>
  <key_elements_from_report>Specific phrases, data points, or direct quotes from the Report Summary supporting the logline and its key insight (comma-separated).</key_elements_from_report>
</idea>

IMPORTANT INSTRUCTIONS:
1. Base ideas on 'Report Summary' and especially 'Previously Extracted Key Insights' if provided.
2. Each <logline> MUST directly reflect one or more key insights.
3. For "depth": Connect insights to 'what if' scenarios, unexpected consequences, or non-obvious underlying causes.
4. Do NOT invent ideas unrelated to the report. Adhere strictly to the XML format.
5. Generate exactly {num_ideas} ideas, each enclosed in <idea>...</idea> tags. No other text outside or between these tags.
"""
        user_prompt = f"""Based on the information below, generate {num_ideas} comic ideas.
Report Summary:
{processed_report_content}
"""
        if extracted_insights:
            user_prompt += f"""
Previously Extracted Key Insights (Use these as primary inspiration for loglines):
<extracted_insights>
{extracted_insights}
</extracted_insights>
"""
        else:
            user_prompt += "\nNo specific pre-extracted insights were provided; base ideas on the overall Report Summary.\n"
        user_prompt += f"""
Original User Query: "{original_query}"
Refined Core Question: "{refined_intent}"
"""
        if attempt > 1:
            user_prompt = f"[Attempt {attempt}/{MAX_RETRY_ATTEMPTS + 1}] Previous attempt to generate ideas did not follow the XML format correctly or was incomplete. Please ensure you follow all instructions and the XML structure precisely for {num_ideas} ideas.\n\n" + user_prompt
        user_prompt += f"\nGenerate {num_ideas} comic ideas now:"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        logger.debug(f"Attempting to generate comic ideas (attempt {attempt})...", extra=extra_log_data)
        result = await self.llm_service.generate_text(messages=messages, max_tokens=2500,
                                                                    temperature=0.75)

        generated_ideas = []
        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM idea generation failed (attempt {attempt}): {result.get('error', 'No text')}",
                         extra=extra_log_data)
            if attempt <= MAX_RETRY_ATTEMPTS:
                return await self._generate_comic_ideas_sllm(processed_report_content, extracted_insights,
                                                             original_query, refined_intent, num_ideas, config,
                                                             trace_id, extra_log_data, attempt + 1)
            return []

        raw_text_output = result["generated_text"].strip()
        idea_blocks = re.findall(r"<idea>(.*?)</idea>", raw_text_output, re.DOTALL)

        if len(idea_blocks) < num_ideas:
            logger.warning(
                f"Expected {num_ideas} idea blocks, but found {len(idea_blocks)} (attempt {attempt}). Raw: {raw_text_output[:300]}",
                extra=extra_log_data)
            if attempt <= MAX_RETRY_ATTEMPTS:  # 조건 추가: 모든 아이디어가 제대로 생성되지 않았으면 재시도
                return await self._generate_comic_ideas_sllm(processed_report_content, extracted_insights,
                                                             original_query, refined_intent, num_ideas, config,
                                                             trace_id, extra_log_data, attempt + 1)

        for idx, block_content in enumerate(idea_blocks):
            if not self._is_valid_idea_xml(block_content):
                logger.warning(
                    f"Invalid XML structure for idea block #{idx + 1} (attempt {attempt}). Content: {block_content[:200]}",
                    extra=extra_log_data)
                if attempt <= MAX_RETRY_ATTEMPTS:  # 개별 아이템이 아닌 전체 결과셋에 대한 재시도
                    logger.info(f"Retrying entire idea generation due to invalid block (attempt {attempt}).",
                                extra=extra_log_data)
                    return await self._generate_comic_ideas_sllm(processed_report_content, extracted_insights,
                                                                 original_query, refined_intent, num_ideas, config,
                                                                 trace_id, extra_log_data, attempt + 1)
                continue  # Max retries: skip this invalid block

            idea = {}
            title_match = re.search(r"<title>(.*?)</title>", block_content, re.DOTALL)
            logline_match = re.search(r"<logline>(.*?)</logline>", block_content, re.DOTALL)
            genre_match = re.search(r"<genre>(.*?)</genre>", block_content, re.DOTALL)
            emotion_match = re.search(r"<target_emotion>(.*?)</target_emotion>", block_content, re.DOTALL)
            elements_match = re.search(r"<key_elements_from_report>(.*?)</key_elements_from_report>", block_content,
                                       re.DOTALL)

            if title_match: idea['title'] = title_match.group(1).strip()
            if logline_match: idea['logline'] = logline_match.group(1).strip()
            if genre_match: idea['genre'] = genre_match.group(1).strip()
            if emotion_match: idea['target_emotion'] = emotion_match.group(1).strip()
            if elements_match:
                idea['key_elements_from_report'] = [el.strip() for el in elements_match.group(1).split(',') if
                                                    el.strip()]
            else:
                idea['key_elements_from_report'] = []

            if idea.get('title') and idea.get('logline'):
                generated_ideas.append(idea)
            else:
                logger.warning(f"Parsed idea block #{idx + 1} missing title or logline (attempt {attempt}).",
                               extra=extra_log_data)
                # 이 경우에도 전체 재시도 고려 가능 (위의 _is_valid_idea_xml 에서 이미 처리될 수 있음)

        logger.info(f"Parsed {len(generated_ideas)} comic ideas from LLM output (attempt {attempt}).",
                    extra=extra_log_data)
        return generated_ideas[:num_ideas] if generated_ideas else []

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        error_log = list(state.error_log or [])
        # 상태에서 재시도 카운트 가져오기 또는 초기화 (LangGraph 레벨에서 관리하는 것이 더 적합)
        # current_retry_attempt = state.get(f"{node_name}_retry_attempt", 1)

        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'node_name': node_name}
        logger.info(
            f"Entering node. Input: {summarize_for_logging(state.model_dump(exclude_none=True), fields_to_show=['original_query'])}",
            extra=extra)

        if not state.report_content:
            # ... (기존 오류 처리) ...
            return {"comic_ideas": [], "current_stage": "n08_scenario_generation", "error_log": error_log}

        comic_ideas_list = []
        try:
            processed_content = await self._preprocess_report_content(state.report_content, trace_id, extra)
            if not processed_content.strip() or len(processed_content) < 50:
                # ... (기존 오류 처리) ...
                return {"comic_ideas": [], "current_stage": "n08_scenario_generation", "error_log": error_log}

            extracted_insights = await self._extract_key_insights_sllm(processed_content, trace_id, extra)

            original_query = state.original_query or "N/A"
            refined_intent = state.query_context.get("refined_intent",
                                                     original_query) if state.query_context else original_query

            comic_ideas_list = await self._generate_comic_ideas_sllm(
                processed_content, extracted_insights, original_query, refined_intent,
                MAX_IDEAS_TO_GENERATE, config, trace_id, extra
            )

            if not comic_ideas_list:
                logger.warning("No comic ideas were generated by the LLM after all attempts.", extra=extra)

            # (이하 기존 로깅 및 반환 로직)
            # ...
            update_dict = {
                "comic_ideas": comic_ideas_list,
                "current_stage": "n08_scenario_generation",
                "error_log": error_log
            }
            logger.info(f"Exiting node. Generated {len(comic_ideas_list)} comic ideas.", extra=extra)
            return update_dict

        except Exception as e:
            # ... (기존 예외 처리) ...
            return {"comic_ideas": [], "error_log": error_log, "current_stage": "n08_scenario_generation",
                    "error_message": str(e)}