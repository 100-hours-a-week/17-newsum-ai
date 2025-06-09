# ai/app/nodes_v2/n08_scenario_generation_node.py
import json
import re  # 필요시 사용 (현재 코드에서는 직접 사용되지 않음)
from datetime import datetime, timezone  # <think> 태그 저장 시 timestamp용
from typing import Any, Dict, List, Optional

from app.workflows.state_v2 import WorkflowState, ScenarioSection  # ScenarioSection 임포트
from app.utils.logger import get_logger, summarize_for_logging  # summarize_for_logging 사용
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)

# 이 노드의 워크플로우 내 순서 정의
NODE_ORDER = 11


class N08ScenarioGenerationNode:
    """
    (업그레이드됨) N07에서 최종 선정된 풍자 아이디어를 바탕으로,
    썸네일 이미지 프롬프트와 4컷 웹툰의 패널별 상세 시나리오를 JSON으로 생성합니다.
    LLM의 <think> 태그 내용을 별도 저장하고, state_v2.py에 맞춰 상태를 관리합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_system_prompt(self) -> str:
        """
        4컷 만화 시나리오 및 썸네일 생성을 위한 LLM 시스템 프롬프트 (영문).
        JSON 스키마를 포함하여 LLM이 정확한 형식으로 응답하도록 유도합니다.
        """
        # LLM이 반환할 JSON 스키마 (N08a가 사용할 필드 포함)
        panel_detail_schema = {
            "panel_number": "<int, e.g., 1>",
            "scene_identifier": "<string, e.g., S01P01>",
            "setting": "<In English: Detailed description of the background and location.>",
            "characters_present": "<In English: Who is in the scene, their appearance, key expressions.>",
            "camera_shot_and_angle": "<In English: e.g., close-up, wide_shot, low_angle, high_angle.>",
            "key_actions_or_events": "<In English: What is happening in this panel.>",
            "lighting_and_atmosphere": "<In English: e.g., bright_daylight, gloomy_night, comedic, tense.>",
            "dialogue_summary_for_image_context": "<In English: A brief summary of dialogue if it directly informs visual elements (e.g., character pointing angrily while shouting). Not the full dialogue text.>",
            "visual_effects_or_key_props": "<In English: e.g., speed_lines, sweat_drop, a specific symbolic object.>",
            "image_style_notes_for_scene": "<In English: Brief notes on desired visual style for this panel, if any (e.g., retro_cartoon, detailed_realism, chibi_style).>",
            "final_image_prompt": "<In English: A consolidated, rich, and descriptive prompt for generating this panel's image, combining all relevant visual elements above. This will be the primary input for the image generation model.>"
        }

        schema_example = {
            "thumbnail_image_prompt": "<In English: A compelling and concise prompt for the overall comic thumbnail image, capturing the core theme or a key moment.>",
            "panels": [panel_detail_schema] * 4  # 4개 패널 예시 (실제로는 각기 다른 내용)
        }
        json_schema_str = json.dumps(schema_example, indent=2)

        # LLM 지시사항 (영문)
        return (
            "You are a professional webtoon scenario writer and an expert AI image prompt engineer.\n"
            "Your task is to take a detailed satirical comic idea (provided by the user) and adapt it into a 4-panel webtoon scenario, along with a suitable thumbnail image prompt.\n"
            "All parts of your response intended for image generation (e.g., thumbnail_prompt, and fields within panels like 'setting', 'characters_present', 'final_image_prompt', etc.) MUST be in ENGLISH.\n"
            "The 'final_image_prompt' for each panel should be particularly descriptive and ready for an image generation model.\n"
            "Ensure the satirical tone and core message of the input idea are effectively translated into the 4 panels.\n"
            "Return ONLY a single, valid JSON object that strictly adheres to the schema provided below. Do NOT include any extraneous text, <think> tags, or comments outside this JSON structure.\n\n"
            "# Output JSON Schema:\n"
            f"{json_schema_str}"
        )

    def _build_user_prompt(self, selected_final_idea: Dict[str, Any], report_summary_text: str,
                           target_audience: str) -> str:
        """
        선택된 최종 아이디어와 추가 컨텍스트를 바탕으로 LLM 사용자 프롬프트를 구성합니다 (영문).
        selected_final_idea는 N07의 FinalSatiricalIdea 구조를 따르는 딕셔너리입니다.
        """
        title = selected_final_idea.get("title_concept", "Untitled Satirical Idea")
        detailed_content = selected_final_idea.get("detailed_content", "No detailed content provided for this idea.")
        # N07의 다른 필드들(techniques, appeal_points, ethical_review)도 필요시 프롬프트에 추가 가능
        applied_techniques = ", ".join(selected_final_idea.get("applied_satire_techniques", ["N/A"]))
        appeal_points = selected_final_idea.get("expected_appeal_points", "N/A")

        # LLM 지시사항 (영문)
        return f"""
# Detailed Satirical Comic Idea to Adapt (from N07 - mostly in Korean, translate visual elements to English for prompts):
## Title/Concept:
{title}

## Detailed Content / Synopsis for Adaptation:
{detailed_content}

## Key Satire Techniques Applied in this Idea:
{applied_techniques}

## Expected Appeal of this Idea:
{appeal_points}

# Supporting Context (Summary of Original Report - for background understanding):
{report_summary_text[:1500]} # 컨텍스트 길이 제한

# Target Audience for the Comic:
{target_audience}

# Your Task:
Based on the detailed satirical comic idea and its context above, generate the 'thumbnail_image_prompt' (in English) and the 4 'panels' (with all image-related fields in English) as specified in the system prompt's JSON schema.
The 4 panels should tell a coherent satirical story derived from the 'Detailed Content / Synopsis for Adaptation'.
Focus on creating vivid, actionable 'final_image_prompt' for each panel.
"""

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        scenario_sec = state.scenario  # ScenarioSection에 결과 저장
        idea_sec = state.idea  # N07 결과 사용
        report_sec = state.report  # 보고서 요약 사용 (컨텍스트용)
        config_sec_dict = state.config.config or {}

        work_id = meta_sec.work_id
        node_name = self.__class__.__name__
        extra_log_base = {"work_id": work_id, "node_name": node_name, "node_order": NODE_ORDER}

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info(f"{node_name} 진입: N07 아이디어 기반 시나리오 생성 시작.", extra=extra_log_base)

        # 1. N07에서 생성된 최종 아이디어 목록에서 아이디어 선택
        #    state.scenario.selected_comic_idea_for_scenario는 N07 아이디어 리스트의 인덱스를 의미한다고 가정.
        #    실제 선택 로직은 UI나 다른 로직에 의해 이 필드에 인덱스가 설정된다고 가정.
        #    여기서는 첫 번째 아이디어를 사용하거나, 인덱스가 있으면 해당 아이디어를 사용.
        final_ideas_list = idea_sec.final_comic_ideas
        selected_idea_index = scenario_sec.selected_comic_idea_for_scenario or 0  # 기본값 0 (첫번째 아이디어)

        if not final_ideas_list or not isinstance(final_ideas_list, list) or \
                not (0 <= selected_idea_index < len(final_ideas_list)):
            msg = f"시나리오 생성을 위한 유효한 최종 아이디어(final_comic_ideas)가 없거나 선택된 인덱스({selected_idea_index})가 잘못되었습니다."
            logger.error(msg, extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"scenario": scenario_sec.model_dump(exclude_unset=True), "meta": meta_sec.model_dump()}

        selected_idea_for_scenario = final_ideas_list[selected_idea_index]
        if not isinstance(selected_idea_for_scenario, dict) or not selected_idea_for_scenario.get("detailed_content"):
            msg = f"선택된 아이디어(인덱스: {selected_idea_index})가 유효한 딕셔너리 형식이 아니거나 'detailed_content'가 없습니다."
            logger.error(msg, extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"scenario": scenario_sec.model_dump(exclude_unset=True), "meta": meta_sec.model_dump()}

        logger.info(f"선택된 아이디어(N07): '{selected_idea_for_scenario.get('title_concept', '제목없음')}'", extra=extra_log_base)

        # 2. LLM 프롬프트에 사용할 컨텍스트 준비 (보고서 요약 등)
        #    N06B의 contextual_summary를 사용하거나, N05의 report_content를 요약.
        #    여기서는 N06B의 contextual_summary를 우선 사용.
        report_summary_for_context = report_sec.contextual_summary
        if not report_summary_for_context and report_sec.report_content:  # 요약이 없으면 보고서 내용 일부 사용
            # 간단히 앞부분만 사용 (실제로는 N06B와 유사한 요약 로직 또는 LLM 호출 필요 가능성)
            temp_text = re.sub(r'<[^>]+>', ' ', report_sec.report_content)
            report_summary_for_context = ' '.join(temp_text.split())[:2000]  # 길이 제한
        elif not report_summary_for_context:
            report_summary_for_context = "배경 컨텍스트 정보가 제공되지 않았습니다."

        target_audience_from_config = config_sec_dict.get("target_audience", "general_public")

        # 3. LLM 프롬프트 구성 및 호출
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(selected_idea_for_scenario, report_summary_for_context,
                                              target_audience_from_config)

        request_id_scenario_gen = f"{work_id}_{node_name}_ScenarioGen"
        llm_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        # 재시도 로직은 포함하지 않음 (LLMService 또는 외부에서 처리 가정)
        # 또는 여기에 간단한 재시도 로직 추가 가능 (예: 패널 수가 4개가 아닌 경우)
        logger.debug("4컷 만화 시나리오 및 썸네일 프롬프트 생성 LLM 호출 시작...", extra=extra_log_base)
        llm_response = await self.llm_service.generate_text(
            messages=llm_messages,
            request_id=request_id_scenario_gen,
            temperature=0.4,  # 시나리오 생성이므로 약간의 창의성 허용, 너무 높지 않게
            max_tokens=3500  # 4개 패널 상세 묘사 및 JSON 구조 포함하므로 충분히 할당
        )

        cleaned_llm_output = llm_response.get("generated_text", "")
        think_content = llm_response.get("think_content")

        if think_content:  # <think> 내용 저장
            meta_sec.llm_think_traces.append({
                "node_name": node_name, "request_id": request_id_scenario_gen,
                "timestamp": datetime.now(timezone.utc).isoformat(), "log_content": think_content
            })
            logger.debug(f"LLM <think> 내용 저장됨 (Request ID: {request_id_scenario_gen})", extra=extra_log_base)

        if not cleaned_llm_output or llm_response.get("error"):
            error_detail = llm_response.get("error", "LLM이 시나리오 생성에 실패했거나 빈 응답을 반환했습니다.")
            logger.error(f"시나리오 생성 LLM 호출 실패: {error_detail}", extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"scenario": scenario_sec.model_dump(exclude_unset=True), "meta": meta_sec.model_dump()}

        logger.debug(f"시나리오 생성 LLM 원본 응답 (정리 후): {summarize_for_logging(cleaned_llm_output, 400)}",
                     extra=extra_log_base)

        # 4. LLM JSON 응답 파싱 및 검증
        try:
            parsed_scenario_json = extract_json(cleaned_llm_output)
            if not isinstance(parsed_scenario_json, dict):
                raise ValueError("파싱된 시나리오 결과가 JSON 객체(딕셔너리)가 아닙니다.")

            thumbnail_prompt_from_llm = parsed_scenario_json.get("thumbnail_image_prompt")
            panels_from_llm = parsed_scenario_json.get("panels")

            if not isinstance(thumbnail_prompt_from_llm, str) or not thumbnail_prompt_from_llm.strip():
                raise ValueError("LLM 응답에 유효한 'thumbnail_image_prompt' (문자열)가 없습니다.")
            if not isinstance(panels_from_llm, list) or len(panels_from_llm) != 4:
                raise ValueError(
                    f"LLM 응답의 'panels' 필드가 리스트가 아니거나, 패널 수가 4개가 아닙니다 (실제: {len(panels_from_llm) if isinstance(panels_from_llm, list) else type(panels_from_llm)}).")

            # 각 패널의 필수 필드 검증 (선택적, 예시: final_image_prompt)
            for i, panel_data in enumerate(panels_from_llm):
                if not isinstance(panel_data, dict) or not panel_data.get("final_image_prompt"):
                    raise ValueError(f"패널 {i + 1}의 데이터가 유효한 딕셔너리가 아니거나 'final_image_prompt'가 없습니다.")

            logger.info("시나리오 생성 LLM JSON 파싱 및 기본 검증 성공.", extra=extra_log_base)
        except Exception as e:
            logger.error(f"시나리오 생성 LLM JSON 파싱 또는 검증 실패: {e}. 원본(정리후): {cleaned_llm_output}", extra=extra_log_base,
                         exc_info=True)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"scenario": scenario_sec.model_dump(exclude_unset=True), "meta": meta_sec.model_dump()}

        # 5. 상태 업데이트
        scenario_sec.thumbnail_image_prompt = thumbnail_prompt_from_llm  # N08a가 사용할 필드
        scenario_sec.comic_scenarios = panels_from_llm  # N08a가 사용할 필드
        # scenario_sec.comic_scenario_thumbnail 필드는 N08에서 원본 '내용'을 저장하는 용도였으나,
        # N07의 결과가 이미 구체적이므로, N08에서는 thumbnail_image_prompt를 직접 생성.
        # 만약 N07 아이디어의 title_concept 등을 저장하고 싶다면 여기에 저장 가능.
        scenario_sec.comic_scenario_thumbnail = selected_idea_for_scenario.get("title_concept", "썸네일 원본 설명 없음")  # 예시

        meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"
        logger.info(f"{node_name} 완료: {len(panels_from_llm)}개 패널 시나리오 및 썸네일 프롬프트 생성 완료.", extra=extra_log_base)

        return {"scenario": scenario_sec.model_dump(exclude_unset=True), "meta": meta_sec.model_dump()}