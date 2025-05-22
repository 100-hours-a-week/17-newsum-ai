# ai/app/nodes/n08_scenario_generation_node.py
import asyncio
import traceback
import uuid
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.config.settings import Settings
from app.services.llm_service import LLMService  # 타입 힌팅을 위해 기존 LLMService 사용

logger = get_logger(__name__)
settings = Settings()
MAX_RETRY_ATTEMPTS_PANEL = 1  # 패널 생성 재시도 (디버깅 시 0으로 설정하여 빠른 실패 확인 가능)


class N08ScenarioGenerationNode:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        logger.info("N08ScenarioGenerationNode initialized with LLMService.")

    def _is_valid_scene_data(self, parsed_data: Dict[str, Any], panel_number: int, extra_log_data: dict) -> bool:
        """파싱된 장면 데이터의 기본 유효성 검사"""
        if not parsed_data or not isinstance(parsed_data, dict):
            logger.warning(f"Validation failed for panel {panel_number}: Parsed_data is None or not a dict.",
                           extra=extra_log_data)
            return False

        required_fields = [
            "panel_number", "scene_identifier", "setting", "characters_present",
            "camera_shot_and_angle", "key_actions_or_events", "lighting_and_atmosphere",
            "dialogue_summary_for_image_context", "visual_effects_or_key_props",
            "image_style_notes_for_scene", "final_image_prompt"
        ]

        for field in required_fields:
            field_value = parsed_data.get(field)
            if field_value is None or (
                    isinstance(field_value, str) and (field_value.lower() == "none" or not field_value.strip())):
                logger.warning(
                    f"Validation failed for panel {panel_number}: Required field '{field}' is missing, 'None', or empty. Value: '{field_value}'",
                    extra=extra_log_data)
                return False

        if not isinstance(parsed_data.get("panel_number"), int) or parsed_data.get("panel_number") != panel_number:
            logger.warning(
                f"Validation failed for panel {panel_number}: 'panel_number' mismatch or not an int (Value: {parsed_data.get('panel_number')}, Expected: {panel_number}).",
                extra=extra_log_data)
            return False

        fip = parsed_data.get("final_image_prompt", "")
        if len(fip) < 20:  # 임의의 최소 길이
            logger.warning(
                f"Validation failed for panel {panel_number}: 'FINAL_IMAGE_PROMPT' is too short (Length: {len(fip)}, Content: '{fip[:50]}...').",
                extra=extra_log_data)
            return False

        logger.debug(f"Panel {panel_number} data is valid.", extra=extra_log_data)
        return True

    def _construct_scene_generation_messages(
            self, comic_idea: dict, panel_number: int, previous_summaries: str,
            thumbnail_prompt_example: str, config: dict, attempt: int = 1, extra_log_data: Optional[dict] = None
    ) -> List[Dict[str, str]]:
        title = comic_idea.get("title", "Untitled Comic Idea")  # Default title if missing
        summary_or_logline = comic_idea.get("summary") or comic_idea.get("logline",
                                                                         "No summary or logline provided.")  # Default text
        genre = comic_idea.get("genre", "general")  # Default genre

        logger.debug(f"Constructing messages for panel {panel_number}, attempt {attempt}. Idea title: '{title}'",
                     extra=extra_log_data)

        system_prompt = f"""You are a professional webtoon scenario writer, acting as a meticulous visual director for Panel {panel_number}.
Your task is to generate a detailed scenario adhering to all instructions.

KEY INSTRUCTIONS FOR VISUAL DESCRIPTIONS (especially for FINAL_IMAGE_PROMPT and related fields):
- Use concrete, unambiguous, visually descriptive terms. Avoid abstract concepts.
- Prioritize vivid adjectives and specific nouns for AI image generator (Flux1).
- Ensure descriptions are directly translatable into visual elements.

STUDY `thumbnail_prompt_example` for its descriptive quality and method. Your `FINAL_IMAGE_PROMPT` must emulate this.

OUTPUT FORMAT (Strictly adhere. Fill all fields. If a field is not applicable, write 'None' BUT ensure it's not empty for critical fields like FINAL_IMAGE_PROMPT. No extra text/markdown):

PANEL_NUMBER: {panel_number}
SCENE_IDENTIFIER: S{panel_number:02d}P{panel_number:02d}
SETTING: (3-5 key visual details: textures, colors, architecture, objects. Focus on what is *seen*.)
CHARACTERS_PRESENT: (Visual features: clothing, hair, accessories; current pose; specific facial expression for this panel.)
CAMERA_SHOT_AND_ANGLE: (Distinct shot e.g., 'close-up', 'medium'; angle e.g., 'eye-level', 'low angle'. State *why* if unusual.)
KEY_ACTIONS_OR_EVENTS: (Visually unfolding moments. Most visually impactful part? What would viewer *see*?)
LIGHTING_AND_ATMOSPHERE: (Lighting e.g., 'bright sun', 'moody twilight'; visual atmosphere e.g., 'tense', 'joyful'. How do they *visually* contribute?)
DIALOGUE_SUMMARY_FOR_IMAGE_CONTEXT: (Brief summary of dialogue/text influencing expression/action.)
VISUAL_EFFECTS_OR_KEY_PROPS: (Special visual effects or key props and their appearance.)
IMAGE_STYLE_NOTES_FOR_SCENE: (1-2 specific artistic style keywords for Flux1 e.g., 'photorealistic', 'detailed anime style', aligned with genre: '{genre}'.)
FINAL_IMAGE_PROMPT: (CRITICAL: Synthesize ALL visual details from above fields into a single, compelling, descriptive paragraph for Flux1. Rich in visual detail, flow naturally, paint a clear picture. Do not just list fields; weave them into a cohesive visual narrative. THIS MUST NOT BE 'None' OR EMPTY.)
"""
        user_prompt = f"Generate Panel {panel_number} for a 4-panel webtoon.\n\n"
        user_prompt += f"Comic Idea:\nTitle: \"{title}\"\nLogline/Summary: \"{summary_or_logline}\"\nGenre: \"{genre}\"\n\n"
        if previous_summaries and previous_summaries != "This is the first panel.":
            user_prompt += f"Context from Previous Panels:\n{previous_summaries}\n\n"
        user_prompt += f"Thumbnail Prompt Example (learn from its quality):\n{thumbnail_prompt_example}\n\n"
        if attempt > 1:
            user_prompt = f"[Attempt {attempt}/{MAX_RETRY_ATTEMPTS_PANEL + 1} for Panel {panel_number}] Previous attempt to generate panel details was incomplete or did not follow the specified format/quality. Please ensure all fields are filled correctly and FINAL_IMAGE_PROMPT is a detailed, synthesized paragraph of at least 20 words.\n\n" + user_prompt
        user_prompt += f"Generate Panel {panel_number} details now, adhering to all system instructions and output format."

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _summarize_previous_scenarios(self, scenarios: list) -> str:
        summary = ""
        if not scenarios:
            return "This is the first panel."
        for sc_idx, sc in enumerate(scenarios):
            panel_num_text = sc.get('panel_number', f'Unknown Panel {sc_idx + 1}')
            summary += (
                f"--- Panel {panel_num_text} Summary ---\n"
                f"Setting: {sc.get('setting', 'N/A')}\n"
                f"Characters & Poses: {sc.get('characters_present', 'N/A')}\n"
                f"Key Actions/Events: {sc.get('key_actions_or_events', 'N/A')}\n"
                f"Dialogue Context: {sc.get('dialogue_summary_for_image_context', 'N/A')}\n"
                f"Atmosphere: {sc.get('lighting_and_atmosphere', 'N/A')}\n"
                f"-------------------------\n"
            )
        return summary.strip()

    def _parse_scene_response(self, llm_text_response: str, panel_number: int, extra_log_data: dict) -> dict:
        result = {}
        # 필드명 유사 매칭을 위한 후보군 정의 (alias 확장)
        field_aliases = {
            "panel_number": ["panel_number", "panel num", "panel no", "panel #", "panel", "number"],
            "scene_identifier": ["scene_identifier", "scene id", "sceneidentifier", "scene", "identifier", "scene code"],
            "setting": ["setting", "scene setting", "background", "location", "environment", "place", "context"],
            "characters_present": ["characters_present", "characters & poses", "characters", "character", "poses", "people", "figures", "cast", "who is present", "who appears"],
            "camera_shot_and_angle": [
                "camera_shot_and_angle", "camera shot and angle", "camera shot", "shot", "angle", "perspective", "viewpoint", "view", "framing", "composition", "camera perspective", "camera angle", "camera viewpoint", "shot type", "frame", "camera framing"
            ],
            "key_actions_or_events": ["key_actions_or_events", "key actions/events", "key actions", "events", "actions", "main action", "main event", "what happens", "activity", "action/event", "event/action", "scene action", "scene event"],
            "lighting_and_atmosphere": ["lighting_and_atmosphere", "lighting & atmosphere", "lighting", "atmosphere", "mood", "ambience", "light", "scene lighting", "scene mood", "scene atmosphere", "illumination"],
            "dialogue_summary_for_image_context": ["dialogue_summary_for_image_context", "dialogue context", "dialogue summary", "dialogue", "conversation", "speech", "text", "spoken words", "what is said", "quote", "line", "lines", "caption"],
            "visual_effects_or_key_props": ["visual_effects_or_key_props", "visual effects or key props", "visual effects", "key props", "props", "special effects", "fx", "important props", "notable props", "notable objects", "key objects", "special visuals", "visual highlight", "visual element"],
            "image_style_notes_for_scene": ["image_style_notes_for_scene", "image style notes", "style notes", "style", "art style", "visual style", "drawing style", "render style", "aesthetic", "artistic style", "scene style", "style keyword", "style guide"],
            "final_image_prompt": ["final_image_prompt", "final image prompt", "image prompt", "prompt", "full prompt", "description", "image description", "scene description", "final prompt", "visual prompt", "visual description"]
        }
        # 필드별로 가장 먼저 매칭되는 것을 우선 추출
        for field, aliases in field_aliases.items():
            found = False
            for alias in aliases:
                # re: 대소문자, 공백, 언더스코어, 하이픈, 콜론 등 유연하게 매칭
                pattern = rf"{alias.replace('_', '[_\- ]*')}\s*[:：]\s*((?:.|\n)*?)(?=\n[A-Z_\- ]+\s*[:：]|\Z)"
                match = re.search(pattern, llm_text_response, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    result[field] = value if value.strip() else "None"
                    found = True
                    break
            if not found:
                result[field] = "None"
                logger.debug(
                    f"Parsing: Field '{field}' not found in response for panel {panel_number}. Snippet: {llm_text_response[:100]}",
                    extra=extra_log_data)
        # panel_number int 변환 보완
        try:
            parsed_pn = result.get("panel_number")
            if isinstance(parsed_pn, str) and parsed_pn.isdigit():
                result["panel_number"] = int(parsed_pn)
            elif not isinstance(parsed_pn, int):
                result["panel_number"] = panel_number
        except Exception:
            result["panel_number"] = panel_number
        if result.get("scene_identifier") == "None" and isinstance(result.get("panel_number"), int):
            pn_val = result["panel_number"]
            result["scene_identifier"] = f"S{pn_val:02d}P{pn_val:02d}"
        # 파싱 결과 요약 로그 (dict comprehension을 f-string 밖에서 처리)
        preview = {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in result.items()}
        logger.debug(
            f"Parsed scene response for panel {panel_number}: {preview}",
            extra=extra_log_data)
        return result

    async def _generate_thumbnail_prompt_example(
            self, comic_idea: dict, trace_id: str, extra_log_data: dict, attempt: int = 1
    ) -> str:
        idea_title = comic_idea.get("title", "Untitled Comic Idea")
        idea_summary_or_logline = comic_idea.get("summary") or comic_idea.get("logline", "No summary/logline.")
        idea_genre = comic_idea.get("genre", "general")

        system_prompt = """You are an expert AI image prompt engineer. Generate a single, compelling, visually rich thumbnail image prompt in English for a webcomic. This prompt serves as a *quality example* for other detailed image prompts. Focus on dynamic poses, hints of conflict/mystery, vibrant colors, and cinematic quality for a webtoon cover. Provide *only* the final image generation prompt string, no fluff or markdown. Ensure the prompt is at least 15 words long."""
        user_prompt = f"""Webcomic Idea:\nTitle: \"{idea_title}\"\nGenre: \"{idea_genre}\"\nCore Summary/Logline: \"{idea_summary_or_logline}\"\n\nGenerate the thumbnail image prompt example now:"""
        if attempt > 1:
            user_prompt = f"[Attempt {attempt}/{MAX_RETRY_ATTEMPTS_PANEL + 1}] Previous attempt to generate thumbnail example was not ideal (e.g. too short or generic). Please ensure a high-quality, descriptive prompt of at least 15 words.\n\n" + user_prompt

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        logger.info(f"Requesting thumbnail prompt example (attempt {attempt}) for idea: '{idea_title}'",
                    extra=extra_log_data)
        fallback_prompt = f"Fallback example: Epic digital painting of '{idea_title}', showcasing {idea_genre} elements with dramatic action and cinematic lighting, webtoon cover art style."

        try:
            llm_response = await self.llm_service.generate_text(messages=messages, max_tokens=250,
                                                                temperature=0.75)  # Increased max_tokens for potentially longer good examples

            if llm_response.get("error") or not llm_response.get("generated_text"):
                error_msg = llm_response.get('error', 'No text returned for thumbnail example')
                logger.error(f"LLMService error for thumbnail example (attempt {attempt}): {error_msg}",
                             extra=extra_log_data)
                if attempt <= MAX_RETRY_ATTEMPTS_PANEL:
                    return await self._generate_thumbnail_prompt_example(comic_idea, trace_id, extra_log_data,
                                                                         attempt + 1)
                logger.warning(f"Max retries for thumbnail example. Using fallback.", extra=extra_log_data)
                return fallback_prompt

            generated_prompt = llm_response.get("generated_text", "").strip()
            # Strip common chatty prefixes/suffixes if LLM adds them
            generated_prompt = re.sub(
                r"^(here's the prompt:|the prompt is:|thumbnail prompt:|image prompt:|prompt:)\s*", "",
                generated_prompt, flags=re.IGNORECASE)
            generated_prompt = generated_prompt.strip("` \n")

            if not generated_prompt or len(generated_prompt.split()) < 10:  # Stricter validation for length
                logger.warning(
                    f"Generated thumbnail example too short or empty after stripping (attempt {attempt}, length {len(generated_prompt.split())} words): '{generated_prompt}'. Retrying.",
                    extra=extra_log_data)
                if attempt <= MAX_RETRY_ATTEMPTS_PANEL:
                    return await self._generate_thumbnail_prompt_example(comic_idea, trace_id, extra_log_data,
                                                                         attempt + 1)
                logger.warning(f"Max retries for short thumbnail example. Using fallback.", extra=extra_log_data)
                return fallback_prompt

            logger.info(f"Generated thumbnail prompt example (attempt {attempt}): '{generated_prompt[:150]}...'",
                        extra=extra_log_data)
            return generated_prompt
        except Exception as e:
            logger.exception(f"LLM call error for thumbnail example (attempt {attempt}, idea: '{idea_title}'): {e}",
                             extra=extra_log_data)
            if attempt <= MAX_RETRY_ATTEMPTS_PANEL:
                return await self._generate_thumbnail_prompt_example(comic_idea, trace_id, extra_log_data, attempt + 1)
            logger.warning(f"Exception and max retries for thumbnail example. Using fallback.", extra=extra_log_data)
            return fallback_prompt

    async def _generate_single_panel_scenario(
            self, comic_idea: dict, panel_number: int, previous_summaries: str,
            thumbnail_prompt_example: str, config: dict, trace_id: str, extra_log_data: dict,
            attempt: int = 1
    ) -> Dict[str, Any]:  # Always return a dict
        logger.debug(f"Generating panel {panel_number}, attempt {attempt}", extra=extra_log_data)
        messages_for_llm = self._construct_scene_generation_messages(
            comic_idea=comic_idea, panel_number=panel_number,
            previous_summaries=previous_summaries,
            thumbnail_prompt_example=thumbnail_prompt_example,
            config=config, attempt=attempt, extra_log_data=extra_log_data
        )

        raw_text_response = ""
        try:
            llm_response = await self.llm_service.generate_text(
                messages=messages_for_llm, max_tokens=800, temperature=0.7
            )

            if llm_response.get("error") or not llm_response.get("generated_text"):
                error_detail = llm_response.get('error', 'No text returned from LLM')
                logger.error(f"LLMService error for scene {panel_number} (attempt {attempt}): {error_detail}",
                             extra=extra_log_data)
                raw_text_response = f"LLM Error: {error_detail}"
                if attempt <= MAX_RETRY_ATTEMPTS_PANEL:
                    return await self._generate_single_panel_scenario(comic_idea, panel_number, previous_summaries,
                                                                      thumbnail_prompt_example, config, trace_id,
                                                                      extra_log_data, attempt + 1)
                return {"panel_data": None, "raw_text": raw_text_response, "error": error_detail}

            generated_text = llm_response.get("generated_text", "")
            raw_text_response = generated_text
            logger.debug(f"Raw LLM response for panel {panel_number} (attempt {attempt}):\n{generated_text}",
                         extra=extra_log_data)

            if not generated_text.strip():
                logger.warning(f"LLM returned empty text for scene {panel_number} (attempt {attempt}).",
                               extra=extra_log_data)
                if attempt <= MAX_RETRY_ATTEMPTS_PANEL:
                    return await self._generate_single_panel_scenario(comic_idea, panel_number, previous_summaries,
                                                                      thumbnail_prompt_example, config, trace_id,
                                                                      extra_log_data, attempt + 1)
                return {"panel_data": None, "raw_text": raw_text_response, "error": "LLM returned empty text."}

            parsed_scene_data = self._parse_scene_response(generated_text, panel_number, extra_log_data)

            if not self._is_valid_scene_data(parsed_scene_data, panel_number, extra_log_data):
                logger.warning(
                    f"Parsed scene data for panel {panel_number} is invalid (attempt {attempt}). Will retry if possible.",
                    extra=extra_log_data)
                if attempt <= MAX_RETRY_ATTEMPTS_PANEL:
                    return await self._generate_single_panel_scenario(comic_idea, panel_number, previous_summaries,
                                                                      thumbnail_prompt_example, config, trace_id,
                                                                      extra_log_data, attempt + 1)
                # Max retries reached, return partially parsed data but mark error
                logger.error(
                    f"Max retries reached for panel {panel_number}. Returning potentially invalid data with error flag.",
                    extra=extra_log_data)
                return {"panel_data": parsed_scene_data, "raw_text": raw_text_response,
                        "error": "Invalid scene data after max retries."}

            logger.info(f"Successfully generated and validated panel {panel_number} (attempt {attempt}).",
                        extra=extra_log_data)
            return {"panel_data": parsed_scene_data, "raw_text": raw_text_response, "error": None}

        except Exception as e:
            error_detail_exc = f"Exception during scene {panel_number} (attempt {attempt}): {type(e).__name__} - {e}"
            logger.exception(error_detail_exc, extra=extra_log_data)
            if attempt <= MAX_RETRY_ATTEMPTS_PANEL:
                return await self._generate_single_panel_scenario(comic_idea, panel_number, previous_summaries,
                                                                  thumbnail_prompt_example, config, trace_id,
                                                                  extra_log_data, attempt + 1)
            return {"panel_data": None, "raw_text": raw_text_response or f"Exception: {str(e)}", "error": str(e),
                    "exception_trace": traceback.format_exc()}

    def _create_error_panel_data(self, panel_number: int, error_message: str) -> Dict[str, Any]:
        """Creates a placeholder dictionary for a panel when generation fails."""
        return {
            "panel_number": panel_number,
            "scene_identifier": f"S{panel_number:02d}P{panel_number:02d}_ERROR",
            "setting": "Error during generation",
            "characters_present": "Error during generation",
            "camera_shot_and_angle": "Error during generation",
            "key_actions_or_events": "Error during generation",
            "lighting_and_atmosphere": "Error during generation",
            "dialogue_summary_for_image_context": "Error during generation",
            "visual_effects_or_key_props": "Error during generation",
            "image_style_notes_for_scene": "Error during generation",
            "final_image_prompt": f"Error: {error_message}. Panel generation failed."
        }

    async def run(self, state: WorkflowState) -> dict:
        """
        시나리오 생성 결과를 scenario Section에, stage/error 등은 meta Section에 직접 할당합니다.
        반환값은 state_v2 구조에 맞게 {"scenario": ..., "meta": ...} 형태로 반환합니다.
        """
        meta = state.meta
        scenario_sec = state.scenario
        idea_sec = state.idea
        config_sec = state.config
        node_name = self.__class__.__name__
        trace_id = meta.trace_id
        comic_id = meta.comic_id
        config = config_sec.config or {}
        current_node_error_log = []
        previous_error_log = list(meta.error_log or [])
        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'node_name': node_name, 'retry_flow': 0}
        logger.info(f"Entering node {node_name}. Trace ID: {trace_id}", extra=extra)
        selected_idea = scenario_sec.selected_comic_idea_for_scenario
        if not selected_idea or not isinstance(selected_idea, dict):
            if idea_sec.comic_ideas and isinstance(idea_sec.comic_ideas, list) and idea_sec.comic_ideas:
                selected_idea = idea_sec.comic_ideas[0]
                logger.info(
                    f"Using first comic idea as selected_comic_idea_for_scenario was not set: '{selected_idea.get('title', 'N/A')}'",
                    extra=extra)
            else:
                error_msg = "No valid comic idea found in state for scenario generation (selected_comic_idea_for_scenario is missing and comic_ideas is empty/invalid)."
                logger.error(error_msg, extra=extra)
                current_node_error_log.append(
                    {"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
                final_error_log = previous_error_log + current_node_error_log
                meta.current_stage = "ERROR"
                meta.error_log = final_error_log
                meta.error_message = f"{node_name}: {error_msg}"
                scenario_sec.comic_scenarios = []
                scenario_sec.thumbnail_image_prompt = ""
                return {"scenario": scenario_sec.model_dump(), "meta": meta.model_dump()}
        else:
            logger.info(f"Successfully retrieved selected_comic_idea: '{selected_idea.get('title', 'N/A')}'",
                        extra=extra)
        scenario_sec.selected_comic_idea_for_scenario = selected_idea
        thumbnail_prompt_example_final = await self._generate_thumbnail_prompt_example(selected_idea, trace_id, extra)
        logger.info(f"Thumbnail prompt example to be used: '{thumbnail_prompt_example_final}'", extra=extra)
        if "Fallback example:" in thumbnail_prompt_example_final:
            current_node_error_log.append({"stage": f"{node_name}._generate_thumbnail_prompt_example",
                                           "warning": "Used fallback thumbnail prompt example.",
                                           "value_used": thumbnail_prompt_example_final,
                                           "timestamp": datetime.now(timezone.utc).isoformat()})
        scenarios = []
        scenarios_raw_texts = []
        num_panels_to_generate = 4
        all_panels_valid = True
        for i in range(num_panels_to_generate):
            panel_number = i + 1
            panel_extra_log = {**extra, 'panel_number_flow': panel_number}
            previous_scenarios_summary = self._summarize_previous_scenarios(scenarios)
            logger.info(f"Starting generation for panel {panel_number}/{num_panels_to_generate}", extra=panel_extra_log)
            panel_generation_result = await self._generate_single_panel_scenario(
                comic_idea=selected_idea, panel_number=panel_number,
                previous_summaries=previous_scenarios_summary,
                thumbnail_prompt_example=thumbnail_prompt_example_final,
                config=config, trace_id=trace_id, extra_log_data=panel_extra_log
            )
            scenarios_raw_texts.append(panel_generation_result.get("raw_text", "No raw text recorded."))
            panel_data = panel_generation_result.get("panel_data")
            panel_error = panel_generation_result.get("error")
            if panel_error or not panel_data:
                error_message_for_log = f"Failed to generate valid data for panel {panel_number}: {panel_error or 'Panel data was None'}"
                logger.error(error_message_for_log, extra=panel_extra_log)
                current_node_error_log.append({
                    "stage": f"{node_name}._generate_scene_{panel_number}",
                    "error": error_message_for_log,
                    "detail": panel_generation_result.get("exception_trace"),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                scenarios.append(self._create_error_panel_data(panel_number, panel_error or "Unknown generation error"))
                all_panels_valid = False
            else:
                logger.info(f"Successfully obtained panel_data for panel {panel_number}", extra=panel_extra_log)
                scenarios.append(panel_data)
        final_error_log = previous_error_log + current_node_error_log
        final_status = "N08_SCENARIO_GENERATION_COMPLETED"
        if not all_panels_valid or len(scenarios) < num_panels_to_generate:
            final_status = "N08_COMPLETED_WITH_PARTIAL_ERRORS"
        if not scenarios:
            final_status = "N08_SCENARIO_GENERATION_FAILED"
        scenario_sec.comic_scenarios = scenarios
        scenario_sec.thumbnail_image_prompt = thumbnail_prompt_example_final
        meta.current_stage = final_status
        meta.error_log = final_error_log
        logger.info(
            f"Exiting node {node_name}. Scenarios count: {len(scenarios)}. "
            f"First scenario FIP empty/error: {'N/A' if not scenarios else (not scenarios[0].get('final_image_prompt') or 'Error' in scenarios[0].get('final_image_prompt', ''))}. "
            f"Thumbnail prompt: '{thumbnail_prompt_example_final[:70]}...'. Status: {final_status}",
            extra=extra
        )
        if scenarios:
            preview = {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in scenarios[0].items()}
            logger.debug(f"First panel data being saved: {preview}", extra=extra)
        return {
            "scenario": scenario_sec.model_dump(),
            "meta": meta.model_dump(),
        }