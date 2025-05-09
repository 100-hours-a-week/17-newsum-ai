# ai/app/nodes/n08_scenario_generation_node.py
import asyncio
import traceback
import uuid
import json
import re  # 정규 표현식 파싱용
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.config.settings import Settings
from app.services.llm_service import LLMService  # 제공된 LLMService 사용

logger = get_logger(__name__)
settings = Settings()


class N08ScenarioGenerationNode:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        logger.info("N08ScenarioGenerationNode initialized with LLMService.")

    def _build_scene_prompt(self, comic_idea: dict, panel_number: int, previous_summaries: str, thumbnail_prompt_example: str) -> str:
        """
        Build an English prompt for LLM to generate a single scene, with context from previous scenes.
        """
        title = comic_idea.get("title", "Untitled Idea")
        summary = comic_idea.get("summary", "A fantastic adventure.")
        genre = comic_idea.get("genre", "adventure")
        prompt = (
            f"You are a professional webtoon scenario writer.\n\n"
            f"We are creating a 4-panel comic. Each panel should naturally connect to the previous one, forming a coherent and engaging short story.\n\n"
            f"Here is the comic idea:\n"
            f"Title: \"{title}\"\n"
            f"Summary: \"{summary}\"\n"
            f"Genre: \"{genre}\"\n\n"
        )
        if previous_summaries:
            prompt += f"Here are the previous panels:\n{previous_summaries}\n\n"
        prompt += (
            f"Now, generate the details for Panel {panel_number}.\n"
            f"All fields below must be filled. If a field is not applicable, write 'None'.\n"
            f"The FINAL_IMAGE_PROMPT must be as detailed, vivid, and visually rich as the following example:\n\n"
            f"Example:\n{thumbnail_prompt_example}\n\n"
            f"Output format (no extra text, no markdown):\n\n"
            f"PANEL_NUMBER: {panel_number}\n"
            f"SCENE_IDENTIFIER: S{panel_number:02d}P{panel_number:02d}\n"
            f"SETTING:\n"
            f"CHARACTERS_PRESENT:\n"
            f"CAMERA_SHOT_AND_ANGLE:\n"
            f"KEY_ACTIONS_OR_EVENTS:\n"
            f"LIGHTING_AND_ATMOSPHERE:\n"
            f"DIALOGUE_SUMMARY_FOR_IMAGE_CONTEXT:\n"
            f"VISUAL_EFFECTS_OR_KEY_PROPS:\n"
            f"IMAGE_STYLE_NOTES_FOR_SCENE:\n"
            f"FINAL_IMAGE_PROMPT:\n"
        )
        return prompt

    def _summarize_previous_scenarios(self, scenarios: list) -> str:
        """
        Summarize previous scenarios for context in the next LLM prompt.
        """
        summary = ""
        for sc in scenarios:
            summary += (
                f"Panel {sc.get('panel_number', '?')}:\n"
                f"SETTING: {sc.get('setting', 'None')}\n"
                f"KEY_ACTIONS_OR_EVENTS: {sc.get('key_actions_or_events', 'None')}\n"
                f"DIALOGUE_SUMMARY_FOR_IMAGE_CONTEXT: {sc.get('dialogue_summary_for_image_context', 'None')}\n"
            )
        return summary

    def _parse_scene_response(self, llm_text_response: str, panel_number: int) -> dict:
        """
        Parse LLM response for a single scene. Fill missing fields with 'None'.
        """
        # Define all required fields
        fields = [
            "panel_number", "scene_identifier", "setting", "characters_present", "camera_shot_and_angle",
            "key_actions_or_events", "lighting_and_atmosphere", "dialogue_summary_for_image_context",
            "visual_effects_or_key_props", "image_style_notes_for_scene", "final_image_prompt"
        ]
        # Flexible label mapping
        label_map = {
            "panel_number": r"PANEL_NUMBER\s*:\s*(\d+)",
            "scene_identifier": r"SCENE_IDENTIFIER\s*:\s*([^\n]+)",
            "setting": r"SETTING\s*:\s*([^\n]*)",
            "characters_present": r"CHARACTERS_PRESENT\s*:\s*([^\n]*)",
            "camera_shot_and_angle": r"CAMERA_SHOT_AND_ANGLE\s*:\s*([^\n]*)",
            "key_actions_or_events": r"KEY_ACTIONS_OR_EVENTS\s*:\s*([^\n]*)",
            "lighting_and_atmosphere": r"LIGHTING_AND_ATMOSPHERE\s*:\s*([^\n]*)",
            "dialogue_summary_for_image_context": r"DIALOGUE_SUMMARY_FOR_IMAGE_CONTEXT\s*:\s*([^\n]*)",
            "visual_effects_or_key_props": r"VISUAL_EFFECTS_OR_KEY_PROPS\s*:\s*([^\n]*)",
            "image_style_notes_for_scene": r"IMAGE_STYLE_NOTES_FOR_SCENE\s*:\s*([^\n]*)",
            "final_image_prompt": r"FINAL_IMAGE_PROMPT\s*:\s*([^\n]*)",
        }
        import re
        result = {}
        for field in fields:
            pattern = label_map[field]
            match = re.search(pattern, llm_text_response, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                result[field] = value if value else "None"
            else:
                result[field] = "None"
        # Ensure panel_number is int
        try:
            result["panel_number"] = int(result["panel_number"])
        except Exception:
            result["panel_number"] = panel_number
        return result

    async def _generate_thumbnail_prompt(self, comic_idea: dict, trace_id: str, extra_log_data: dict) -> str:
        # ... 기존 썸네일 프롬프트 생성 코드 유지 ...
        # (썸네일 프롬프트 예시를 각 장면 프롬프트에 활용하기 위해 반환값을 사용)
        idea_title = comic_idea.get("title", "Untitled Idea")
        idea_summary = comic_idea.get("summary", "A fantastic adventure.")
        idea_genre = comic_idea.get("genre", "adventure")
        user_prompt_content = (
            f"I need a compelling and visually striking thumbnail image prompt (in English) for a webcomic based on the following idea. "
            f"The thumbnail should grab attention and represent the story's essence.\n\n"
            f"Comic Idea Details:\n"
            f"Title: \"{idea_title}\"\n"
            f"Genre: \"{idea_genre}\"\n"
            f"Core Summary: \"{idea_summary}\"\n\n"
            f"Based on this, please generate a single, concise, and powerful image prompt for an AI image generator. "
            f"It should be in English and focus on creating an impactful thumbnail. "
            f"Think about incorporating elements like: key characters in a dynamic pose, a hint of the central conflict or mystery, vibrant colors, and an art style like 'eye-catching webtoon cover art, cinematic quality'.\n\n"
            f"Please provide only the final image generation prompt string as your response, nothing else. Just the prompt itself, without any conversational fluff before or after it."
        )
        logger.info(f"N08: Requesting thumbnail prompt generation from LLM for idea: '{idea_title}'", extra=extra_log_data)
        try:
            llm_response = await self.llm_service.generate_text(prompt=user_prompt_content, max_tokens=150, temperature=0.7)
            if llm_response.get("error"):
                logger.error(f"N08: LLMService error for thumbnail prompt: {llm_response['error']}", extra=extra_log_data)
                return ""
            generated_prompt = llm_response.get("generated_text")
            if generated_prompt:
                final_thumbnail_prompt = generated_prompt.strip()
                # Remove common prefixes
                common_prefixes = ["here is the prompt:", "the prompt is:", "thumbnail prompt:", "image prompt:", "prompt:"]
                for prefix in common_prefixes:
                    if final_thumbnail_prompt.lower().startswith(prefix):
                        final_thumbnail_prompt = final_thumbnail_prompt[len(prefix):].strip()
                if final_thumbnail_prompt.startswith("```") and final_thumbnail_prompt.endswith("```"):
                    final_thumbnail_prompt = final_thumbnail_prompt.strip("` \n")
                logger.info(f"N08: Generated thumbnail prompt: '{final_thumbnail_prompt[:150]}...'", extra=extra_log_data)
                return final_thumbnail_prompt
            else:
                logger.error("N08: LLMService returned no text for thumbnail prompt.", extra=extra_log_data)
                return ""
        except Exception as e:
            logger.exception(f"N08: LLM call error for thumbnail prompt (idea: '{idea_title}'): {type(e).__name__} - {e}", extra=extra_log_data)
            return ""

    async def run(self, state: WorkflowState) -> dict:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')
        current_node_error_log = []
        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        logger.info(f"Entering node {node_name}. Generating scenarios and thumbnail prompt (SLM optimized).", extra=extra)

        selected_idea = state.selected_comic_idea_for_scenario
        if not selected_idea or not isinstance(selected_idea, dict):
            if state.comic_ideas and isinstance(state.comic_ideas, list) and state.comic_ideas:
                selected_idea = state.comic_ideas[0]
                logger.warning(f"N08: selected_comic_idea_for_scenario not found, using first from comic_ideas: '{selected_idea.get('title')}'", extra=extra)
            else:
                error_msg = "No valid comic idea found in state for N08."
                logger.error(error_msg, extra=extra)
                current_node_error_log.append({"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
                state.error_log.extend(current_node_error_log)
                return {"current_stage": "ERROR", "error_log": state.error_log, "error_message": f"{node_name}: {error_msg}"}

        # 1. Generate thumbnail prompt first (to use as example in scene prompts)
        thumbnail_image_prompt_final = await self._generate_thumbnail_prompt(selected_idea, trace_id, extra)
        if not thumbnail_image_prompt_final:
            logger.warning("N08: Failed to generate thumbnail prompt.", extra=extra)
            current_node_error_log.append({"stage": f"{node_name}._generate_thumbnail_prompt", "error": "Thumbnail prompt generation failed.", "timestamp": datetime.now(timezone.utc).isoformat()})

        # 2. Generate 4 scenes, one by one, using previous scenes as context
        scenarios = []
        scenarios_raw_texts = []  # DEBUG: collect raw LLM responses
        for i in range(4):
            previous_summaries = self._summarize_previous_scenarios(scenarios) if i > 0 else ""
            prompt = self._build_scene_prompt(
                comic_idea=selected_idea,
                panel_number=i+1,
                previous_summaries=previous_summaries,
                thumbnail_prompt_example=thumbnail_image_prompt_final or "Vibrant webtoon-style illustration of a frantic, quantum-powered squirrel in mid-air, surrounded by swirling acorns and shattered suburban houses, with a bewildered suburban mom and a shocked kid in the foreground, as a bright, glowing acorn floats out of the squirrel's paw, casting a distorted, warped glow on the chaotic scene, dynamic camera angle from above, bold lines, and bright, poppy colors."
            )
            try:
                llm_response = await self.llm_service.generate_text(prompt=prompt, max_tokens=600, temperature=0.65)
                if llm_response.get("error"):
                    logger.error(f"N08: LLMService error for scene {i+1}: {llm_response['error']}", extra=extra)
                    current_node_error_log.append({"stage": f"{node_name}._generate_scene_{i+1}", "error": llm_response['error'], "timestamp": datetime.now(timezone.utc).isoformat()})
                    scenarios.append({"panel_number": i+1, "scene_identifier": f"S{i+1:02d}P{i+1:02d}", "setting": "None", "characters_present": "None", "camera_shot_and_angle": "None", "key_actions_or_events": "None", "lighting_and_atmosphere": "None", "dialogue_summary_for_image_context": "None", "visual_effects_or_key_props": "None", "image_style_notes_for_scene": "None", "final_image_prompt": "None"})
                    scenarios_raw_texts.append("")
                    continue
                generated_text = llm_response.get("generated_text", "")
                scene = self._parse_scene_response(generated_text, i+1)
                scenarios.append(scene)
                scenarios_raw_texts.append(generated_text)  # DEBUG: save raw response
            except Exception as e:
                logger.exception(f"N08: Exception during scene {i+1} generation: {e}", extra=extra)
                current_node_error_log.append({"stage": f"{node_name}._generate_scene_{i+1}", "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()})
                scenarios.append({"panel_number": i+1, "scene_identifier": f"S{i+1:02d}P{i+1:02d}", "setting": "None", "characters_present": "None", "camera_shot_and_angle": "None", "key_actions_or_events": "None", "lighting_and_atmosphere": "None", "dialogue_summary_for_image_context": "None", "visual_effects_or_key_props": "None", "image_style_notes_for_scene": "None", "final_image_prompt": "None"})
                scenarios_raw_texts.append("")

        state.error_log.extend(current_node_error_log)
        final_status = "N08_SCENARIO_GENERATION_COMPLETED"
        if not scenarios or any(sc.get("final_image_prompt", "None") == "None" for sc in scenarios):
            final_status = "N08_COMPLETED_WITH_PARTIAL_ERRORS"
        if not scenarios:
            final_status = "N08_SCENARIO_GENERATION_FAILED"

        update_dict = {
            "comic_scenarios": scenarios,
            "scenarios_raw_texts": scenarios_raw_texts,  # DEBUG: add raw LLM responses
            "selected_comic_idea_for_scenario": selected_idea,
            "thumbnail_image_prompt": thumbnail_image_prompt_final,
            "current_stage": final_status,
            "error_log": state.error_log
        }
        logger.info(
            f"Exiting node {node_name}. Scenarios: {len(scenarios)}, Thumbnail prompt: {'Generated' if thumbnail_image_prompt_final else 'Not generated'}. Status: {final_status}",
            extra=extra
        )
        return update_dict


async def main_test_n08():
    # (이전 답변의 main_test_n08 내용과 거의 동일하게 유지)
    # LLMService 초기화 부분 및 WorkflowState 아이디어는 영어로 유지
    print("--- N08ScenarioGenerationNode Test (SLM Optimized Prompts, TypeError Fix Attempt) ---")
    logger.info("N08 Test: 시작")

    if not settings.LLM_API_ENDPOINT:
        logger.error("N08 Test: LLM_API_ENDPOINT 설정이 없습니다.")
        return

    llm_service_instance = LLMService()
    node = N08ScenarioGenerationNode(llm_service=llm_service_instance)

    trace_id = f"test-trace-n08-slmfix-{uuid.uuid4().hex[:8]}"
    comic_id = f"test-comic-n08-slmfix-{uuid.uuid4().hex[:8]}"

    sample_idea = {
        "title": "The Quantum Squirrel",
        "summary": "A normal squirrel accidentally gains quantum powers and causes reality-bending chaos in a quiet suburban neighborhood while searching for the ultimate acorn.",
        "genre": "Sci-Fi Comedy, Slapstick",
        "characters": ["Quirky the Squirrel", "Professor Nuts (an eccentric scientist)"],
    }

    state = WorkflowState(
        trace_id=trace_id, comic_id=comic_id,
        original_query="Webtoon about a squirrel with quantum powers.",
        config={"writer_id": "test_writer_n08_slmfix"},
        selected_comic_idea_for_scenario=sample_idea,
        current_stage="N07_IDEATION_COMPLETED", error_log=[]
    )
    logger.info(f"N08 Test: WorkflowState prepared. Comic ID: {comic_id}, Idea: '{sample_idea.get('title')}'")

    result_update = None
    try:
        result_update = await node.run(state)
        logger.info(f"N08 Test: node.run() result summary: {result_update}")
        print(f"\n[INFO] N08 Node Run Complete. Final Stage: {result_update.get('current_stage')}")
        print("\n========== FULL N08 DEBUG OUTPUT ==========")
        # 썸네일 프롬프트 전체 출력
        print(f"\n[THUMBNAIL PROMPT]\n{result_update.get('thumbnail_image_prompt')}")
        # 각 시나리오별 LLM raw 응답 및 파싱 결과 전체 출력
        comic_scenarios_output = result_update.get('comic_scenarios', [])
        scenarios_raw_texts = result_update.get('scenarios_raw_texts', [])
        print(f"\n[SCENARIOS] (Total: {len(comic_scenarios_output)})")
        for i, (sc, raw) in enumerate(zip(comic_scenarios_output, scenarios_raw_texts)):
            print(f"\n--- Scenario {i+1} ({sc.get('scene_identifier')}) ---")
            print(f"[LLM RAW RESPONSE]:\n{raw}")
            print("[PARSED FIELDS]:")
            for k, v in sc.items():
                print(f"  {k}: {v}")
        # 에러 로그 전체 출력
        error_log = result_update.get('error_log', [])
        print(f"\n[ERROR LOG] (Total: {len(error_log)})")
        for err in error_log:
            print(f"  - Stage: {err.get('stage')}, Error: {err.get('error')}, Detail: {err.get('detail', '')}")
        print("\n========== END OF N08 DEBUG OUTPUT ==========")
    except Exception as e:
        logger.error(f"N08 Test: Exception in main test execution: {e}", exc_info=True)
        print(f"[ERROR] Exception in N08 test: {e}")
    finally:
        if hasattr(llm_service_instance, 'close') and asyncio.iscoroutinefunction(
                llm_service_instance.close):  # type: ignore
            await llm_service_instance.close()  # type: ignore
    logger.info("N08 Test: 완료")
    print("--- N08ScenarioGenerationNode Test (SLM Optimized Prompts, TypeError Fix Attempt) End ---")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)  # 필요시 DEBUG로 변경하여 파싱 로직 확인
    asyncio.run(main_test_n08())