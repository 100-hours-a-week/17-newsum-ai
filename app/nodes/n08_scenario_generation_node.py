# ai/app/nodes/n08_scenario_generation_node.py
import asyncio
import traceback
import uuid
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.config.settings import Settings
from app.services.llm_service import LLMService  # 제공된 LLMService 사용

logger = get_logger(__name__)
settings = Settings()  # 전역 settings 객체 사용


class N08ScenarioGenerationNode:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        logger.info("N08ScenarioGenerationNode initialized with LLMService.")

    async def _generate_scenarios_from_idea(
            self,
            comic_idea: Dict[str, Any],
            num_target_scenes: int,
            trace_id: str,
            extra_log_data: dict
    ) -> List[Dict[str, Any]]:
        # LLM과의 대화 시작을 위한 시스템 메시지 (선택적, LLMService가 처리할 수도 있음)
        # system_message = "You are a creative assistant helping to generate detailed comic scenarios."

        scenarios: List[Dict[str, Any]] = []
        idea_title = comic_idea.get("title", "Untitled Idea")
        idea_summary = comic_idea.get("summary", "No summary provided.")
        idea_genre = comic_idea.get("genre", "Unknown Genre")
        idea_characters = comic_idea.get("characters", [])

        # 사용자가 LLM에게 요청하는 메시지 (영어)
        user_prompt_content = (
            f"Hello! I have a comic idea and I need your help to flesh out the scenarios for it. "
            f"The goal is to create approximately {num_target_scenes} distinct scenes or panels.\n\n"
            f"Here's the core idea:\n"
            f"Title: \"{idea_title}\"\n"
            f"Genre: \"{idea_genre}\"\n"
            f"Main Characters (if any): {', '.join(idea_characters) if idea_characters else 'Not specified'}\n"
            f"Summary: \"{idea_summary}\"\n\n"
            f"For each scene, please provide the following details in a JSON object format. I need a list of these JSON objects. "
            f"Each JSON object in the list should represent one scene and must contain these exact keys:\n"
            f"- \"panel_number\": (Integer) The sequence number of the panel, starting from 1.\n"
            f"- \"scene_identifier\": (String) A unique ID for the scene (e.g., \"S01P01_SceneName\" or \"Cut01\").\n"
            f"- \"scene_description\": (String) A detailed visual description of the scene. Describe the setting, characters present, their actions, expressions, mood, and any important background elements. This description should be rich enough for an artist to draw the scene.\n"
            f"- \"dialogue\": (String, optional) Key dialogue spoken by characters in this scene. If no dialogue, this can be an empty string or omitted.\n"
            f"- \"sfx\": (String, optional) Any sound effects in this scene (e.g., \"BOOM!\", \"Whizzz\"). If none, this can be an empty string or omitted.\n"
            f"- \"final_image_prompt\": (String) Based on the scene_description and dialogue, craft a concise and highly descriptive final prompt in English, optimized for an image generation AI. "
            f"This prompt should clearly state the visual elements, character actions, emotions, setting details, composition, and desired art style (e.g., '{settings.IMAGE_DEFAULT_STYLE_PROMPT or 'vibrant webtoon art style, dynamic camera angle'}'). Be very specific!\n\n"
            f"Please ensure your entire response is a single JSON array, where each element is a JSON object structured as described above. For example:\n"
            f"[\n"
            f"  {{\n"
            f"    \"panel_number\": 1,\n"
            f"    \"scene_identifier\": \"S01P01_Introduction\",\n"
            f"    \"scene_description\": \"A young inventor, Alex, stands in a cluttered workshop, a half-finished robot on the table. Sunlight streams through a window.\",\n"
            f"    \"dialogue\": \"Alex: Almost... there!\",\n"
            f"    \"sfx\": \"Sparks!\",\n"
            f"    \"final_image_prompt\": \"({settings.IMAGE_DEFAULT_STYLE_PROMPT or 'detailed anime style'}) Young inventor Alex in a messy workshop, focused on a robot, sunlight, sparks flying, medium shot.\"\n"
            f"  }},\n"
            f"  {{... another scene ...}}\n"
            f"]\n\n"
            f"Can you generate these scenarios for me?"
        )

        logger.info(f"N08: Requesting scenario generation from LLM for idea: '{idea_title}'", extra=extra_log_data)
        try:
            max_tokens = settings.LLM_MAX_TOKENS_SCENARIO
            # LLMService의 generate_text는 prompt 문자열 하나를 받음 (내부적으로 messages 형식으로 변환)
            llm_response = await self.llm_service.generate_text(
                prompt=user_prompt_content,
                max_tokens=max_tokens,
                temperature=0.6,  # 시나리오 생성에는 약간의 일관성 필요
            )

            if llm_response.get("error"):
                logger.error(f"N08: LLMService returned an error for scenario generation: {llm_response['error']}",
                             extra=extra_log_data)
                logger.debug(f"N08: LLMService raw error response: {llm_response.get('raw_response')}",
                             extra=extra_log_data)
                return []

            generated_text = llm_response.get("generated_text")
            if not generated_text:
                logger.error("N08: LLMService returned no generated text for scenarios.", extra=extra_log_data)
                return []

            try:
                # LLM 응답이 JSON 문자열의 리스트 형태인지 확인하고 파싱
                # LLM이 대화형식으로 추가 텍스트를 반환할 수 있으므로, JSON 부분만 추출 시도
                json_start_index = generated_text.find('[')
                json_end_index = generated_text.rfind(']') + 1
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    json_str = generated_text[json_start_index:json_end_index]
                    parsed_scenarios = json.loads(json_str)
                    if isinstance(parsed_scenarios, list):
                        valid_scenarios = []
                        for i, sc_item in enumerate(parsed_scenarios):
                            if isinstance(sc_item, dict) and all(k in sc_item for k in
                                                                 ["scene_identifier", "scene_description",
                                                                  "final_image_prompt"]):
                                sc_item.setdefault("panel_number", i + 1)
                                valid_scenarios.append(sc_item)
                            else:
                                logger.warning(
                                    f"N08: Scenario item {i} from LLM is missing required fields or is not a dict. Item: {summarize_for_logging(sc_item)}",
                                    extra=extra_log_data)
                        scenarios = valid_scenarios
                        logger.info(
                            f"N08: Successfully parsed {len(scenarios)} scenarios from LLM response for idea: '{idea_title}'.",
                            extra=extra_log_data)
                    else:
                        logger.error(
                            f"N08: Parsed scenario data from LLM is not a list. Received: {summarize_for_logging(generated_text)}",
                            extra=extra_log_data)
                else:
                    logger.error(
                        f"N08: Could not find valid JSON array in LLM response. Received: {summarize_for_logging(generated_text)}",
                        extra=extra_log_data)

            except json.JSONDecodeError as e:
                logger.error(
                    f"N08: Failed to parse scenarios JSON from LLM: {e}. Response text: {summarize_for_logging(generated_text)}",
                    extra=extra_log_data)
            except Exception as e:
                logger.error(
                    f"N08: Error processing LLM scenario response content: {e}. Content: {summarize_for_logging(generated_text)}",
                    extra=extra_log_data)

        except Exception as e:
            logger.exception(
                f"N08: Error during LLM call for scenario generation (idea: '{idea_title}'): {type(e).__name__} - {e}",
                extra=extra_log_data)

        return scenarios

    async def _generate_thumbnail_prompt(
            self,
            comic_idea: Dict[str, Any],
            # comic_scenarios: List[Dict[str, Any]], # 시나리오 참고 가능
            trace_id: str,
            extra_log_data: dict
    ) -> Optional[str]:
        idea_title = comic_idea.get("title", "Untitled Idea")
        idea_summary = comic_idea.get("summary", "A fantastic adventure.")
        idea_genre = comic_idea.get("genre", "adventure")

        user_prompt_content = (
            f"I need a compelling and visually striking thumbnail prompt (in English) for a webcomic based on the following idea. "
            f"The thumbnail should grab attention and represent the story's essence.\n\n"
            f"Comic Idea Details:\n"
            f"Title: \"{idea_title}\"\n"
            f"Genre: \"{idea_genre}\"\n"
            f"Core Summary: \"{idea_summary}\"\n\n"
            f"Please generate a single, concise, and highly descriptive image prompt suitable for an AI image generator. "
            f"The prompt should be in English and focus on creating an impactful thumbnail. "
            f"Consider incorporating keywords related to the genre, main characters (if easily depictable), a key visual element, or the overall mood. "
            f"The art style should be something like: '{settings.IMAGE_DEFAULT_STYLE_PROMPT or 'vibrant and dynamic webtoon art, cinematic lighting'}'.\n\n"
            f"For example, a good thumbnail prompt might be: "
            f"\"({settings.IMAGE_DEFAULT_STYLE_PROMPT or 'epic fantasy webtoon style'}) A lone hero silhouetted against a fiery dragon, dramatic backlighting, high contrast, captivating thumbnail art.\"\n\n"
            f"Just provide the final image generation prompt string, nothing else. Your response should be only the prompt itself."
        )

        logger.info(f"N08: Requesting thumbnail prompt generation from LLM for idea: '{idea_title}'",
                    extra=extra_log_data)
        try:
            llm_response = await self.llm_service.generate_text(
                prompt=user_prompt_content,
                max_tokens=150,  # 썸네일 프롬프트는 보통 짧음
                temperature=0.7,
            )

            if llm_response.get("error"):
                logger.error(
                    f"N08: LLMService returned an error for thumbnail prompt generation: {llm_response['error']}",
                    extra=extra_log_data)
                return None

            generated_prompt = llm_response.get("generated_text")
            if generated_prompt:
                final_thumbnail_prompt = generated_prompt.strip()
                # LLM이 추가적인 설명 없이 프롬프트만 반환하도록 유도했지만, 필요시 후처리
                # 예: "Here is your thumbnail prompt:" 같은 머리말 제거
                if "prompt:" in final_thumbnail_prompt.lower():
                    final_thumbnail_prompt = final_thumbnail_prompt.split("prompt:", 1)[-1].strip()

                logger.info(
                    f"N08: Generated thumbnail prompt for idea '{idea_title}': {final_thumbnail_prompt[:150]}...",
                    extra=extra_log_data)
                return final_thumbnail_prompt
            else:
                logger.error("N08: LLMService returned no generated text for thumbnail prompt.", extra=extra_log_data)
                return None

        except Exception as e:
            logger.exception(
                f"N08: Error during LLM call for thumbnail prompt generation (idea: '{idea_title}'): {type(e).__name__} - {e}",
                extra=extra_log_data)
            return None

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')
        current_node_error_log = []

        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        logger.info(f"Entering node {node_name}. Generating scenarios and thumbnail prompt.", extra=extra)

        generated_scenarios_final: List[Dict[str, Any]] = []
        thumbnail_image_prompt_final: Optional[str] = None

        selected_idea = state.selected_comic_idea_for_scenario
        if not selected_idea or not isinstance(selected_idea, dict):
            if state.comic_ideas and isinstance(state.comic_ideas, list) and state.comic_ideas:
                selected_idea = state.comic_ideas[0]
                logger.warning(
                    f"N08: selected_comic_idea_for_scenario not found, using first from comic_ideas: '{selected_idea.get('title')}'",
                    extra=extra)
            else:
                error_msg = "No valid comic idea found in state to generate scenarios."
                logger.error(error_msg, extra=extra)
                current_node_error_log.append(
                    {"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
                state.error_log.extend(current_node_error_log)
                return {"current_stage": "ERROR", "error_log": state.error_log,
                        "error_message": f"{node_name}: {error_msg}"}

        try:
            num_scenes = settings.SCENARIO_TARGET_SCENES_COUNT
            generated_scenarios_final = await self._generate_scenarios_from_idea(
                selected_idea, num_scenes, trace_id, extra  # type: ignore
            )

            if not generated_scenarios_final:
                error_msg = f"Failed to generate any scenarios for idea: '{selected_idea.get('title')}'"  # type: ignore
                logger.error(error_msg, extra=extra)
                current_node_error_log.append({"stage": f"{node_name}._generate_scenarios", "error": error_msg,
                                               "timestamp": datetime.now(timezone.utc).isoformat()})

            thumbnail_image_prompt_final = await self._generate_thumbnail_prompt(
                selected_idea, generated_scenarios_final, trace_id, extra  # type: ignore
            )
            if not thumbnail_image_prompt_final:
                logger.warning("N08: Failed to generate thumbnail prompt, proceeding without it.", extra=extra)
                current_node_error_log.append(
                    {"stage": f"{node_name}._generate_thumbnail_prompt", "error": "Thumbnail prompt generation failed.",
                     "timestamp": datetime.now(timezone.utc).isoformat()})

            state.error_log.extend(current_node_error_log)

            final_status = "N08_SCENARIO_GENERATION_COMPLETED"
            if not generated_scenarios_final:
                final_status = "N08_SCENARIO_GENERATION_FAILED"
            elif current_node_error_log:
                final_status = "N08_COMPLETED_WITH_PARTIAL_ERRORS"

            update_dict = {
                "comic_scenarios": generated_scenarios_final,
                "selected_comic_idea_for_scenario": selected_idea,
                "thumbnail_image_prompt": thumbnail_image_prompt_final,
                "current_stage": final_status,
                "error_log": state.error_log
            }
            logger.info(
                f"Exiting node {node_name}. Scenarios: {len(generated_scenarios_final)}, "
                f"Thumbnail prompt generated: {bool(thumbnail_image_prompt_final)}. Status: {final_status}",
                extra=extra
            )
            return update_dict

        except Exception as e:
            error_msg = f"N08: Unexpected critical error in {node_name} execution: {type(e).__name__} - {e}"
            logger.exception(error_msg, extra=extra)
            current_node_error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                                           "timestamp": datetime.now(timezone.utc).isoformat()})
            state.error_log.extend(current_node_error_log)
            return {
                "comic_scenarios": generated_scenarios_final,
                "thumbnail_image_prompt": thumbnail_image_prompt_final,
                "error_log": state.error_log,
                "current_stage": "ERROR",
                "error_message": f"{node_name} Exception: {error_msg}"
            }


async def main_test_n08():
    # (이전 답변의 main_test_n08 내용과 거의 동일하게 유지, LLMService 초기화 부분만 확인)
    print("--- N08ScenarioGenerationNode Test (Upgraded with English Prompts) ---")
    logger.info("N08 Test: 시작")

    if not settings.LLM_API_ENDPOINT:  # LLMService가 이 설정을 사용
        logger.error("N08 Test: LLM_API_ENDPOINT 설정이 없습니다. 테스트를 진행할 수 없습니다.")
        print("[오류] LLM_API_ENDPOINT 환경변수/설정이 필요합니다.")
        return

    llm_service_instance = LLMService()  # 제공된 llm_service.py 사용
    node = N08ScenarioGenerationNode(llm_service_instance=llm_service_instance)

    trace_id = f"test-trace-n08-en-{uuid.uuid4().hex[:8]}"
    comic_id = f"test-comic-n08-en-{uuid.uuid4().hex[:8]}"

    sample_idea = {
        "title": "The Cosmic Cat's Great Adventure",  # 영어 제목
        "summary": "An ordinary house cat stumbles upon a mysterious spaceship and ends up exploring galaxies, making new alien friends, and saving a planet from a comical villain.",
        # 영어 요약
        "genre": "Sci-Fi Comedy Adventure",  # 영어 장르
        "characters": ["Mimi the Cat", "Zorp the Alien", "Commander Cuddles (the villain)"],  # 영어 캐릭터
    }

    state = WorkflowState(
        trace_id=trace_id, comic_id=comic_id,
        original_query="Make a webtoon about a cat traveling in space.",  # 영어 쿼리
        config={"writer_id": "test_writer_n08_en_prompts"},
        selected_comic_idea_for_scenario=sample_idea,
        current_stage="N07_IDEATION_COMPLETED", error_log=[]
    )
    logger.info(f"N08 Test: WorkflowState prepared. Comic ID: {comic_id}, Idea: '{sample_idea.get('title')}'")

    result_update = None
    try:
        result_update = await node.run(state)
        logger.info(f"N08 Test: node.run() result summary: {summarize_for_logging(result_update, max_len=1500)}")

        print(f"\n[INFO] N08 Node Run Complete. Final Stage: {result_update.get('current_stage')}")
        comic_scenarios_output = result_update.get('comic_scenarios', [])
        print(f"  Generated Scenarios ({len(comic_scenarios_output)} items):")
        for i, sc in enumerate(comic_scenarios_output):
            print(f"    --- Scenario {i + 1} ({sc.get('scene_identifier')}) ---")
            print(f"      Description: {summarize_for_logging(sc.get('scene_description'), 50)}")
            print(f"      Image Prompt: {summarize_for_logging(sc.get('final_image_prompt'), 70)}")

        print(
            f"\n  Generated Thumbnail Image Prompt: {summarize_for_logging(result_update.get('thumbnail_image_prompt'), 150)}")

        if result_update.get('error_log'):
            print(f"  Error Log ({len(result_update['error_log'])} entries):")  # type: ignore
            for err in result_update['error_log']:  # type: ignore
                print(f"    - Stage: {err.get('stage')}, Error: {summarize_for_logging(err.get('error'))}")

    except Exception as e:
        logger.error(f"N08 Test: Exception in main test execution: {e}", exc_info=True)
        print(f"[ERROR] Exception in N08 test: {e}")
    finally:
        if hasattr(llm_service_instance, 'close') and asyncio.iscoroutinefunction(
                llm_service_instance.close):  # type: ignore
            await llm_service_instance.close()  # type: ignore
            logger.info("N08 Test: LLMService resources closed.")

    logger.info("N08 Test: 완료")
    print("--- N08ScenarioGenerationNode Test (Upgraded with English Prompts) End ---")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main_test_n08())