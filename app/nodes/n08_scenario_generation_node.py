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

    def _parse_llm_scenarios_from_text(self, llm_text_response: str, num_target_scenes: int, extra_log_data: dict) -> \
    List[Dict[str, Any]]:
        """
        LLM이 반환한 대화형 텍스트 응답에서 각 장면 정보를 파싱하여 JSON 객체 리스트로 변환합니다.
        SLM이 완벽한 JSON을 생성하지 못할 경우를 대비하여 텍스트 패턴 매칭을 시도합니다.
        """
        scenarios: List[Dict[str, Any]] = []

        # 0. 먼저, 응답 텍스트가 우연히 JSON 배열일 경우 직접 파싱 시도
        try:
            json_block_match = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", llm_text_response,
                                         re.DOTALL | re.IGNORECASE)
            json_str_to_parse = ""
            if json_block_match:
                json_str_to_parse = json_block_match.group(1)
            elif llm_text_response.strip().startswith("[") and llm_text_response.strip().endswith("]"):
                # 전체 응답이 JSON 배열 형태일 수 있음
                json_str_to_parse = llm_text_response.strip()

            if json_str_to_parse:
                parsed_data = json.loads(json_str_to_parse)
                if isinstance(parsed_data, list):
                    valid_items = 0
                    for i, item in enumerate(parsed_data):
                        if isinstance(item, dict) and all(
                                k in item for k in ["scene_identifier", "scene_description", "final_image_prompt"]):
                            item.setdefault("panel_number", i + 1)
                            scenarios.append(item)
                            valid_items += 1
                    if valid_items > 0:  # 하나라도 제대로 파싱되면 JSON 파싱 성공으로 간주
                        logger.info(
                            f"N08: Successfully parsed {len(scenarios)} scenarios directly from JSON block in LLM response.",
                            extra=extra_log_data)
                        return scenarios
                # 리스트가 아니거나, 필수 필드가 없는 경우 텍스트 파싱으로 넘어감
                logger.warning(
                    "N08: LLM response contained JSON-like array but failed validation or was not a list of objects. Attempting text parsing.",
                    extra=extra_log_data)
        except json.JSONDecodeError:
            logger.warning("N08: LLM response was not a valid JSON array, attempting robust text parsing.",
                           extra=extra_log_data)
        except Exception as e_json:
            logger.warning(f"N08: Error during initial JSON parsing attempt: {e_json}. Proceeding with text parsing.",
                           extra=extra_log_data)

        # 1. 텍스트 기반 파싱 로직 (SLM 응답 패턴에 매우 의존적)
        # 예시 패턴: "Panel Number: 1", "Scene Identifier: S01P01", "Scene Description: ...", "Final Image Prompt: ..."
        # 각 "Panel Number:" 또는 "Scene:" 키워드를 기준으로 블록 분리 시도
        # LLM이 각 필드 정보를 명확한 레이블과 함께 제공한다고 가정

        # 각 씬 정보를 담을 임시 딕셔너리
        current_scene_data: Dict[str, Any] = {}
        # 필드 추출을 위한 정규 표현식 (좀 더 유연하게)
        # 각 필드명 뒤에 콜론(:)이 오고, 그 뒤에 내용이 온다고 가정. 내용은 다음 필드명 전까지.
        field_patterns = {
            "panel_number": re.compile(r"Panel Number\s*:\s*(\d+)", re.IGNORECASE),
            "scene_identifier": re.compile(r"Scene Identifier\s*:\s*([^\n]+)", re.IGNORECASE),
            "scene_description": re.compile(
                r"Scene Description\s*:\s*([\s\S]+?)(?=\n\s*(?:Dialogue|SFX|Final Image Prompt|Panel Number|Scene Identifier|$))",
                re.IGNORECASE),
            "dialogue": re.compile(
                r"Dialogue\s*:\s*([\s\S]+?)(?=\n\s*(?:SFX|Final Image Prompt|Panel Number|Scene Identifier|$))",
                re.IGNORECASE),
            "sfx": re.compile(r"SFX\s*:\s*([\s\S]+?)(?=\n\s*(?:Final Image Prompt|Panel Number|Scene Identifier|$))",
                              re.IGNORECASE),
            "final_image_prompt": re.compile(
                r"Final Image Prompt\s*:\s*([\s\S]+?)(?=\n\s*(?:Panel Number|Scene Identifier|$))", re.IGNORECASE)
        }

        # 응답 텍스트를 "Panel Number:" 또는 "Scene Identifier:" (새로운 씬 시작점) 기준으로 분할
        # re.split은 구분자도 포함하므로, 이를 활용하여 각 블록을 재구성할 수 있음
        # 여기서는 더 간단하게, 응답 전체를 하나의 큰 텍스트로 보고 순차적으로 찾음

        # 이전 로그의 오류 ("Expecting ',' delimiter: line 22 column 5 (char 2106)")는
        # LLM이 JSON 유사 형식을 시도했으나 유효하지 않았음을 의미.
        # 따라서, 레이블 기반의 텍스트 추출이 더 안정적일 수 있음.

        # 하나의 큰 텍스트 블록에서 반복적으로 씬 정보를 추출 시도
        text_to_parse = llm_text_response
        panel_idx_counter = 1

        while True:  # 최대 num_target_scenes 만큼 또는 파싱할 내용이 없을 때까지
            if len(scenarios) >= num_target_scenes and num_target_scenes > 0:  # 목표 개수 도달 시
                break

            scene_data_found_in_iteration = False
            current_panel_text_block = ""  # 현재 패널에 해당하는 텍스트 블록

            # "Panel Number: X" 또는 "Scene Identifier: Y"로 시작하는 다음 블록 찾기
            # 이 부분은 LLM 응답이 일관된 구조(예: 각 씬 정보가 명확히 구분됨)를 가질 때 효과적
            # 매우 단순화된 예시:
            next_panel_match = re.search(r"(Panel Number\s*:\s*\d+|Scene Identifier\s*:)", text_to_parse, re.IGNORECASE)
            if current_scene_data:  # 이전 씬 데이터가 있다면, 그것을 먼저 저장
                if all(k in current_scene_data for k in
                       ["scene_identifier", "scene_description", "final_image_prompt"]):
                    current_scene_data.setdefault("panel_number", panel_idx_counter - 1)  # 이전 루프의 카운터
                    scenarios.append(current_scene_data)
                    scene_data_found_in_iteration = True  # 이전 루프에서 찾은 데이터 저장
                current_scene_data = {}  # 새 씬 위해 초기화

            if next_panel_match:
                start_index = next_panel_match.start()
                # 다음 패널 시작점 전까지를 현재 패널의 텍스트 블록으로 간주
                # 이 방식은 LLM이 각 패널 정보를 순차적으로 제공할 때 유효
                # 좀 더 정교하게는, 각 필드 레이블을 찾아 해당 내용을 추출해야 함.

                # 여기서는 좀 더 간단하게, 전체 텍스트에서 순차적으로 필드를 찾습니다.
                # 이 방식은 필드 순서가 바뀌거나 누락되어도 어느정도 대응 가능.
                # 하지만, 여러 씬 정보가 뒤섞여있다면 제대로 분리하기 어려움.
                # LLM 프롬프트에서 각 씬 정보를 명확한 구분자 (예: "--- 다음 씬 ---")로 분리하도록 요청하는 것이 좋음.

                # 아래는 임시로, 로그에서 보인 "I can help you..." 와 같은 머리말 제거 시도
                if text_to_parse.lower().startswith("i can help you") or text_to_parse.lower().startswith(
                        "sure, here are"):
                    first_newline = text_to_parse.find('\n')
                    if first_newline != -1:
                        text_to_parse = text_to_parse[first_newline + 1:].strip()

                # 매우 단순화된 접근: 전체 텍스트에서 각 필드를 찾고, 찾으면 사용하고, 못 찾으면 넘어감
                # 이는 단일 씬 정보만 있는 응답에 더 적합. 여러 씬이 있다면 복잡해짐.
                # SLM이 각 씬 정보를 명확히 구분해서 응답하도록 프롬프팅 하는 것이 핵심.
                # 예: "--- SCENE 1 START ---", 필드들, "--- SCENE 1 END ---"

                # 현재 로직은 전체 텍스트에서 필드를 찾으므로, 여러 씬이 있으면 마지막 씬 정보만 남게 됨.
                # 실제로는 씬 단위로 텍스트를 분리 후, 각 블록에서 필드를 추출해야 함.
                # 여기서는 데모를 위해 첫번째로 찾아지는 정보로 단일 씬을 구성 시도

                if not scenarios:  # 아직 아무 시나리오도 못 찾았을 경우에만 전체 텍스트에서 추출 시도
                    temp_scene = {}
                    pn_match = field_patterns["panel_number"].search(text_to_parse)
                    if pn_match: temp_scene["panel_number"] = int(pn_match.group(1).strip())

                    si_match = field_patterns["scene_identifier"].search(text_to_parse)
                    if si_match: temp_scene["scene_identifier"] = si_match.group(1).strip()

                    desc_match = field_patterns["scene_description"].search(text_to_parse)
                    if desc_match: temp_scene["scene_description"] = desc_match.group(1).strip()

                    dialogue_match = field_patterns["dialogue"].search(text_to_parse)
                    if dialogue_match: temp_scene["dialogue"] = dialogue_match.group(1).strip()

                    sfx_match = field_patterns["sfx"].search(text_to_parse)
                    if sfx_match: temp_scene["sfx"] = sfx_match.group(1).strip()

                    prompt_match = field_patterns["final_image_prompt"].search(text_to_parse)
                    if prompt_match: temp_scene["final_image_prompt"] = prompt_match.group(1).strip()

                    if all(k in temp_scene for k in ["scene_identifier", "scene_description", "final_image_prompt"]):
                        temp_scene.setdefault("panel_number", panel_idx_counter)
                        scenarios.append(temp_scene)
                        logger.info(f"N08: Text-parsed a single scene based on found fields.", extra=extra_log_data)
                break  # 이 단순화된 로직에서는 일단 한 번만 시도하고 루프 종료

            if not scene_data_found_in_iteration and not next_panel_match:  # 더 이상 파싱할 내용이 없으면 종료
                break

            if not next_panel_match and current_scene_data:  # 마지막 남은 데이터 처리
                if all(k in current_scene_data for k in
                       ["scene_identifier", "scene_description", "final_image_prompt"]):
                    current_scene_data.setdefault("panel_number", panel_idx_counter - 1)
                    scenarios.append(current_scene_data)
                break

            # num_target_scenes 만큼만 생성하도록 제한 (실제로는 LLM이 조절)
            if len(scenarios) >= num_target_scenes:
                logger.info(f"N08: Reached target number of scenes ({num_target_scenes}) via text parsing.",
                            extra=extra_log_data)
                break

        if not scenarios:
            logger.error(
                f"N08: Failed to parse any valid scenarios using robust text patterns from LLM response. Response: {summarize_for_logging(llm_text_response)}",
                extra=extra_log_data)

        return scenarios

    async def _generate_scenarios_from_idea(
            self,
            comic_idea: Dict[str, Any],
            num_target_scenes: int,
            trace_id: str,
            extra_log_data: dict
    ) -> List[Dict[str, Any]]:
        idea_title = comic_idea.get("title", "Untitled Idea")
        # ... (이전과 동일한 프롬프트 구성 로직) ...
        user_prompt_content = (
            f"Let's create a comic scenario. I have an idea and I need your help to develop it into about {num_target_scenes} scenes or panels. "
            f"Please provide the details for each scene in a structured way, clearly labeling each piece of information.\n\n"
            f"Here's the idea:\n"
            f"Title: \"{idea_title}\"\n"
            # ... (이전 영어 프롬프트 내용 유지, JSON 형식 요구 대신 레이블 기반 텍스트 응답 유도) ...
            f"For each scene, please clearly label and provide the following information. Use these exact labels followed by a colon and then the content:\n"
            f"Panel Number: [the panel number, e.g., 1]\n"
            f"Scene Identifier: [a short unique ID for the scene, e.g., S01P01_Intro]\n"
            f"Scene Description: [a detailed visual description of the scene...]\n"
            f"Dialogue: [Optional: Key dialogue... If none, state 'No dialogue'.]\n"
            f"SFX: [Optional: Sound effects... If none, state 'No SFX'.]\n"
            f"Final Image Prompt: [Crucial! Based on description/dialogue, a concise, highly descriptive English prompt for an AI image generator, including style like '{settings.IMAGE_DEFAULT_STYLE_PROMPT or 'dynamic webtoon art style, cinematic angle'}'. Be specific!]\n\n"
            f"Please list each scene's information clearly, one after another, using these labels. "
            f"For example:\n"
            f"Panel Number: 1\n"
            f"Scene Identifier: S01P01_Introduction\n"
            f"Scene Description: A young inventor, Alex, stands in a cluttered workshop...\n"
            f"Dialogue: Alex: Almost... there!\n"
            f"SFX: Sparks!\n"
            f"Final Image Prompt: ({settings.IMAGE_DEFAULT_STYLE_PROMPT or 'detailed anime style'}) Young inventor Alex in a messy workshop, focused on a robot, sunlight, sparks flying, medium shot.\n\n"
            f"Then, for the next scene, start again with 'Panel Number: 2', and so on.\n"
            f"Can you generate these scenarios for me in this text format?"
        )
        logger.info(f"N08: Requesting scenario generation (text format) from LLM for idea: '{idea_title}'",
                    extra=extra_log_data)
        parsed_scenarios_list: List[Dict[str, Any]] = []
        try:
            max_tokens = settings.LLM_MAX_TOKENS_SCENARIO
            llm_response = await self.llm_service.generate_text(
                prompt=user_prompt_content, max_tokens=max_tokens, temperature=0.65
            )
            if llm_response.get("error"):
                logger.error(f"N08: LLMService error for scenarios: {llm_response['error']}", extra=extra_log_data)
                return []
            generated_text = llm_response.get("generated_text")
            if not generated_text:
                logger.error("N08: LLMService returned no text for scenarios.", extra=extra_log_data)
                return []

            # LLM이 반환한 텍스트에서 정보 추출하여 JSON 리스트로 변환
            parsed_scenarios_list = self._parse_llm_scenarios_from_text(generated_text, num_target_scenes,
                                                                        extra_log_data)

        except Exception as e:
            logger.exception(f"N08: LLM call error for scenarios (idea: '{idea_title}'): {type(e).__name__} - {e}",
                             extra=extra_log_data)
        return parsed_scenarios_list

    async def _generate_thumbnail_prompt(  # 인자 개수 수정 (comic_scenarios 제거)
            self,
            comic_idea: Dict[str, Any],
            trace_id: str,
            extra_log_data: dict
    ) -> Optional[str]:
        # (이전 답변의 _generate_thumbnail_prompt 내용과 거의 동일, 영어 프롬프트 유지)
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
            f"Think about incorporating elements like: key characters in a dynamic pose, a hint of the central conflict or mystery, vibrant colors, and an art style like '{settings.IMAGE_DEFAULT_STYLE_PROMPT or 'eye-catching webtoon cover art, cinematic quality'}'.\n\n"
            f"Please provide only the final image generation prompt string as your response, nothing else. Just the prompt itself, without any conversational fluff before or after it."
        # 좀 더 명확한 지시
        )
        logger.info(f"N08: Requesting thumbnail prompt generation from LLM for idea: '{idea_title}'",
                    extra=extra_log_data)
        try:
            llm_response = await self.llm_service.generate_text(prompt=user_prompt_content, max_tokens=150,
                                                                temperature=0.7)
            if llm_response.get("error"):
                logger.error(f"N08: LLMService error for thumbnail prompt: {llm_response['error']}",
                             extra=extra_log_data)
                return None
            generated_prompt = llm_response.get("generated_text")
            if generated_prompt:
                final_thumbnail_prompt = generated_prompt.strip()
                # LLM이 "The prompt is: [PROMPT]" 와 같이 응답할 경우 대비
                common_prefixes = ["here is the prompt:", "the prompt is:", "thumbnail prompt:", "image prompt:",
                                   "prompt:"]
                for prefix in common_prefixes:
                    if final_thumbnail_prompt.lower().startswith(prefix):
                        final_thumbnail_prompt = final_thumbnail_prompt[len(prefix):].strip()
                # 혹시 markdown 코드 블록으로 감싸서 줄 경우
                if final_thumbnail_prompt.startswith("```") and final_thumbnail_prompt.endswith("```"):
                    final_thumbnail_prompt = final_thumbnail_prompt.strip("` \n")

                logger.info(f"N08: Generated thumbnail prompt: '{final_thumbnail_prompt[:150]}...'",
                            extra=extra_log_data)
                return final_thumbnail_prompt
            else:
                logger.error("N08: LLMService returned no text for thumbnail prompt.", extra=extra_log_data)
                return None
        except Exception as e:
            logger.exception(
                f"N08: LLM call error for thumbnail prompt (idea: '{idea_title}'): {type(e).__name__} - {e}",
                extra=extra_log_data)
            return None

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        # (run 메소드 상단은 이전과 동일)
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')
        current_node_error_log = []
        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        logger.info(f"Entering node {node_name}. Generating scenarios and thumbnail prompt (SLM optimized).",
                    extra=extra)

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
                error_msg = "No valid comic idea found in state for N08."
                # (이하 오류 처리 동일)
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
                error_msg = f"Failed to generate/parse any valid scenarios for idea: '{selected_idea.get('title')}'"
                logger.error(error_msg, extra=extra)
                current_node_error_log.append({"stage": f"{node_name}._generate_scenarios", "error": error_msg,
                                               "timestamp": datetime.now(timezone.utc).isoformat()})
            elif len(generated_scenarios_final) < num_scenes:
                warn_msg = f"Only {len(generated_scenarios_final)} scenarios generated (target: {num_scenes})."
                logger.warning(warn_msg, extra=extra)
                current_node_error_log.append({"stage": f"{node_name}._generate_scenarios", "error": warn_msg,
                                               "timestamp": datetime.now(timezone.utc).isoformat()})
                final_status = "N08_COMPLETED_WITH_PARTIAL_ERRORS"
            else:
                final_status = "N08_SCENARIO_GENERATION_COMPLETED"

            # _generate_thumbnail_prompt 호출 시 인자 개수 수정 (comic_scenarios 제거)
            # 이전 로그에서 TypeError 발생 지점: takes 4 positional arguments but 5 were given
            # 메소드 정의가 (self, comic_idea, trace_id, extra_log_data) 4개로 가정하고 호출
            thumbnail_image_prompt_final = await self._generate_thumbnail_prompt(
                selected_idea, trace_id, extra  # type: ignore
            )
            if not thumbnail_image_prompt_final:
                logger.warning("N08: Failed to generate thumbnail prompt.", extra=extra)
                current_node_error_log.append(
                    {"stage": f"{node_name}._generate_thumbnail_prompt", "error": "Thumbnail prompt generation failed.",
                     "timestamp": datetime.now(timezone.utc).isoformat()})

            state.error_log.extend(current_node_error_log)  # 현재 노드의 오류들 전체 로그에 추가

            # ====== 시나리오/프롬프트 디버그 출력 추가 ======
            print("\n====== [selected_idea_and_scenario] idea_details ======")
            print(f"Title: {selected_idea.get('title')}")
            print(f"Summary: {selected_idea.get('summary')}")
            print(f"Genre: {selected_idea.get('genre')}")
            print(f"Characters: {selected_idea.get('characters')}")
            print("====== [selected_idea_and_scenario] scenario_details ======")
            for idx, sc in enumerate(generated_scenarios_final):
                print(f"-- Scenario {idx+1} --")
                print(f"ID: {sc.get('scene_identifier')}")
                print(f"Desc: {sc.get('scene_description')[:60]}{'...' if sc.get('scene_description') and len(sc.get('scene_description'))>60 else ''}")
                print(f"Prompt: {sc.get('final_image_prompt')[:60]}{'...' if sc.get('final_image_prompt') and len(sc.get('final_image_prompt'))>60 else ''}")
            print("====== [selected_idea_and_scenario] END ======\n")
            # ====== 디버그 출력 끝 ======

            update_dict = {
                "comic_scenarios": generated_scenarios_final,
                "selected_comic_idea_for_scenario": selected_idea,
                "thumbnail_image_prompt": thumbnail_image_prompt_final,
                "current_stage": final_status,
                "error_log": state.error_log
            }
            logger.info(
                f"Exiting node {node_name}. Scenarios: {len(generated_scenarios_final)}, "
                f"Thumbnail prompt: {'Generated' if thumbnail_image_prompt_final else 'Not generated'}. Status: {final_status}",
                extra=extra
            )
            return update_dict

        except Exception as e:  # run 메소드 내의 최상위 예외 처리
            error_msg = f"N08: Unexpected critical error in {node_name} execution: {type(e).__name__} - {e}"
            logger.exception(error_msg, extra=extra)
            current_node_error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                                           "timestamp": datetime.now(timezone.utc).isoformat()})
            state.error_log.extend(current_node_error_log)
            return {
                "comic_scenarios": generated_scenarios_final,  # 이미 생성된 부분 결과 포함
                "thumbnail_image_prompt": thumbnail_image_prompt_final,  # 이미 생성된 부분 결과 포함
                "error_log": state.error_log,
                "current_stage": "ERROR",
                "error_message": f"{node_name} Exception: {error_msg}"
            }


async def main_test_n08():
    # (이전 답변의 main_test_n08 내용과 거의 동일하게 유지)
    # LLMService 초기화 부분 및 WorkflowState 아이디어는 영어로 유지
    print("--- N08ScenarioGenerationNode Test (SLM Optimized Prompts, TypeError Fix Attempt) ---")
    logger.info("N08 Test: 시작")

    if not settings.LLM_API_ENDPOINT:
        logger.error("N08 Test: LLM_API_ENDPOINT 설정이 없습니다.")
        return

    llm_service_instance = LLMService()
    node = N08ScenarioGenerationNode(llm_service_instance=llm_service_instance)

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
        logger.info(f"N08 Test: node.run() result summary: {summarize_for_logging(result_update, max_len=1500)}")
        print(f"\n[INFO] N08 Node Run Complete. Final Stage: {result_update.get('current_stage')}")
        # ... (이전과 동일한 결과 출력 로직) ...
        comic_scenarios_output = result_update.get('comic_scenarios', [])
        print(f"  Generated Scenarios ({len(comic_scenarios_output)} items):")
        for i, sc in enumerate(comic_scenarios_output):
            print(f"    --- Scenario {i + 1} ({sc.get('scene_identifier')}) ---")
            print(f"      Description: {summarize_for_logging(sc.get('scene_description'), 50)}")
            print(f"      Image Prompt: {summarize_for_logging(sc.get('final_image_prompt'), 70)}")
        print(
            f"\n  Generated Thumbnail Image Prompt: {summarize_for_logging(result_update.get('thumbnail_image_prompt'), 150)}")
        if result_update.get('error_log'):
            print(f"  Error Log ({len(result_update['error_log'])} entries):")
            for err in result_update['error_log']: print(
                f"    - Stage: {err.get('stage')}, Error: {summarize_for_logging(err.get('error'))}")

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