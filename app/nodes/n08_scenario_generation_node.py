# ai/app/nodes/n08_scenario_generation_node.py
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import re
import json
from pathlib import Path

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging, PROJECT_ROOT
from app.services.llm_service import LLMService
from app.config.settings import Settings # 설정값 사용 예시

logger = get_logger(__name__)
settings = Settings() # 설정 객체 로드

# 시나리오의 원하는 대략적인 장면 수 또는 길이
TARGET_SCENES_COUNT = settings.SCENARIO_TARGET_SCENES_COUNT or 5 # 설정 또는 기본값
# 결과 저장 기본 디렉토리
DEFAULT_RESULTS_BASE_DIR_N08 = PROJECT_ROOT / "results"

class N08ScenarioGenerationNode:
    """
    N07에서 생성된 만화 아이디어 중 하나를 선택하여, 이미지 생성에 최적화된
    상세하고 구조화된 시나리오를 작성하고, 모든 아이디어와 생성된 시나리오를
    JSON 파일로 저장합니다.
    """

    def __init__(self, llm_service: LLMService, results_base_dir: Optional[Path] = None):
        self.llm_service = llm_service
        self.results_base_dir = results_base_dir or DEFAULT_RESULTS_BASE_DIR_N08
        logger.info(f"N08: Scenario JSON save directory base set to: {self.results_base_dir}")


    async def _generate_scenario_from_idea_sllm(
        self, comic_idea: Dict[str, Any], original_query: str, refined_intent: str,
        num_scenes: int, config: dict, trace_id: str, extra_log_data: dict
    ) -> Optional[Dict[str, Any]]: # 시나리오 텍스트와 추가 정보 반환
        """LLM을 사용하여 선택된 아이디어로부터 만화 시나리오 생성 (업그레이드된 프롬프트 사용)"""
        writer_persona = config.get('writer_id', 'default_storyteller')
        target_audience = config.get('target_audience', 'general_audience')
        # 주석: 설정 또는 아이디어 자체에서 전역 만화 스타일 키워드를 가져옵니다.
        overall_comic_style = config.get("overall_comic_style", "digital painting, comic book art style, detailed illustration")

        idea_title = comic_idea.get('title', 'N/A')
        idea_logline = comic_idea.get('logline', 'N/A')
        idea_genre = comic_idea.get('genre', 'N/A')
        idea_emotion = comic_idea.get('target_emotion', 'N/A')
        key_elements_list = comic_idea.get('key_elements_from_report', [])
        idea_keywords = ", ".join(key_elements_list) if isinstance(key_elements_list, list) else (key_elements_list or "N/A")

        # --- 업그레이드된 프롬프트 ---
        prompt = f"""[System] You are a skilled comic scenario writer with the persona '{writer_persona}', targeting a '{target_audience}' audience, and you understand how to write scenarios optimized for AI image generation using models like FLUX.
Based on the following 'Comic Idea', write a detailed comic scenario consisting of approximately {num_scenes} key scenes.
For each scene, provide a detailed description focusing on visual elements that an AI image generator can interpret. Structure each scene clearly using the following labels followed by their descriptions on new lines:

--- SCENE [Scene Number] START ---
SETTING: Detailed description of the environment, time of day, weather, key background elements.
CHARACTERS_PRESENT: List characters. For each: NAME, APPEARANCE_NOTES (consistent clothing/hair/features), POSE_AND_EXPRESSION (specific action, pose, facial details).
CAMERA_SHOT_AND_ANGLE: Suggested shot (e.g., close-up, medium shot, establishing shot) and angle (e.g., eye-level, low-angle).
KEY_ACTIONS_OR_EVENTS: The main things happening in the scene plot-wise.
LIGHTING_AND_ATMOSPHERE: Describe the lighting (e.g., dim, bright, moody, cinematic) and overall mood.
(Optional) DIALOGUE_SUMMARY_FOR_IMAGE_CONTEXT: Very brief dialogue summary if visually relevant.
(Optional) VISUAL_EFFECTS_OR_KEY_PROPS: Special effects (motion blur, impact lines) or important objects.
(Optional) IMAGE_STYLE_NOTES_FOR_SCENE: Specific style notes for THIS scene if different from the overall style.
--- SCENE [Scene Number] END ---

Overall Comic Style (Apply generally unless overridden by scene notes): {overall_comic_style}

Ensure the scenario evokes the '{idea_emotion}' emotion and reflects the '{idea_genre}' genre.
Try to naturally incorporate the 'Keywords from Report': {idea_keywords}
The overall story should align with the logline.

Comic Idea:
- Title: {idea_title}
- Logline: {idea_logline}
- Genre: {idea_genre}
- Target Emotion: {idea_emotion}
- Suggested Overall Comic Style: {overall_comic_style} # 아이디어에도 스타일 제안 포함 가능

Original User Query (for context only): "{original_query}"
Refined Core Question (for context only): "{refined_intent}"

[Task]
Write the comic scenario following the structured format above. Use the START/END markers for each scene. Ensure a clear beginning and end to the overall narrative. Adhere strictly to the labeled structure within each scene block.
"""
        logger.debug(f"Attempting to generate scenario for idea '{idea_title}' with upgraded prompt...", extra=extra_log_data)
        # 주석: 상세한 출력을 위해 max_tokens 증가
        result = await self.llm_service.generate_text(prompt=prompt, max_tokens=settings.LLM_MAX_TOKENS_SCENARIO or 4000, temperature=0.6) # 온도는 약간 낮춰 일관성 및 디테일 유도

        if "error" in result or not result.get("generated_text"):
            logger.error(f"sLLM scenario generation failed for idea '{idea_title}': {result.get('error', 'No text generated')}", extra=extra_log_data)
            return None

        scenario_text = result["generated_text"].strip()
        logger.info(f"Successfully generated scenario text for idea '{idea_title}'. Length: {len(scenario_text)}", extra=extra_log_data)

        # 주석: 생성된 시나리오 텍스트와 함께 기반 아이디어 제목 반환.
        # 추가적인 구조화된 파싱은 N09에서 수행하거나 여기서 수행할 수 있음. 여기서는 텍스트만 반환.
        return {
            "idea_title": idea_title,
            "scenario_text": scenario_text,
            # 주석: 대략적인 장면 수는 여기서 세지 않고, N09의 파싱 결과에 따라 결정하는 것이 더 정확할 수 있음.
        }

    def _save_scenario_data_to_json(
        self, comic_id_str: str, all_ideas: List[Dict[str, Any]], generated_scenario: Optional[Dict[str, Any]],
        extra_log_data: dict
    ) -> Optional[str]:
        """생성된 아이디어들과 (단일) 시나리오를 JSON 파일로 저장합니다."""
        if not comic_id_str:
            logger.error("Cannot save scenario JSON: comic_id is missing.", extra=extra_log_data)
            return None

        try:
            data_to_save = {
                "comic_id": comic_id_str,
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                "all_generated_ideas": all_ideas or [], # N07 결과 (없으면 빈 리스트)
                "selected_idea_and_scenario": None
            }
            # 주석: generated_scenario는 {'idea_title': ..., 'scenario_text': ...} 형태
            if generated_scenario and generated_scenario.get('scenario_text'):
                # 시나리오의 기반이 된 아이디어 찾기
                base_idea = next((idea for idea in (all_ideas or []) if idea.get('title') == generated_scenario.get('idea_title')), None)
                data_to_save["selected_idea_and_scenario"] = {
                    "idea_details": base_idea, # 찾은 아이디어 또는 None
                    "scenario_details": generated_scenario # 생성된 시나리오 데이터
                }
            elif generated_scenario: # 시나리오는 생성 시도했으나 텍스트가 없는 경우 (오류 등)
                 data_to_save["selected_idea_and_scenario"] = { "idea_title_attempted": generated_scenario.get('idea_title'), "error": "Scenario text generation failed or empty."}


            save_dir = self.results_base_dir / comic_id_str
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / "scenarios_result.json" # 파일명 지정

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)

            saved_path_str = str(file_path.resolve())
            logger.info(f"Scenario data successfully saved to: {saved_path_str}", extra=extra_log_data)
            return saved_path_str

        except OSError as e:
            logger.exception(f"Failed to save scenario JSON file system error for comic_id {comic_id_str}: {e}", extra=extra_log_data)
            return None
        except Exception as e:
            logger.exception(f"Unexpected error during scenario JSON saving for comic_id {comic_id_str}: {e}", extra=extra_log_data)
            return None


    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')
        error_log = list(state.error_log or [])

        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        comic_ideas_count = len(state.comic_ideas) if state.comic_ideas else 0
        logger.info(
            f"Entering node. Input State Summary: {summarize_for_logging(state.model_dump(exclude_none=True), fields_to_show=['current_stage'])}. Number of comic ideas received: {comic_ideas_count}",
            extra=extra
        )

        if not state.comic_ideas:
            logger.warning("No comic ideas found from N07. Cannot generate scenario. Saving empty data.", extra=extra)
            if comic_id:
                 self._save_scenario_data_to_json(comic_id, [], None, extra) # 빈 아이디어/시나리오 정보 저장
            return {
                "comic_scenarios": [], # 상태에는 빈 리스트 저장
                "selected_comic_idea_for_scenario": None,
                "current_stage": "n09_image_generation", # 시나리오는 없지만 이미지 생성 시도 가능 (선택적) 또는 "DONE"
                "error_log": error_log
            }

        generated_scenarios_list = [] # 상태에 저장할 리스트 (단일 요소 포함)
        selected_idea_for_state: Optional[Dict[str, Any]] = None
        generated_scenario_data_for_json: Optional[Dict[str, Any]] = None # JSON 저장용

        try:
            # 주석: 단일 시나리오 생성을 위해 첫 번째 아이디어를 선택합니다.
            idea_to_process = state.comic_ideas[0]
            selected_idea_for_state = idea_to_process.copy() # 상태 저장을 위해 복사
            logger.info(f"Selected comic idea for scenario generation: '{idea_to_process.get('title', 'N/A')}'", extra=extra)

            original_query = state.original_query or "N/A"
            refined_intent = state.query_context.get("refined_intent", original_query) if state.query_context else original_query

            # 주석: 업그레이드된 프롬프트를 사용하여 시나리오 생성
            generated_scenario_data_for_json = await self._generate_scenario_from_idea_sllm(
                idea_to_process,
                original_query,
                refined_intent,
                TARGET_SCENES_COUNT,
                config,
                trace_id,
                extra
            )

            if generated_scenario_data_for_json:
                generated_scenarios_list.append(generated_scenario_data_for_json) # 상태에 저장
            else:
                logger.warning(f"LLM failed to generate a scenario for idea: '{idea_to_process.get('title', 'N/A')}'", extra=extra)
                # 주석: 시나리오 생성 실패 시에도 JSON 저장은 시도 (실패 정보 포함)
                generated_scenario_data_for_json = {"idea_title": idea_to_process.get('title'), "error": "Scenario generation failed"}


            # --- 아이디어 및 시나리오 파일 저장 ---
            saved_json_path = None
            if comic_id:
                saved_json_path = self._save_scenario_data_to_json(
                    comic_id_str=comic_id,
                    all_ideas=state.comic_ideas, # N07의 모든 아이디어
                    generated_scenario=generated_scenario_data_for_json, # 생성된 (또는 실패 정보 포함) 시나리오 데이터
                    extra_log_data=extra
                )
                if not saved_json_path:
                    error_log.append({
                        "stage": f"{node_name}._save_scenario_data_to_json",
                        "error": "Failed to save scenarios_result.json",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            else:
                logger.warning("comic_id is not available, cannot save scenarios_result.json.", extra=extra)
            # --- 파일 저장 완료 ---

            # 주석: WorkflowState 업데이트 준비
            update_dict = {
                "comic_scenarios": generated_scenarios_list, # 생성된 시나리오 데이터 리스트 (최대 1개)
                "selected_comic_idea_for_scenario": selected_idea_for_state, # 어떤 아이디어를 사용했는지
                "current_stage": "n09_image_generation", # 다음 노드로
                "error_log": error_log
            }

            # 주석: 로깅을 위한 요약 정보 생성
            log_summary_update = {
                "current_stage": update_dict["current_stage"],
                "comic_scenarios_count": len(generated_scenarios_list),
                "selected_comic_idea_title": selected_idea_for_state.get('title', 'N/A') if selected_idea_for_state else 'N/A',
                "saved_scenario_json_path": saved_json_path # 저장 경로 로깅 추가
            }

            logger.info(
                 f"Exiting node. Generated {len(generated_scenarios_list)} scenario(s). Output Update Summary: {summarize_for_logging(log_summary_update)}",
                extra=extra
            )
            return update_dict

        except IndexError:
            logger.warning("Comic ideas list was empty. Cannot generate scenario. Saving empty data.", extra=extra)
            if comic_id:
                self._save_scenario_data_to_json(comic_id, [], None, extra)
            return {
                "comic_scenarios": [], "selected_comic_idea_for_scenario": None,
                "current_stage": "n09_image_generation", "error_log": error_log
            }
        except Exception as e:
            error_msg = f"Unexpected error in {node_name} execution: {e}"
            logger.exception(error_msg, extra=extra)
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            # 주석: 예외 발생 시에도 JSON 저장 시도
            if comic_id:
                 self._save_scenario_data_to_json(comic_id, state.comic_ideas or [], generated_scenario_data_for_json, extra)

            return {
                "comic_scenarios": generated_scenarios_list,
                "selected_comic_idea_for_scenario": selected_idea_for_state,
                "error_log": error_log,
                "current_stage": "ERROR", # 오류 상태로 종료
                "error_message": f"{node_name} Exception: {error_msg}"
            }