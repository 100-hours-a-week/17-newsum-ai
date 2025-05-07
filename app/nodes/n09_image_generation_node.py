# ai/app/nodes/n09_image_generation_node.py
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import re
from pathlib import Path
import asyncio

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging, PROJECT_ROOT
from app.services.image_service import ImageService # ImageService 사용
from app.config.settings import Settings # 설정값 사용 예시

logger = get_logger(__name__)
settings = Settings() # 설정 객체 로드

DEFAULT_RESULTS_BASE_DIR_N09 = PROJECT_ROOT / "results"
# 설정 또는 기본값 사용
MAX_PARALLEL_IMAGE_GENERATIONS = settings.IMAGE_MAX_PARALLEL_TASKS or 3

class N09ImageGenerationNode:
    """
    N08에서 생성된 시나리오의 각 장면 설명을 프롬프트로 사용하여 이미지를 생성하고,
    생성된 이미지 정보(경로 등)를 상태(State)에 저장합니다.
    업그레이드된 N08의 구조화된 시나리오를 파싱하여 활용합니다.
    """

    def __init__(self, image_service: ImageService, results_base_dir: Optional[Path] = None):
        self.image_service = image_service
        self.results_base_dir = results_base_dir or DEFAULT_RESULTS_BASE_DIR_N09
        logger.info(f"N09: Image save directory base set to: {self.results_base_dir}")

    def _parse_structured_scene(self, scene_block_text: str) -> Dict[str, Any]:
        """
        N08에서 생성된 구조화된 장면 텍스트 블록을 파싱하여 딕셔너리로 반환합니다.
        예시: "SETTING: 상세 설명\nCHARACTERS_PRESENT: 이름, 묘사..."
        """
        parsed_data = {}
        # 주석: 레이블(예: "SETTING:", "CHARACTERS_PRESENT:")을 기준으로 파싱합니다.
        # 각 레이블은 새 줄로 시작한다고 가정합니다.
        lines = scene_block_text.strip().split('\n')
        current_label = None
        accumulated_value = ""

        # 주석: 주요 레이블 정의 (N08 프롬프트와 일치해야 함)
        known_labels = [
            "SETTING", "CHARACTERS_PRESENT", "CAMERA_SHOT_AND_ANGLE",
            "KEY_ACTIONS_OR_EVENTS", "LIGHTING_AND_ATMOSPHERE",
            "DIALOGUE_SUMMARY_FOR_IMAGE_CONTEXT", "VISUAL_EFFECTS_OR_KEY_PROPS",
            "IMAGE_STYLE_NOTES_FOR_SCENE"
        ]

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            # 레이블 감지 (콜론 포함)
            found_label = None
            for label in known_labels:
                if stripped_line.startswith(f"{label}:"):
                    found_label = label
                    # 레이블 뒤의 내용 추출 (첫 줄)
                    value_part = stripped_line[len(label)+1:].strip()
                    break

            if found_label:
                # 이전 레이블의 누적된 값 저장
                if current_label and accumulated_value.strip():
                    parsed_data[current_label] = accumulated_value.strip()

                # 새 레이블과 값 시작
                current_label = found_label
                accumulated_value = value_part
            elif current_label:
                # 현재 레이블에 대한 내용 누적 (여러 줄에 걸친 설명 처리)
                accumulated_value += "\n" + stripped_line

        # 마지막 레이블의 누적된 값 저장
        if current_label and accumulated_value.strip():
            parsed_data[current_label] = accumulated_value.strip()

        # 주석: CHARACTERS_PRESENT는 추가 파싱이 필요할 수 있습니다 (예: 리스트로 변환).
        # 여기서는 일단 문자열로 저장합니다.

        return parsed_data

    def _extract_scenes_from_scenario(self, scenario_text: str) -> List[Dict[str, Any]]:
        """
        업그레이드된 N08 시나리오 텍스트에서 구조화된 장면들을 추출하고 파싱합니다.
        """
        scenes_data = []
        # 주석: "--- SCENE X START ---" 와 "--- SCENE X END ---" 마커 사이의 내용을 추출합니다.
        scene_blocks = re.findall(r"--- SCENE \d+ START ---(.*?)--- SCENE \d+ END ---", scenario_text, re.DOTALL | re.IGNORECASE)

        if not scene_blocks and scenario_text.strip():
            # 마커가 없으면 전체 텍스트를 단일 장면으로 간주하고 파싱 시도
             logger.warning("Scenario text does not contain START/END markers. Attempting to parse the whole text as one scene.")
             parsed_single_scene = self._parse_structured_scene(scenario_text)
             if parsed_single_scene: # 파싱 성공 시
                 scenes_data.append({
                     "scene_identifier": "Scene 1 (Overall)",
                     "parsed_elements": parsed_single_scene,
                     "raw_text": scenario_text.strip() # 원본 텍스트도 포함
                 })
             else:
                  logger.warning("Failed to parse the scenario text as a structured scene.")
                  # 구조화 파싱 실패 시, 그냥 통 텍스트를 description으로 사용 (Fallback)
                  scenes_data.append({ "scene_identifier": "Scene 1 (Fallback)", "description_for_prompt": scenario_text.strip(), "parsed_elements": {}})

        else:
            for i, block in enumerate(scene_blocks):
                scene_identifier = f"Scene {i + 1}" # 번호는 순서대로 부여
                # 주석: 각 블록 내부를 파싱하여 구조화된 데이터 생성
                parsed_elements = self._parse_structured_scene(block)
                if parsed_elements: # 파싱 성공 시에만 추가
                    scenes_data.append({
                        "scene_identifier": scene_identifier,
                        "parsed_elements": parsed_elements,
                        "raw_text": block.strip() # 파싱 전 원본 블록 텍스트
                    })
                else:
                     logger.warning(f"Failed to parse structured elements for Scene {i+1}. Raw block: {block[:100]}...")
                     # 파싱 실패 시 fallback으로 raw text 사용
                     scenes_data.append({ "scene_identifier": scene_identifier, "description_for_prompt": block.strip(), "parsed_elements": {}})


        logger.info(f"Extracted and parsed {len(scenes_data)} scenes from scenario.")
        return scenes_data

    def _create_image_prompt_from_parsed_scene(
        self,
        scene_info: Dict[str, Any], # scene_identifier, parsed_elements 포함
        overall_comic_style: List[str],
        global_negative_prompt: Optional[str] = ""
    ) -> Dict[str, str]:
        """
        파싱된 장면 요소들을 조합하여 효과적인 이미지 생성 프롬프트를 만듭니다.
        """
        parsed = scene_info.get("parsed_elements", {})
        raw_desc_fallback = scene_info.get("description_for_prompt", "") # 파싱 실패 시 대비

        # 주석: 파싱된 요소들을 우선적으로 사용하여 프롬프트 구성
        camera = parsed.get("CAMERA_SHOT_AND_ANGLE", "")
        # 캐릭터 정보는 더 상세히 처리 가능 (예: 이름과 포즈/표정만 요약)
        characters = parsed.get("CHARACTERS_PRESENT", "")
        setting = parsed.get("SETTING", "")
        action = parsed.get("KEY_ACTIONS_OR_EVENTS", "")
        lighting = parsed.get("LIGHTING_AND_ATMOSPHERE", "")
        vfx_props = parsed.get("VISUAL_EFFECTS_OR_KEY_PROPS", "")
        scene_style = parsed.get("IMAGE_STYLE_NOTES_FOR_SCENE", "") # 장면별 스타일

        prompt_parts = [
            camera, characters, action, setting, lighting, vfx_props,
            scene_style, # 장면별 스타일 우선
            ", ".join(overall_comic_style) # 전역 스타일 추가
        ]

        # 주석: 파싱된 요소가 너무 부실하면 원본 설명을 fallback으로 사용 가능
        final_positive_prompt = ", ".join(filter(None, [p.strip() for p in prompt_parts if p]))
        if len(final_positive_prompt) < 50 and raw_desc_fallback: # 임계값 예시
             logger.warning(f"Parsed elements yielded short prompt for {scene_info.get('scene_identifier')}. Using raw description as fallback.")
             final_positive_prompt = f"{raw_desc_fallback}, {', '.join(overall_comic_style)}"


        negative_prompt = global_negative_prompt or "(worst quality, low quality:1.2), deformed, blurry, text, letters, signature, watermark"

        return {
            "positive_prompt": final_positive_prompt,
            "negative_prompt": negative_prompt
        }


    async def _generate_and_save_image_task(
        self,
        scene_info: Dict[str, Any], # scene_identifier, parsed_elements, raw_text 포함
        image_prompt_details: Dict[str, str],
        comic_id_str: str,
        scene_index: int,
        config: dict,
        extra_log_data: dict
    ) -> Dict[str, Any]:
        """단일 장면에 대한 이미지를 생성/저장하고 결과 반환."""

        generation_result = {
            "scene_identifier": scene_info.get("scene_identifier", f"Scene {scene_index + 1}"),
            "prompt_used": image_prompt_details["positive_prompt"],
            "negative_prompt_used": image_prompt_details["negative_prompt"],
            "image_path": None,
            "image_url": None,
            "error": None,
            "generation_parameters": {}
        }

        seed = config.get("image_generation_seed")
        steps = config.get("image_generation_steps", settings.IMAGE_DEFAULT_STEPS or 30)
        guidance_scale = config.get("image_generation_guidance_scale", settings.IMAGE_DEFAULT_GUIDANCE or 3.5)
        additional_params = config.get("image_generation_extra_params", {})

        generation_result["generation_parameters"] = {
            "seed": seed, "steps": steps, "guidance_scale": guidance_scale, **additional_params
        }

        try:
            logger.info(f"Generating image for {generation_result['scene_identifier']}...", extra=extra_log_data)
            service_result = await self.image_service.generate_image(
                prompt=image_prompt_details["positive_prompt"],
                negative_prompt=image_prompt_details["negative_prompt"],
                seed=seed,
                num_inference_steps=steps, # ImageService가 이 키를 처리한다고 가정
                guidance_scale=guidance_scale,
                **additional_params
            )

            if "error" not in service_result:
                if service_result.get("image_path"):
                    generation_result["image_path"] = service_result["image_path"]
                    logger.info(f"Image for {generation_result['scene_identifier']} saved to {generation_result['image_path']}", extra=extra_log_data)
                elif service_result.get("image_url"):
                    generation_result["image_url"] = service_result["image_url"]
                    logger.info(f"Image URL for {generation_result['scene_identifier']} received: {generation_result['image_url']}", extra=extra_log_data)
                else:
                     generation_result["error"] = "Image service succeeded but returned no path or URL."
                     logger.warning(f"No image path or URL received for {generation_result['scene_identifier']}.", extra=extra_log_data)
            else:
                generation_result["error"] = service_result["error"]
                logger.warning(f"Image generation failed for {generation_result['scene_identifier']}: {service_result['error']}", extra=extra_log_data)

        except Exception as e:
            error_msg = f"Error during image generation task for {generation_result['scene_identifier']}: {e}"
            logger.exception(error_msg, extra=extra_log_data)
            generation_result["error"] = error_msg

        return generation_result


    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')
        error_log = list(state.error_log or [])

        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}

        num_scenarios = len(state.comic_scenarios) if state.comic_scenarios else 0
        logger.info(
            f"Entering node. Number of scenarios received: {num_scenarios}. Current stage: {state.current_stage}",
            extra=extra
        )

        if not state.comic_scenarios or not isinstance(state.comic_scenarios, list) or not state.comic_scenarios[0].get("scenario_text"):
            logger.warning("No valid scenario text found in state.comic_scenarios. Skipping image generation.", extra=extra)
            return { "generated_comic_images": [], "current_stage": "DONE", "error_log": error_log }

        scenario_to_process = state.comic_scenarios[0]
        scenario_text = scenario_to_process.get("scenario_text", "")
        idea_context = state.selected_comic_idea_for_scenario or {}
        idea_title_for_context = idea_context.get("title", "Untitled Comic")
        idea_genre_for_context = idea_context.get("genre", "General")

        # 주석: 전역 스타일 키워드 및 네거티브 프롬프트 로드
        overall_comic_style = config.get("overall_comic_style", ["digital art", "cinematic lighting"])
        global_negative_prompt = config.get("global_image_negative_prompt", settings.IMAGE_DEFAULT_NEGATIVE_PROMPT or "(worst quality, low quality:1.3), text, signature")


        generated_images_info_list = []

        if not comic_id:
             logger.error("comic_id is missing in state. Cannot save images.", extra=extra)
             error_log.append({ "stage": node_name, "error": "comic_id missing", "timestamp": datetime.now(timezone.utc).isoformat()})
             # comic_id 없으면 이미지 저장이 불가하므로 에러 처리
             return { "generated_comic_images": [], "current_stage": "ERROR", "error_log": error_log, "error_message": f"{node_name} Error: comic_id missing."}

        try:
            # 1. 시나리오에서 구조화된 장면들 추출 및 파싱
            scenes_for_images = self._extract_scenes_from_scenario(scenario_text)
            if not scenes_for_images:
                logger.warning("Could not extract any scenes from the scenario text. Skipping image generation.", extra=extra)
                return { "generated_comic_images": [], "current_stage": "DONE", "error_log": error_log }

            # 2. 각 장면에 대한 이미지 생성 태스크 준비
            image_generation_tasks = []
            for i, scene_info_item in enumerate(scenes_for_images):
                # 주석: 파싱된 정보를 바탕으로 프롬프트 생성
                prompt_details = self._create_image_prompt_from_parsed_scene(
                    scene_info=scene_info_item,
                    overall_comic_style=overall_comic_style,
                    global_negative_prompt=global_negative_prompt
                )
                image_generation_tasks.append(
                    self._generate_and_save_image_task(
                        scene_info=scene_info_item,
                        image_prompt_details=prompt_details,
                        comic_id_str=comic_id,
                        scene_index=i,
                        config=config,
                        extra_log_data=extra
                    )
                )

            # 3. 이미지 생성 병렬 실행
            semaphore = asyncio.Semaphore(MAX_PARALLEL_IMAGE_GENERATIONS)
            async def run_with_semaphore(task_coro):
                async with semaphore:
                    return await task_coro

            if image_generation_tasks:
                logger.info(f"Starting {len(image_generation_tasks)} image generation tasks in parallel (max {MAX_PARALLEL_IMAGE_GENERATIONS} at a time)...", extra=extra)
                task_results = await asyncio.gather(*(run_with_semaphore(task) for task in image_generation_tasks), return_exceptions=True)

                for result_item in task_results:
                    if isinstance(result_item, Exception):
                        logger.error(f"An image generation task failed with an exception: {result_item}", extra=extra)
                        error_log.append({
                            "stage": f"{node_name}.image_task_exception", "error": str(result_item),
                            "detail": traceback.format_exc() if hasattr(result_item, '__traceback__') else None,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        generated_images_info_list.append({
                            "scene_identifier": "Unknown (task exception)", "prompt_used": "N/A",
                            "image_path": None, "image_url": None, "error": str(result_item)
                        })
                    elif isinstance(result_item, dict):
                        generated_images_info_list.append(result_item)
                        if result_item.get("error"):
                             error_log.append({
                                 "stage": f"{node_name}.image_generation_service_error",
                                 "scene": result_item.get("scene_identifier"),
                                 "error": result_item["error"],
                                 "timestamp": datetime.now(timezone.utc).isoformat()
                             })

            # 상태 업데이트 준비
            update_dict = {
                "generated_comic_images": generated_images_info_list,
                "current_stage": "DONE", # N09가 마지막 노드
                "error_log": error_log
            }

            log_summary_update = {
                "current_stage": update_dict["current_stage"],
                "generated_comic_images_count": len(generated_images_info_list),
                "images_with_errors": sum(1 for img in generated_images_info_list if img.get("error"))
            }

            logger.info(
                 f"Exiting node. Processed {len(scenes_for_images)} scenes, generated {len(generated_images_info_list)} image results ({log_summary_update['images_with_errors']} errors). Output Update Summary: {summarize_for_logging(log_summary_update)}",
                extra=extra
            )
            return update_dict

        except Exception as e:
            error_msg = f"Unexpected error in {node_name} execution: {e}"
            logger.exception(error_msg, extra=extra)
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            return {
                "generated_comic_images": generated_images_info_list,
                "error_log": error_log,
                "current_stage": "ERROR",
                "error_message": f"{node_name} Exception: {error_msg}"
            }
        finally:
             # 서비스 클라이언트 세션 닫기
             if hasattr(self, 'image_service') and self.image_service:
                 await self.image_service.close()