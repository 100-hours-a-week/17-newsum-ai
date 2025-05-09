# ai/app/nodes/n09_image_generation_node.py
import asyncio
import traceback
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging  # PROJECT_ROOT는 여기서 직접 사용 안함
from app.config.settings import Settings
from app.services.image_service import ImageService  # 제공된 ImageService 사용

logger = get_logger(__name__)
settings = Settings()  # 전역 settings 객체 사용

MAX_PARALLEL_IMAGE_GENERATION = settings.IMAGE_MAX_PARALLEL_TASKS or 1


class N09ImageGenerationNode:
    def __init__(self, image_service: ImageService):
        self.image_service = image_service
        logger.info(f"N09ImageGenerationNode initialized. Max parallel tasks: {MAX_PARALLEL_IMAGE_GENERATION}")
        # ImageService 초기화 시 storage_base_path 로깅은 ImageService 내부에서 수행
        logger.info(
            f"N09 using ImageService with endpoint: {self.image_service.endpoint} and storage: {self.image_service.storage_base_path or 'Not configured (URL only expected)'}")

    async def _generate_single_image_entry(
            self,
            prompt: str,
            scene_identifier: str,
            is_thumbnail: bool,
            image_style_config: Optional[Dict[str, Any]],  # config에서 가져온 스타일
            trace_id: str,
            extra_log_data: dict
    ) -> Dict[str, Any]:
        # (이전 답변의 _generate_single_image_entry 메소드 내용과 동일)
        image_result_info = {
            "scene_identifier": scene_identifier,
            "prompt_used": prompt,
            "image_path": None,
            "image_url": None,
            "is_thumbnail": is_thumbnail,
            "error": None,
            "raw_service_response": None
        }
        if not prompt:
            error_msg = f"Image prompt is missing for {'thumbnail' if is_thumbnail else scene_identifier}."
            logger.warning(f"N09: {error_msg}", extra=extra_log_data)
            image_result_info["error"] = error_msg
            return image_result_info

        log_target = "thumbnail" if is_thumbnail else f"scene '{scene_identifier}'"
        logger.info(f"N09: Generating {log_target} image, Prompt (English): '{prompt[:100]}...'", extra=extra_log_data)

        try:
            generation_params = image_style_config or {}
            negative_prompt = generation_params.pop("negative_prompt", settings.IMAGE_DEFAULT_NEGATIVE_PROMPT)

            api_response = await self.image_service.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                **generation_params
            )
            image_result_info["raw_service_response"] = api_response
            if api_response.get("error"):
                image_result_info["error"] = api_response["error"]
                error_detail = api_response.get("details", api_response['error'])
                logger.warning(f"N09: Failed to generate {log_target} image: {error_detail}", extra=extra_log_data)
            else:
                image_result_info["image_path"] = api_response.get("image_path")
                image_result_info["image_url"] = api_response.get("image_url")
                if image_result_info["image_path"]:
                    logger.info(
                        f"N09: Image for {log_target} generated and saved locally: {image_result_info['image_path']}",
                        extra=extra_log_data)
                elif image_result_info["image_url"]:
                    logger.info(f"N09: Image URL for {log_target} received: {image_result_info['image_url']}",
                                extra=extra_log_data)
                else:
                    image_result_info[
                        "error"] = "ImageService returned no path or URL, and no error."  # ImageService의 반환값 보장 필요
                    logger.warning(
                        f"N09: No image path or URL from ImageService for {log_target}. Response: {summarize_for_logging(api_response)}",
                        extra=extra_log_data)
        except Exception as e:
            error_msg = f"N09: Unexpected error generating {log_target} image: {type(e).__name__} - {e}"
            logger.exception(error_msg, extra=extra_log_data)
            image_result_info["error"] = error_msg
            image_result_info["raw_service_response"] = {"error": error_msg, "details": traceback.format_exc()}
        return image_result_info

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        # (이전 답변의 run 메소드 내용과 거의 동일, 오류 상태 결정 로직은 이전 로그 분석 후 수정된 버전 사용)
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        if not comic_id:
            error_msg = "N09 Error: comic_id is missing. Cannot proceed."
            logger.error(error_msg, extra={'trace_id': trace_id, 'node_name': node_name})
            state.error_log.append(
                {"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            return {"current_stage": "ERROR", "error_log": state.error_log, "error_message": error_msg}

        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')
        image_style_config_main = config.get('image_generation_style', {})
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        logger.info(f"Entering node {node_name}. Generating images for scenarios and thumbnail.", extra=extra_log_data)

        generated_images_overall: List[Dict[str, Any]] = []
        node_specific_error_log = []
        tasks = []

        # 1. 컷 이미지 생성 작업 추가
        if state.comic_scenarios and isinstance(state.comic_scenarios, list):
            for i, scene_info in enumerate(state.comic_scenarios):
                if isinstance(scene_info, dict) and scene_info.get("final_image_prompt"):
                    scene_id_for_style = scene_info.get("scene_identifier", f"default_scene_{i}")
                    current_scene_style = image_style_config_main.get(scene_id_for_style,
                                                                      image_style_config_main.get("default", {}))
                    tasks.append(self._generate_single_image_entry(
                        prompt=scene_info["final_image_prompt"],
                        scene_identifier=scene_info.get("scene_identifier", f"S_UNKNOWN_{i + 1}"),
                        is_thumbnail=False, image_style_config=current_scene_style,
                        trace_id=trace_id, extra_log_data=extra_log_data  # type: ignore
                    ))
                else:
                    scene_id = scene_info.get("scene_identifier", f"Scene_at_index_{i}")
                    err_msg = f"Skipping image generation for scene '{scene_id}' (index {i}) due to missing 'final_image_prompt' or invalid scene_info."
                    logger.warning(err_msg, extra=extra_log_data)
                    generated_images_overall.append(
                        {"scene_identifier": scene_id, "prompt_used": scene_info.get("final_image_prompt"),
                         "image_path": None, "image_url": None, "is_thumbnail": False, "error": err_msg})
                    node_specific_error_log.append(
                        {"stage": f"{node_name}.input_validation", "scene": scene_id, "error": err_msg,
                         "timestamp": datetime.now(timezone.utc).isoformat()})
        else:
            logger.warning("N09: No comic scenarios in state. Skipping main image generation.", extra=extra_log_data)

        # 2. 썸네일 이미지 생성 작업 추가
        if state.thumbnail_image_prompt:
            thumbnail_style = image_style_config_main.get("thumbnail", image_style_config_main.get("default", {}))
            tasks.append(self._generate_single_image_entry(
                prompt=state.thumbnail_image_prompt, scene_identifier="thumbnail_01", is_thumbnail=True,
                image_style_config=thumbnail_style, trace_id=trace_id, extra_log_data=extra_log_data  # type: ignore
            ))
        else:
            logger.warning("N09: No thumbnail_image_prompt in state. Skipping thumbnail generation.",
                           extra=extra_log_data)
            node_specific_error_log.append(
                {"stage": f"{node_name}.thumbnail_skipped", "error": "Thumbnail prompt missing",
                 "timestamp": datetime.now(timezone.utc).isoformat()})
            generated_images_overall.append(
                {"scene_identifier": "thumbnail_01", "prompt_used": None, "image_path": None, "image_url": None,
                 "is_thumbnail": True, "error": "Thumbnail prompt missing"})

        if tasks:
            logger.info(
                f"N09: Starting {len(tasks)} total image generation tasks. Max parallel: {MAX_PARALLEL_IMAGE_GENERATION}",
                extra=extra_log_data)
            semaphore = asyncio.Semaphore(MAX_PARALLEL_IMAGE_GENERATION)

            async def run_with_semaphore(task_coro):
                async with semaphore: return await task_coro

            results = await asyncio.gather(*(run_with_semaphore(task) for task in tasks), return_exceptions=True)
            for result_item in results:
                if isinstance(result_item, Exception):
                    err_msg = f"N09: Task failed: {result_item}";
                    logger.error(err_msg, extra=extra_log_data, exc_info=result_item)
                    node_specific_error_log.append({"stage": f"{node_name}.task_exception", "error": str(result_item),
                                                    "detail": traceback.format_exc(),
                                                    "timestamp": datetime.now(timezone.utc).isoformat()})
                    generated_images_overall.append(
                        {"scene_identifier": "Unknown (task exception)", "error": str(result_item),
                         "is_thumbnail": False})
                elif isinstance(result_item, dict):
                    generated_images_overall.append(result_item)
                    if result_item.get("error"):
                        node_specific_error_log.append(
                            {"stage": f"{node_name}.image_service_error", "scene": result_item.get("scene_identifier"),
                             "is_thumbnail": result_item.get("is_thumbnail"), "error": result_item["error"],
                             "timestamp": datetime.now(timezone.utc).isoformat()})
        else:
            logger.info("N09: No image generation tasks to run.", extra=extra_log_data)

        state.error_log.extend(node_specific_error_log)

        # 최종 상태 결정 로직 (TypeError 해결된 버전)
        final_status = "N09_IMAGE_GENERATION_COMPLETED"
        attempted_images_with_prompts = [
            img for img in generated_images_overall
            if img.get("prompt_used") is not None or
               (img.get("is_thumbnail") and state.thumbnail_image_prompt)  # 썸네일 프롬프트가 있었던 경우
        ]

        has_actual_errors = False
        for img in attempted_images_with_prompts:
            img_error_val = img.get("error", "")  # None일 경우 빈 문자열로 처리
            if img_error_val and \
                    img_error_val != "Thumbnail prompt missing" and \
                    "Skipping image generation" not in img_error_val:
                has_actual_errors = True
                break

        expected_images_count = len(state.comic_scenarios or [])
        generated_images_count = len([
            img for img in generated_images_overall
            if (img.get("image_path") or img.get("image_url")) and not img.get("is_thumbnail")
        ])
        if expected_images_count > 0 and generated_images_count < expected_images_count:
            warn_msg = f"Only {generated_images_count} images generated (target: {expected_images_count})."
            logger.warning(warn_msg, extra=extra_log_data)
            node_specific_error_log.append({
                "stage": f"{node_name}.image_generation",
                "error": warn_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            final_status = "N09_COMPLETED_WITH_PARTIAL_ERRORS"
        elif expected_images_count > 0 and generated_images_count == 0:
            error_msg = "Failed to generate any images for provided scenarios."
            logger.error(error_msg, extra=extra_log_data)
            node_specific_error_log.append({
                "stage": f"{node_name}.image_generation",
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            final_status = "ERROR"

        if has_actual_errors:
            final_status = "N09_COMPLETED_WITH_ERRORS"

        successful_generations = [
            img for img in attempted_images_with_prompts
            if not img.get("error") and (img.get("image_path") or img.get("image_url"))
        ]

        if attempted_images_with_prompts and not successful_generations:
            if has_actual_errors:  # 실제 오류로 인해 성공 못한 경우
                final_status = "N09_IMAGE_GENERATION_FAILED"
        elif not attempted_images_with_prompts and (state.comic_scenarios or state.thumbnail_image_prompt):
            final_status = "N09_IMAGE_GENERATION_SKIPPED"  # 입력은 있었으나 작업이 생성 안됨

        update_dict = {"generated_comic_images": generated_images_overall, "current_stage": final_status,
                       "error_log": state.error_log}
        logger.info(
            f"Exiting node {node_name}. Status: {final_status}. Total image entries: {len(generated_images_overall)}. Successful generations: {len(successful_generations)}.",
            extra=extra_log_data)
        return update_dict

    async def close_services(self):
        if hasattr(self.image_service, 'close') and asyncio.iscoroutinefunction(
                self.image_service.close):  # type: ignore
            await self.image_service.close()  # type: ignore
            logger.info("N09: ImageService resources closed.")


async def main_test_n09_with_thumbnail():
    # (이전 답변의 main_test_n09_with_thumbnail 코드와 거의 동일하게 유지)
    print("--- N09ImageGenerationNode Test (Upgraded with English Prompts & Thumbnail) ---")
    logger.info("N09 Test (Thumbnail): 시작")

    if not settings.IMAGE_SERVER_URL:  # ImageService가 이 설정을 사용
        logger.error("N09 Test: IMAGE_SERVER_URL 설정이 없습니다.")
        print("[오류] IMAGE_SERVER_URL 환경변수/설정이 필요합니다.")
        return

    image_service_instance = ImageService()  # 전역 settings 사용
    node = N09ImageGenerationNode(image_service=image_service_instance)

    trace_id = f"test-trace-n09-en-thumb-{uuid.uuid4().hex[:8]}"
    comic_id = f"test-comic-n09-en-thumb-{uuid.uuid4().hex[:8]}"

    sample_scenarios_for_n09 = [
        {"scene_identifier": "S01P01_CastleView",
         "final_image_prompt": f"({settings.IMAGE_DEFAULT_STYLE_PROMPT or 'epic fantasy art'}) A lone knight in shining armor, gazing at a majestic, ancient castle perched on a misty mountain peak at sunrise, cinematic lighting, ultra detailed."},
        {"scene_identifier": "S01P02_TavernMeeting",
         "final_image_prompt": f"({settings.IMAGE_DEFAULT_STYLE_PROMPT or 'cozy medieval tavern interior'}) Two adventurers, a rugged warrior and a mysterious mage, secretly exchanging a glowing map over a wooden table, candlelight, shadows."},
        {"scene_identifier": "S01P03_NoPromptTest", "final_image_prompt": None}
    ]
    sample_thumbnail_prompt_for_n09 = f"({settings.IMAGE_DEFAULT_STYLE_PROMPT or 'vibrant webtoon cover art'}) Dynamic shot of a brave knight wielding a luminous sword against a shadowy dragon, intense magical energy, a hint of a mystical forest in the background, eye-catching thumbnail."

    state = WorkflowState(
        trace_id=trace_id, comic_id=comic_id,
        original_query="Generate a fantasy comic with a knight and dragon, including a thumbnail.",
        config={"writer_id": "test_writer_n09_en_thumb",
                "image_generation_style": {
                    "default": {"seed": 112233, "num_inference_steps": 25, "guidance_scale": 7.0},  # 예시 파라미터
                    "thumbnail": {"seed": 778899, "num_inference_steps": 30, "guidance_scale": 7.5,
                                  "aspect_ratio": "1:1"},
                    "S01P01_CastleView": {"guidance_scale": 8.0}
                }},
        comic_scenarios=sample_scenarios_for_n09,
        thumbnail_image_prompt=sample_thumbnail_prompt_for_n09,
        current_stage="N08_SCENARIO_GENERATION_COMPLETED",
        error_log=[]
    )
    logger.info(f"N09 Test (Thumbnail): WorkflowState 준비 완료. Comic ID: {comic_id}")

    result_update = None
    try:
        result_update = await node.run(state)
        logger.info(f"N09 Test (Thumbnail): node.run() result: {summarize_for_logging(result_update, max_len=1200)}")
        print(f"\n[INFO] N09 Node Run Complete. Final Stage: {result_update.get('current_stage')}")
        # (이하 출력 로직은 이전과 동일)
        generated_images_output = result_update.get('generated_comic_images', [])
        print(f"  Generated/Attempted Image Info ({len(generated_images_output)} items):")
        for img_info in generated_images_output:
            print(
                f"    - Scene/ID: {img_info.get('scene_identifier')}, Is Thumbnail: {img_info.get('is_thumbnail')}, Path: {img_info.get('image_path')}, URL: {img_info.get('image_url')}, Error: {img_info.get('error')}")
            if img_info.get('image_path') and Path(img_info['image_path']).exists(): print(
                f"      File '{img_info['image_path']}' exists locally.")
    except Exception as e:
        logger.error(f"N09 Test (Thumbnail): Exception in main test execution: {e}", exc_info=True)
        print(f"[ERROR] Exception in N09 test: {e}")
    finally:
        await node.close_services()
        if result_update and result_update.get('generated_comic_images'):
            logger.info("N09 Test (Thumbnail): Cleaning up generated image files...")
            for img_info in result_update['generated_comic_images']:
                if img_info.get('image_path'):
                    try:
                        img_p = Path(img_info['image_path'])
                        if img_p.is_file(): img_p.unlink()
                        logger.info(f"N09 Test (Thumbnail): Deleted - {img_p}")
                    except Exception as e_del:
                        logger.warning(
                            f"N09 Test (Thumbnail): Failed to delete image file - {img_info.get('image_path')}: {e_del}")
    logger.info("N09 Test (Thumbnail): 완료")
    print("--- N09ImageGenerationNode Test (with Thumbnail, English Prompts) End ---")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] [%(node_name)s] %(message)s')
    # setup_logging() # YAML 설정 사용 시
    asyncio.run(main_test_n09_with_thumbnail())