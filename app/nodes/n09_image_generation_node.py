# ai/app/nodes/n09_image_generation_node.py
import asyncio
import traceback
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.config.settings import Settings
from app.services.image_service import ImageService  # 제공된 ImageService 사용

logger = get_logger(__name__)
settings = Settings()

MAX_PARALLEL_IMAGE_GENERATION = settings.IMAGE_MAX_PARALLEL_TASKS or 1


class N09ImageGenerationNode:
    def __init__(self, image_service: ImageService):
        self.image_service = image_service
        logger.info(f"N09ImageGenerationNode initialized. Max parallel tasks: {MAX_PARALLEL_IMAGE_GENERATION}")
        logger.info(
            f"ImageService will use storage path: {self.image_service.storage_base_path or 'Not configured (URL only expected)'}")
        # ImageService의 endpoint가 문자열로 변환되었는지 확인 (ImageService __init__에서 처리)
        if isinstance(self.image_service.endpoint, str):
            logger.info(f"ImageService endpoint type is str: {self.image_service.endpoint}")
        else:
            logger.warning(
                f"ImageService endpoint type is NOT str: {type(self.image_service.endpoint)}. This might cause issues with httpx.")

    async def _generate_single_image_entry(
            self,
            prompt: str,
            scene_identifier: str,
            is_thumbnail: bool,
            image_style_config: Optional[Dict[str, Any]],
            trace_id: str,
            extra_log_data: dict
    ) -> Dict[str, Any]:
        image_result_info = {
            "scene_identifier": scene_identifier,
            "prompt_used": prompt,  # 영어 프롬프트 사용
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
            # negative_prompt는 settings에서 기본값을 가져오거나, style_config에서 오버라이드 가능
            negative_prompt = generation_params.pop("negative_prompt", settings.IMAGE_DEFAULT_NEGATIVE_PROMPT)

            # ImageService.generate_image는 영어 프롬프트를 기대
            api_response = await self.image_service.generate_image(
                prompt=prompt,  # 영어 프롬프트
                negative_prompt=negative_prompt,
                **generation_params
            )

            image_result_info["raw_service_response"] = api_response

            if api_response.get("error"):
                image_result_info["error"] = api_response["error"]
                # 오류 메시지에 API의 상세 응답 포함 (민감 정보 주의)
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
                    image_result_info["error"] = "ImageService returned no path or URL, and no error."
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
        # (이전 답변의 run 메소드와 거의 동일, 로깅 및 오류 메시지 약간 수정)
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
            for i, scene_info in enumerate(state.comic_scenarios):  # 인덱스 추가
                if isinstance(scene_info, dict) and scene_info.get("final_image_prompt"):
                    scene_id_for_style = scene_info.get("scene_identifier", f"default_scene_{i}")
                    current_scene_style = image_style_config_main.get(scene_id_for_style,
                                                                      image_style_config_main.get("default", {}))

                    tasks.append(
                        self._generate_single_image_entry(
                            prompt=scene_info["final_image_prompt"],  # 영어 프롬프트
                            scene_identifier=scene_info.get("scene_identifier", f"S_UNKNOWN_{i + 1}"),
                            is_thumbnail=False,
                            image_style_config=current_scene_style,
                            trace_id=trace_id,  # type: ignore
                            extra_log_data=extra_log_data
                        )
                    )
                else:
                    scene_id = scene_info.get("scene_identifier", f"Scene_at_index_{i}")
                    err_msg = f"Skipping image generation for scene '{scene_id}' (index {i}) due to missing 'final_image_prompt' or invalid scene_info."
                    logger.warning(err_msg, extra=extra_log_data)
                    generated_images_overall.append({
                        "scene_identifier": scene_id, "prompt_used": scene_info.get("final_image_prompt"),
                        "image_path": None, "image_url": None, "is_thumbnail": False, "error": err_msg})
                    node_specific_error_log.append({
                        "stage": f"{node_name}.input_validation", "scene": scene_id,
                        "error": err_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
        else:
            logger.warning("N09: No comic scenarios found in state. Skipping main image generation.",
                           extra=extra_log_data)

        # 2. 썸네일 이미지 생성 작업 추가
        if state.thumbnail_image_prompt:  # N08에서 생성된 영어 썸네일 프롬프트
            thumbnail_style = image_style_config_main.get("thumbnail", image_style_config_main.get("default", {}))
            tasks.append(
                self._generate_single_image_entry(
                    prompt=state.thumbnail_image_prompt,
                    scene_identifier="thumbnail_01",
                    is_thumbnail=True,
                    image_style_config=thumbnail_style,
                    trace_id=trace_id,  # type: ignore
                    extra_log_data=extra_log_data
                )
            )
        else:
            logger.warning("N09: No thumbnail_image_prompt found in state. Skipping thumbnail generation.",
                           extra=extra_log_data)
            node_specific_error_log.append({
                "stage": f"{node_name}.thumbnail_skipped", "error": "Thumbnail prompt missing",
                "timestamp": datetime.now(timezone.utc).isoformat()})
            generated_images_overall.append({
                "scene_identifier": "thumbnail_01", "prompt_used": None,
                "image_path": None, "image_url": None, "is_thumbnail": True, "error": "Thumbnail prompt missing"
            })

        # 3. 모든 이미지 생성 작업 병렬 실행
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
                    err_msg = f"N09: Task failed with unhandled exception: {result_item}"
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
        final_status = "N09_IMAGE_GENERATION_COMPLETED"
        attempted_images = [img for img in generated_images_overall if img.get("prompt_used") is not None or img.get(
            "scene_identifier") == "thumbnail_01" and state.thumbnail_image_prompt]

        if any(img.get("error") for img in attempted_images if
               img.get("error") != "Thumbnail prompt missing" and "Skipping image generation" not in img.get("error",
                                                                                                             "")):  # 스킵/프롬프트없음 외의 실제 오류
            final_status = "N09_COMPLETED_WITH_ERRORS"

        successful_generations = [img for img in attempted_images if
                                  not img.get("error") and (img.get("image_path") or img.get("image_url"))]
        if attempted_images and not successful_generations:  # 시도는 했으나 성공한 이미지가 하나도 없는 경우
            final_status = "N09_IMAGE_GENERATION_FAILED"
        elif not attempted_images and (state.comic_scenarios or state.thumbnail_image_prompt):  # 입력은 있었으나 작업이 생성 안됨
            final_status = "N09_IMAGE_GENERATION_SKIPPED"

        update_dict = {"generated_comic_images": generated_images_overall, "current_stage": final_status,
                       "error_log": state.error_log}
        logger.info(
            f"Exiting node {node_name}. Status: {final_status}. Total image entries: {len(generated_images_overall)}. Successful generations: {len(successful_generations)}.",
            extra=extra_log_data)
        return update_dict

    async def close_services(self):  # 이전과 동일
        if hasattr(self.image_service, 'close') and asyncio.iscoroutinefunction(
                self.image_service.close):  # type: ignore
            await self.image_service.close()  # type: ignore
            logger.info("N09: ImageService resources closed.")


async def main_test_n09_with_thumbnail():  # 이전 답변의 main_test_n09_with_thumbnail 사용
    # (이전 답변의 main_test_n09_with_thumbnail 코드 복사, 영어 프롬프트 사용 및 WorkflowState 필드명 확인)
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

    # N08에서 생성되었을 법한 영어 프롬프트 예시
    sample_scenarios_for_n09 = [
        {"scene_identifier": "S01P01_CastleView",
         "final_image_prompt": f"({settings.IMAGE_DEFAULT_STYLE_PROMPT or 'epic fantasy art'}) A lone knight in shining armor, gazing at a majestic, ancient castle perched on a misty mountain peak at sunrise, cinematic lighting, ultra detailed."},
        {"scene_identifier": "S01P02_TavernMeeting",
         "final_image_prompt": f"({settings.IMAGE_DEFAULT_STYLE_PROMPT or 'cozy medieval tavern interior'}) Two adventurers, a rugged warrior and a mysterious mage, secretly exchanging a glowing map over a wooden table, candlelight, shadows."},
        {"scene_identifier": "S01P03_NoPromptTest", "final_image_prompt": None}  # 프롬프트 없는 경우
    ]
    sample_thumbnail_prompt_for_n09 = f"({settings.IMAGE_DEFAULT_STYLE_PROMPT or 'vibrant webtoon cover art'}) Dynamic shot of a brave knight wielding a luminous sword against a shadowy dragon, intense magical energy, a hint of a mystical forest in the background, eye-catching thumbnail."

    state = WorkflowState(
        trace_id=trace_id, comic_id=comic_id,
        original_query="Generate a fantasy comic with a knight and dragon, including a thumbnail.",
        config={"writer_id": "test_writer_n09_en_thumb",
                "image_generation_style": {
                    "default": {"seed": 112233, "num_inference_steps": 25, "guidance_scale": 7.0},
                    "thumbnail": {"seed": 778899, "num_inference_steps": 30, "guidance_scale": 7.5,
                                  "aspect_ratio": "1:1"},  # 썸네일은 1:1 비율 가정
                    "S01P01_CastleView": {"guidance_scale": 8.0}  # 특정 씬 오버라이드
                }},
        comic_scenarios=sample_scenarios_for_n09,
        thumbnail_image_prompt=sample_thumbnail_prompt_for_n09,
        current_stage="N08_SCENARIO_GENERATION_COMPLETED",
        error_log=[]
    )
    logger.info(f"N09 Test (Thumbnail): WorkflowState prepared. Comic ID: {comic_id}")

    result_update = None
    try:
        result_update = await node.run(state)
        logger.info(f"N09 Test (Thumbnail): node.run() result: {summarize_for_logging(result_update, max_len=1200)}")

        print(f"\n[INFO] N09 Node Run Complete. Final Stage: {result_update.get('current_stage')}")
        generated_images_output = result_update.get('generated_comic_images', [])
        print(f"  Generated/Attempted Image Info ({len(generated_images_output)} items):")
        for img_info in generated_images_output:
            print(f"    - Scene/ID: {img_info.get('scene_identifier')}, "
                  f"Is Thumbnail: {img_info.get('is_thumbnail')}, "
                  f"Path: {img_info.get('image_path')}, URL: {img_info.get('image_url')}, "
                  f"Error: {img_info.get('error')}")
            if img_info.get('image_path') and Path(img_info['image_path']).exists():
                print(f"      File '{img_info['image_path']}' exists locally.")
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
                    except Exception as e:
                        logger.warning(
                            f"N09 Test (Thumbnail): Failed to delete image file - {img_info.get('image_path')}: {e}")
    logger.info("N09 Test (Thumbnail): 완료")
    print("--- N09ImageGenerationNode Test (with Thumbnail, English Prompts) End ---")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(main_test_n09_with_thumbnail())