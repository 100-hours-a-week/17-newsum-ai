import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.workflows.state_v2 import WorkflowState, DEFAULT_IMAGE_MODE, FLUX_BASE_MODES, XL_BASE_MODES
from app.utils.logger import get_logger, summarize_for_logging
from app.config.settings import Settings
from app.services.image_service import ImageService

logger = get_logger(__name__)
settings = Settings()

class N09ImageGenerationNode:
    """
    Sequential image generation node. Uses prompts from state.image.refined_prompts
    to generate images one by one, storing results in state.image.generated_comic_images.
    """

    def __init__(self, image_service: ImageService):
        self.image_service = image_service
        logger.info(f"N09 initialized.")

    async def _generate_single_image_entry(
        self,
        prompt: str,
        model_name: str,
        scene_identifier: str,
        is_thumbnail: bool,
        image_style_config: Optional[Dict[str, Any]],
        trace_id: str,
        extra_log_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = {
            "scene_identifier": scene_identifier,
            "prompt_used": prompt,
            "image_path": None,
            "image_url": None,
            "is_thumbnail": is_thumbnail,
            "model_name": model_name,
            "error": None,
            "raw_service_response": None
        }

        if not prompt:
            result["error"] = "Prompt is empty"
            logger.warning(f"[N09] Missing prompt for {scene_identifier}", extra=extra_log_data)
            return result

        try:
            generation_params = image_style_config or {}
            negative_prompt = generation_params.pop(
                "negative_prompt", settings.IMAGE_DEFAULT_NEGATIVE_PROMPT
            )

            api_response = await self.image_service.generate_image(
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                **generation_params
            )
            result["raw_service_response"] = api_response
            if api_response.get("error"):
                result["error"] = api_response["error"]
                logger.warning(
                    f"[N09] Image generation failed for {scene_identifier}: {result['error']}",
                    extra=extra_log_data
                )
            else:
                result["image_path"] = api_response.get("image_path")
                result["image_url"] = api_response.get("image_url")
        except Exception as e:
            result["error"] = str(e)
            result["raw_service_response"] = {"traceback": traceback.format_exc()}
            logger.exception(f"[N09] Unexpected error for {scene_identifier}", extra=extra_log_data)
        return result

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta = state.meta
        image_sec = state.image
        config = state.config.config or {}
        trace_id = meta.trace_id
        comic_id = meta.comic_id
        node_name = self.__class__.__name__
        extra_log = {"trace_id": trace_id, "comic_id": comic_id, "node_name": node_name}

        logger.info(
            f"[{node_name}] 이미지 생성 노드 진입. trace_id={trace_id}, comic_id={comic_id}",
            extra=extra_log
        )

        refined_prompts = image_sec.refined_prompts or []
        if not refined_prompts:
            msg = f"[{node_name}] No refined prompts found"
            logger.error(msg, extra=extra_log)
            meta.current_stage = "ERROR"
            meta.error_message = msg
            return {"image": image_sec.model_dump(), "meta": meta.model_dump()}

        results: List[Dict[str, Any]] = []
        for entry in refined_prompts:
            scene_id = entry.get("scene_identifier")
            prompt = entry.get("prompt_used", "").strip()
            model_name = entry.get("model_name") or DEFAULT_IMAGE_MODE
            is_thumbnail = (scene_id.lower() == "thumbnail")
            if model_name not in FLUX_BASE_MODES and model_name not in XL_BASE_MODES:
                model_name = DEFAULT_IMAGE_MODE
            style_config = config.get("image_generation_style", {}).get(
                scene_id,
                config.get("image_generation_style", {}).get("default", {})
            )
            logger.debug(
                f"[{node_name}] Generating for scene={scene_id}, model={model_name}, thumbnail={is_thumbnail}",
                extra=extra_log
            )
            res = await self._generate_single_image_entry(
                prompt=prompt,
                model_name=model_name,
                scene_identifier=scene_id,
                is_thumbnail=is_thumbnail,
                image_style_config=style_config,
                trace_id=trace_id,
                extra_log_data=extra_log
            )
            logger.info(
                f"[{node_name}] Task result: {scene_id} | error={res.get('error')}",
                extra=extra_log
            )
            results.append(res)

        image_sec.generated_comic_images = results
        meta.current_stage = "N10_FINALIZE_AND_NOTIFY"

        logger.info(
            f"[{node_name}] 이미지 생성 완료. 총 엔트리: {len(results)}", extra=extra_log
        )
        return {"image": image_sec.model_dump(), "meta": meta.model_dump()}
