# ai/app/nodes_v2/n09_image_generation_node.py

import asyncio
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.config.settings import Settings
from app.services.image_service import ImageService

logger = get_logger(__name__)
settings = Settings()

MAX_PARALLEL_IMAGE_GENERATION = settings.IMAGE_MAX_PARALLEL_TASKS or 1


class N09ImageGenerationNode:
    def __init__(self, image_service: ImageService):
        self.image_service = image_service
        logger.info(f"N09 initialized. Max parallel: {MAX_PARALLEL_IMAGE_GENERATION}")

    async def _generate_single_image_entry(
        self,
        prompt: str,
        mode: str,
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
            "mode": mode,
            "error": None,
            "raw_service_response": None
        }

        if not prompt:
            result["error"] = "Prompt is empty"
            logger.warning(f"[N09] Missing prompt for {scene_identifier}", extra=extra_log_data)
            return result

        try:
            generation_params = image_style_config or {}
            negative_prompt = generation_params.pop("negative_prompt", settings.IMAGE_DEFAULT_NEGATIVE_PROMPT)

            api_response = await self.image_service.generate_image(
                mode=mode,
                prompt=prompt,
                negative_prompt=negative_prompt,
                **generation_params
            )
            result["raw_service_response"] = api_response
            if api_response.get("error"):
                result["error"] = api_response["error"]
                logger.warning(f"[N09] Image generation failed for {scene_identifier}: {result['error']}",
                               extra=extra_log_data)
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

        extra_log = {"trace_id": trace_id, "comic_id": comic_id, "node": "N09"}

        refined_prompts = image_sec.refined_prompts
        if not refined_prompts:
            msg = "[N09] No refined prompts found"
            logger.error(msg, extra=extra_log)
            meta.current_stage = "ERROR"
            meta.error_message = msg
            return {"image": image_sec.model_dump(), "meta": meta.model_dump()}

        tasks = []
        for entry in refined_prompts:
            scene_id = entry.get("scene_identifier")
            prompt = entry.get("prompt_used", "").strip()
            mode = entry.get("mode", "flux")
            is_thumbnail = scene_id.lower() == "thumbnail"

            style_config = config.get("image_generation_style", {}).get(
                scene_id, config.get("image_generation_style", {}).get("default", {})
            )

            tasks.append(
                self._generate_single_image_entry(
                    prompt=prompt,
                    mode=mode,
                    scene_identifier=scene_id,
                    is_thumbnail=is_thumbnail,
                    image_style_config=style_config,
                    trace_id=trace_id,
                    extra_log_data=extra_log
                )
            )

        results = await asyncio.gather(*tasks)
        image_sec.generated_comic_images = results
        meta.current_stage = "N09_IMAGE_GENERATION_COMPLETED"

        logger.info(f"[N09] Completed image generation: {len(results)} entries", extra=extra_log)
        return {"image": image_sec.model_dump(), "meta": meta.model_dump()}
