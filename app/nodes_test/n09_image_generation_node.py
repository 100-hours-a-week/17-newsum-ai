# ai/app/nodes_v2/n09_image_generation_node.py

import traceback
from typing import Any, Dict, List, Optional

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger
from app.config.settings import Settings
from app.services.image_service import ImageService
from app.config.image_style_config import IMAGE_STYLE_CONFIGS, DEFAULT_IMAGE_MODE

logger = get_logger(__name__)
settings = Settings()

class N09ImageGenerationNode:
    """
    n08a에서 생성된 refined_prompts를 이용해
    Flux Dev(또는 설정된 image_mode) 모델로 이미지를 생성합니다.
    """

    def __init__(self, image_service: ImageService):
        self.image_service = image_service
        logger.info("N09ImageGenerationNode initialized.")

    async def _generate_single_image_entry(
        self,
        scene_identifier: str,
        prompt: str,
        model_name: str,
        is_thumbnail: bool,
        image_style_config: Dict[str, Any],
        trace_id: str,
        extra_log_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "scene_identifier": scene_identifier,
            "prompt_used": prompt,
            "model_name": model_name,
            "image_path": None,
            "image_url": None,
            "is_thumbnail": is_thumbnail,
            "error": None,
            "raw_service_response": None
        }

        if not prompt:
            result["error"] = "Prompt is empty"
            logger.warning(f"[N09] Missing prompt for {scene_identifier}", extra=extra_log_data)
            return result

        try:
            # 스타일별 negative_prompt
            negative_prompt = image_style_config.get(
                "negative_prompt",
                settings.IMAGE_DEFAULT_NEGATIVE_PROMPT
            )

            api_response = await self.image_service.generate_image(
                model_name=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt
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
        trace_id = meta.trace_id
        comic_id = getattr(meta, "comic_id", None)
        node_name = self.__class__.__name__
        extra_log = {"trace_id": trace_id, "comic_id": comic_id, "node_name": node_name}

        logger.info(
            f"[{node_name}] Entering image generation node. trace_id={trace_id}, comic_id={comic_id}",
            extra=extra_log
        )

        refined_prompts = image_sec.refined_prompts or []
        if not refined_prompts:
            msg = f"[{node_name}] No refined_prompts found"
            logger.error(msg, extra=extra_log)
            meta.current_stage = "ERROR"
            meta.error_message = msg
            return {"image": image_sec.model_dump(), "meta": meta.model_dump()}

        results: List[Dict[str, Any]] = []
        for entry in refined_prompts:
            scene_id = entry.get("scene_identifier")
            prompt_text = entry.get("prompt_used", "").strip()
            model_name = entry.get("model_name") or DEFAULT_IMAGE_MODE
            is_thumbnail = (scene_id.lower() == "thumbnail")
            style_conf = IMAGE_STYLE_CONFIGS.get(model_name, IMAGE_STYLE_CONFIGS[DEFAULT_IMAGE_MODE])

            logger.debug(
                f"[{node_name}] Generating for scene={scene_id}, model={model_name}, thumbnail={is_thumbnail}",
                extra=extra_log
            )

            res = await self._generate_single_image_entry(
                scene_identifier=scene_id,
                prompt=prompt_text,
                model_name=model_name,
                is_thumbnail=is_thumbnail,
                image_style_config=style_conf,
                trace_id=trace_id,
                extra_log_data=extra_log
            )

            logger.info(
                f"[{node_name}] Scene={scene_id} generation result | error={res.get('error')}",
                extra=extra_log
            )
            results.append(res)

        image_sec.generated_comic_images = results
        meta.current_stage = "N10_FINALIZE_AND_NOTIFY"

        logger.info(
            f"[{node_name}] Image generation complete. Total entries: {len(results)}",
            extra=extra_log
        )
        return {"image": image_sec.model_dump(), "meta": meta.model_dump()}