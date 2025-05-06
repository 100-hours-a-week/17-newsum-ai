# app/nodes/17_imager_node.py (Refactored)

import asyncio
import re
import json
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.services.image_server_client_v2 import ImageGenerationClient
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

class ImagerNode:
    """
    시나리오 각 패널에 대한 이미지를 생성합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["scenarios", "trace_id", "comic_id", "config"]
    outputs: List[str] = ["image_urls", "node17_processing_stats", "error_message"]

    def __init__(self, image_client: ImageGenerationClient):
        if not image_client: raise ValueError("ImageGenerationClient is required.")
        self.image_client = image_client
        logger.info("ImagerNode initialized.")

    # --- MODIFIED: Added extra_log_data argument ---
    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.image_model = config.get("image_model", settings.DEFAULT_IMAGE_MODEL)
        self.img_height = int(config.get("image_height", settings.DEFAULT_IMAGE_HEIGHT))
        self.img_width = int(config.get("image_width", settings.DEFAULT_IMAGE_WIDTH))
        self.negative_prompt = config.get("image_negative_prompt", settings.DEFAULT_IMAGE_NEGATIVE_PROMPT)
        self.style_preset = config.get("image_style_preset", settings.DEFAULT_IMAGE_STYLE_PRESET)
        self.default_style = config.get("image_default_style", settings.DEFAULT_IMAGE_STYLE)
        self.max_prompt_len = int(config.get("max_image_prompt_len", settings.DEFAULT_MAX_IMAGE_PROMPT_LEN))
        self.enable_controlnet = config.get("enable_controlnet", settings.ENABLE_CONTROLNET)
        self.controlnet_type = config.get("controlnet_type", settings.DEFAULT_CONTROLNET_TYPE)
        self.controlnet_weight = float(config.get("controlnet_weight", settings.DEFAULT_CONTROLNET_WEIGHT))

        logger.debug(f"Runtime config loaded. Model: {self.image_model}, Size: {self.img_width}x{self.img_height}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Style Preset: {self.style_preset}, Default Style: {self.default_style}", extra=extra_log_data) # MODIFIED
        logger.debug(f"ControlNet Enabled: {self.enable_controlnet}, Type: {self.controlnet_type}, Weight: {self.controlnet_weight}", extra=extra_log_data) # MODIFIED

    def _construct_image_prompt(self, panel_data: Dict[str, Any]) -> str:
        """이미지 생성 API를 위한 프롬프트 구성"""
        description = panel_data.get('panel_description', '').strip()
        tags = panel_data.get('seed_tags', [])

        prompt_parts = [description]
        # Ensure tags is a list of non-empty strings
        if tags and isinstance(tags, list):
            valid_tags = [str(tag).strip() for tag in tags if isinstance(tag, (str, int, float)) and str(tag).strip()]
            if valid_tags:
                shuffled_tags = random.sample(valid_tags, len(valid_tags))
                prompt_parts.append(", ".join(shuffled_tags))
        if self.default_style and isinstance(self.default_style, str): # Add style if valid
            prompt_parts.append(self.default_style)

        full_prompt = re.sub(r'\s+', ' ', ", ".join(part for part in prompt_parts if part)).strip()
        # Truncate if needed
        truncated_prompt = full_prompt[:self.max_prompt_len]
        if len(full_prompt) > self.max_prompt_len:
             truncated_prompt += "..."
             # logger.debug(f"Image prompt truncated to {self.max_prompt_len} chars.")
        return truncated_prompt

    # --- MODIFIED: Added extra_log_data argument ---
    def _get_controlnet_input(self, panel_index: int, previous_image_url: Optional[str], panel_data: Dict[str, Any], trace_id: Optional[str], comic_id: Optional[str]) -> Optional[Dict]:
        """ControlNet 입력 결정 (Placeholder)"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'panel_index': panel_index + 1} # MODIFIED
        if not self.enable_controlnet: return None

        if self.controlnet_type == "previous_panel" and panel_index > 0 and previous_image_url:
            logger.debug("Using previous panel image for ControlNet.", extra=extra_log_data) # MODIFIED
            return {"control_type": "reference", "image_url": previous_image_url, "weight": self.controlnet_weight}
        # Add other ControlNet types here if needed
        else:
            # logger.debug("No applicable ControlNet strategy found.", extra=extra_log_data) # MODIFIED
            return None

    @tenacity.retry(
        stop=stop_after_attempt(settings.IMAGE_API_RETRIES),
        wait=wait_fixed(2) + wait_exponential(multiplier=1.5, max=15),
        retry=tenacity.retry_if_exception_type(Exception),
        # MODIFIED: Pass trace_id/comic_id to before_sleep logger
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying image gen (Attempt {retry_state.attempt_number}, PanelIdx: {retry_state.args[1]}). Error: {retry_state.outcome.exception()}",
            extra={'trace_id': retry_state.args[3], 'comic_id': getattr(retry_state.args[0], 'comic_id', 'N/A')} # Access comic_id if possible
        ),
        reraise=True
    )
    async def _generate_panel_image(
        self, panel_data: Dict[str, Any], panel_index: int, previous_image_url: Optional[str],
        trace_id: Optional[str], comic_id: Optional[str] # MODIFIED: Added comic_id
        ) -> Optional[str]:
        """단일 패널 이미지 생성 (실제 클라이언트 사용)"""
        image_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'panel_index': panel_index + 1} # MODIFIED
        logger.info("Starting image generation...", extra=image_log_data) # MODIFIED

        prompt = self._construct_image_prompt(panel_data)
        logger.debug(f"Prompt: {prompt}", extra=image_log_data) # MODIFIED
        if not prompt:
             logger.warning("Prompt is empty. Skipping image generation.", extra=image_log_data) # MODIFIED
             return None

        # Pass IDs
        controlnet_params = self._get_controlnet_input(panel_index, previous_image_url, panel_data, trace_id, comic_id)

        try:
            # Pass IDs if client supports them
            result = await self.image_client.generate_image(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                model=self.image_model,
                height=self.img_height,
                width=self.img_width,
                style_preset=self.style_preset,
                controlnet_params=controlnet_params,
                seed=random.randint(0, 2**32 - 1),
                # Pass trace_id, comic_id if client supports them
                # trace_id=trace_id,
                # comic_id=comic_id
            )

            # Validate result structure
            if not isinstance(result, dict):
                logger.error(f"Image generation API returned non-dict result: {type(result)}", extra=image_log_data) # MODIFIED
                raise ValueError("Image generation API returned unexpected type")

            if "error" in result:
                 error_msg = result['error']
                 logger.error(f"Image generation API error: {error_msg}", extra=image_log_data) # MODIFIED
                 raise RuntimeError(f"Image generation failed: {error_msg}")
            elif "image_url" in result and result["image_url"] and isinstance(result["image_url"], str): # Check type
                 image_url = result["image_url"]
                 logger.info(f"Image generated successfully (URL): {image_url[:80]}...", extra=image_log_data) # MODIFIED
                 return image_url
            elif "image_path" in result and result["image_path"] and isinstance(result["image_path"], str): # Check type
                 image_path = result["image_path"]
                 logger.info(f"Image generated successfully (Path): {image_path}", extra=image_log_data) # MODIFIED
                 return image_path
            else:
                 logger.error(f"Image generation API returned unexpected result structure: {result}", extra=image_log_data) # MODIFIED
                 raise ValueError("Image generation returned no valid URL or path")

        except RetryError as e:
             logger.error(f"Image generation failed after multiple retries: {e}", extra=image_log_data) # MODIFIED
             raise # Re-raise for the main run method to catch
        except Exception as e:
            logger.error(f"Image generation attempt failed: {e.__class__.__name__}", exc_info=True, extra=image_log_data) # MODIFIED
            raise # Re-raise for retry or main run method

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """모든 시나리오 패널에 대한 이미지 생성 실행"""
        start_time = datetime.now(timezone.utc)
        # --- MODIFIED: Get trace_id and comic_id safely ---
        comic_id = getattr(state, 'comic_id', 'unknown_comic')
        trace_id = getattr(state, 'trace_id', comic_id)
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        # -------------------------------------------------

        node_class_name = self.__class__.__name__
        # --- ADDED: Start Logging ---
        logger.info(f"--- Executing {node_class_name} ---", extra=extra_log_data)
        logger.debug(f"Entering state:\n{summarize_for_logging(state, is_state_object=True)}", extra=extra_log_data)
        # --------------------------

        scenarios = getattr(state, 'scenarios', None) # Safe access
        config = getattr(state, 'config', {}) or {}

        # --- ADDED: Input Validation ---
        if not scenarios or not isinstance(scenarios, list) or len(scenarios) != 4 or not all(isinstance(p, dict) for p in scenarios):
            error_message = "Valid 4-panel 'scenarios' list of dictionaries is required for image generation."
            logger.error(error_message, extra=extra_log_data)
            end_time = datetime.now(timezone.utc)
            node17_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "image_urls": [],
                "node17_processing_stats": node17_processing_stats,
                "error_message": error_message
            }
            # --- ADDED: End Logging (Error Case) ---
            logger.debug(f"Returning updates (invalid scenarios):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Error: Invalid Scenarios) --- (Elapsed: {node17_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        # --------------------------

        logger.info(f"Starting image generation for {len(scenarios)} panels...", extra=extra_log_data)

        generated_image_outputs: List[Optional[str]] = [None] * len(scenarios)
        previous_image_output: Optional[str] = None
        task_errors: List[str] = []

        # Generate sequentially to allow ControlNet dependency
        for i, panel_data in enumerate(scenarios):
            # Ensure panel_data is a dict
            if not isinstance(panel_data, dict):
                msg = f"Panel {i+1} data is not a dictionary. Skipping."
                logger.error(msg, extra=extra_log_data)
                task_errors.append(msg)
                generated_image_outputs[i] = None
                continue

            try:
                # Pass IDs
                image_output = await self._generate_panel_image(
                    panel_data, i, previous_image_output, trace_id, comic_id
                )
                generated_image_outputs[i] = image_output
                previous_image_output = image_output
            except Exception as e:
                # Error logged within _generate_panel_image or its retry handler
                msg = f"Panel {i+1} generation ultimately failed: {e.__class__.__name__}"
                task_errors.append(msg)
                generated_image_outputs[i] = None
                # Decide if failure should prevent using this image for next panel's ControlNet
                # previous_image_output = None # Option 1: Reset ControlNet input on failure
                # previous_image_output = previous_image_output # Option 2: Keep using last successful (current behavior)

        final_image_outputs = [url for url in generated_image_outputs if url is not None]
        failed_panels = len(scenarios) - len(final_image_outputs)
        logger.info(f"Image generation complete. Success: {len(final_image_outputs)}, Failed: {failed_panels}.", extra=extra_log_data)

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"Errors during image generation: {final_error_message}", extra=extra_log_data)

        end_time = datetime.now(timezone.utc)
        node17_processing_stats = (end_time - start_time).total_seconds()

        update_data: Dict[str, Any] = {
            "image_urls": final_image_outputs,
            "node17_processing_stats": node17_processing_stats,
            "error_message": final_error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if final_error_message or failed_panels > 0 else logger.info
        log_level(f"Image generation result: {len(final_image_outputs)}/{len(scenarios)} images generated. Errors: {final_error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node17_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}