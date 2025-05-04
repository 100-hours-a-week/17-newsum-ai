# app/nodes/19_postprocessor_node.py (Refactored)

import asyncio
import os
import hashlib
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import aiohttp
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.services.storage_client_v2 import StorageClient # S3 업로드용
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

PILLOW_AVAILABLE = False
try:
    from PIL import Image, ImageDraw, ImageFont
    PILLOW_AVAILABLE = True
    logger.info("Pillow library found.")
except ImportError:
    Image, ImageDraw, ImageFont = None, None, None
    logger.error("Pillow library not installed. Image composition disabled.")

class PostprocessorNode:
    """
    생성된 이미지와 텍스트를 최종 만화 형식으로 후처리합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = [ # MODIFIED: Added comic_id, trace_id
        "image_urls", "scenarios", "translated_text", "comic_id", "chosen_idea",
        "trace_id", "config"
    ]
    outputs: List[str] = ["final_comic", "node19_processing_stats", "error_message"]

    def __init__(self, storage_client: Optional[StorageClient] = None):
        self.storage_client = storage_client
        self.font_cache: Dict[Tuple[str, int], Optional[ImageFont.FreeTypeFont]] = {}
        logger.info(f"PostprocessorNode initialized {'with' if storage_client else 'without'} StorageClient.")
        if not PILLOW_AVAILABLE: logger.error("Image composition disabled.")

    # --- MODIFIED: Added extra_log_data argument ---
    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.final_width = int(config.get("final_image_width", settings.DEFAULT_FINAL_IMAGE_WIDTH))
        self.font_path = config.get("default_font_path", settings.DEFAULT_FONT_PATH)
        self.font_size_ratio = int(config.get("text_overlay_font_size_ratio", settings.DEFAULT_FONT_SIZE_RATIO))
        self.font_color = config.get("text_overlay_color", settings.DEFAULT_FONT_COLOR)
        self.upload_to_s3 = config.get("upload_to_s3", settings.UPLOAD_TO_S3)
        self.save_dir = config.get("final_comic_save_dir", settings.DEFAULT_COMIC_SAVE_DIR)
        self.output_format = config.get("final_image_format", settings.DEFAULT_IMAGE_FORMAT).upper()
        self.output_quality = int(config.get("final_image_quality", settings.DEFAULT_IMAGE_QUALITY))
        self.http_timeout = config.get('http_timeout', settings.DEFAULT_HTTP_TIMEOUT)
        self.max_alt_text_len = int(config.get("max_alt_text_len", settings.DEFAULT_MAX_ALT_TEXT_LEN))
        logger.debug("Postprocessor runtime config loaded.", extra=extra_log_data) # MODIFIED

    # --- MODIFIED: Added extra_log_data argument ---
    def _load_font(self, trace_id: Optional[str], comic_id: Optional[str]) -> Optional[ImageFont.FreeTypeFont]:
        """폰트 로드 (캐싱 포함)"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not PILLOW_AVAILABLE or not self.font_path: return None
        font_key = (self.font_path, self.font_size_ratio) # Key depends on path and base ratio/size
        if font_key in self.font_cache: return self.font_cache[font_key]

        if not os.path.exists(self.font_path):
            logger.error(f"Font not found: {self.font_path}", extra=extra_log_data) # MODIFIED
            self.font_cache[font_key] = None; return None
        try:
            # Load with a reasonable base size (e.g., 20) - actual size adjusted later
            font_template = ImageFont.truetype(self.font_path, size=20)
            logger.info(f"Loaded font template: {self.font_path}", extra=extra_log_data) # MODIFIED
            self.font_cache[font_key] = font_template; return font_template
        except Exception as e:
            logger.exception(f"Failed to load font {self.font_path}", extra=extra_log_data) # MODIFIED
            self.font_cache[font_key] = None; return None

    @tenacity.retry(
        stop=stop_after_attempt(settings.IMAGE_DOWNLOAD_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=tenacity.retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True
    )
    async def _download_image(self, url: str, session: aiohttp.ClientSession, trace_id: Optional[str], comic_id: Optional[str]) -> Optional[bytes]: # MODIFIED: Added comic_id
        """이미지 다운로드 (재시도 포함)"""
        if not url or not isinstance(url, str): return None # Added type check
        dl_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url[:80]} # MODIFIED
        logger.debug("Downloading image...", extra=dl_log_data) # MODIFIED
        try:
             # Increased timeout slightly for downloads
             async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.http_timeout * 1.5)) as response:
                  response.raise_for_status()
                  content = await response.read()
                  logger.debug(f"Downloaded {len(content)} bytes.", extra=dl_log_data) # MODIFIED
                  return content
        except RetryError as e:
             logger.error(f"Download failed after retries: {e}", extra=dl_log_data) # MODIFIED
             raise # Re-raise for gather to catch
        except Exception as e:
             logger.error(f"Download failed: {e}", exc_info=True, extra=dl_log_data) # MODIFIED
             raise # Re-raise for gather/retry

    # --- MODIFIED: Added extra_log_data argument ---
    def _select_panel_text(self, panel_index: int, scenarios: List[Dict], translated_text: Optional[List[Dict]], trace_id: Optional[str], comic_id: Optional[str]) -> str:
        """번역된 텍스트가 있으면 사용, 없으면 원본 사용"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'panel_index': panel_index + 1} # MODIFIED
        original_dialogue = ""
        try:
            if scenarios and isinstance(scenarios, list) and panel_index < len(scenarios) and isinstance(scenarios[panel_index], dict):
                original_dialogue = scenarios[panel_index].get('dialogue', '') or "" # Default to empty string
            else:
                 logger.warning("Invalid scenarios data or index.", extra=extra_log_data) # MODIFIED
                 return "" # Return empty if scenario data invalid

            # Check translated_text only if it's a list
            if translated_text and isinstance(translated_text, list):
                scene_num_to_find = panel_index + 1
                for item in translated_text:
                    # Check item structure and scene number
                    if isinstance(item, dict) and item.get('scene') == scene_num_to_find:
                        # Use translated text if it's a non-empty string
                        trans_dialogue = item.get('translated_dialogue')
                        if isinstance(trans_dialogue, str) and trans_dialogue.strip():
                            # logger.debug("Using translated dialogue.", extra=extra_log_data) # MODIFIED
                            return trans_dialogue
                        elif trans_dialogue == "": # Explicitly empty translation
                             # logger.debug("Using empty translated dialogue.", extra=extra_log_data) # MODIFIED
                             return ""
                        else: # Translation failed (None) or missing
                             logger.warning("Translation failed or missing for panel, using original.", extra=extra_log_data) # MODIFIED
                             return original_dialogue # Fallback to original
                # Loop finished without finding matching scene
                # logger.debug("Translation not found for panel, using original.", extra=extra_log_data) # MODIFIED
                return original_dialogue
            else: # No translation data provided
                 # logger.debug("No translation data, using original dialogue.", extra=extra_log_data) # MODIFIED
                 return original_dialogue
        except Exception as e: # Catch unexpected errors during access
             logger.exception("Error selecting panel text.", extra=extra_log_data) # MODIFIED use exception
             return "" # Return empty on error

    # --- MODIFIED: Added extra_log_data argument ---
    def _compose_comic(self, panel_images_bytes: List[Optional[bytes]], panel_texts: List[str], trace_id: Optional[str], comic_id: Optional[str]) -> Optional[Image.Image]:
        """4컷 만화 이미지 합성"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not PILLOW_AVAILABLE: logger.error("Pillow unavailable.", extra=extra_log_data); return None # MODIFIED

        panel_images_pil = []
        # Validate input bytes and load images
        for i, img_bytes in enumerate(panel_images_bytes):
            if isinstance(img_bytes, bytes):
                try:
                    img = Image.open(BytesIO(img_bytes))
                    # Ensure image is in RGB for consistent processing
                    panel_images_pil.append(img.convert("RGB") if img.mode != 'RGB' else img)
                except Exception as img_err:
                    logger.error(f"Failed to load image bytes for panel {i+1}: {img_err}", extra=extra_log_data) # MODIFIED
                    return None # Cannot proceed if any image fails to load
            else:
                logger.error(f"Missing image data for panel {i+1}.", extra=extra_log_data) # MODIFIED
                return None # Cannot proceed

        if len(panel_images_pil) != 4:
            logger.error(f"Incorrect number of valid images loaded ({len(panel_images_pil)}), need 4.", extra=extra_log_data) # MODIFIED
            return None

        logger.info("Composing 2x2 comic...", extra=extra_log_data) # MODIFIED
        try:
            # Assume all panels have same original dimensions after generation (or use first panel's)
            orig_width, orig_height = panel_images_pil[0].size
            if orig_width <= 0 or orig_height <= 0: raise ValueError("Invalid panel dimensions")

            panel_width = self.final_width // 2
            aspect_ratio = orig_height / orig_width
            panel_height = int(panel_width * aspect_ratio)
            final_height = panel_height * 2

            resized_panels = [img.resize((panel_width, panel_height), Image.Resampling.LANCZOS) for img in panel_images_pil]
            final_image = Image.new('RGB', (self.final_width, final_height), color='white')

            # Paste panels
            final_image.paste(resized_panels[0], (0, 0))
            final_image.paste(resized_panels[1], (panel_width, 0))
            final_image.paste(resized_panels[2], (0, panel_height))
            final_image.paste(resized_panels[3], (panel_width, panel_height))

            # Add text overlay
            # Pass IDs
            font_template = self._load_font(trace_id, comic_id)
            if font_template:
                 # Dynamic font size based on panel height
                 font_size = max(10, panel_height // self.font_size_ratio) # Ensure minimum size
                 try:
                     # Use font_variant if available, otherwise reload with size
                     try: font = font_template.font_variant(size=font_size)
                     except AttributeError: font = ImageFont.truetype(font_template.path, size=font_size)

                     draw = ImageDraw.Draw(final_image)
                     positions = [(0, 0), (panel_width, 0), (0, panel_height), (panel_width, panel_height)] # Top-left corners
                     padding = font_size // 4 # Smaller padding

                     for i, text in enumerate(panel_texts):
                          if text and isinstance(text, str): # Check text validity
                               px, py_top = positions[i]
                               # Simplified text positioning: bottom-center within panel
                               try:
                                    bbox = draw.textbbox((0, 0), text, font=font) # Estimate size
                                    text_width = bbox[2] - bbox[0]
                                    text_height = bbox[3] - bbox[1]
                                    # Center horizontally, place near bottom vertically
                                    tx = px + max(padding, (panel_width - text_width) // 2)
                                    ty = py_top + panel_height - text_height - padding
                                    draw.text((tx, ty), text, fill=self.font_color, font=font)
                               except Exception as txt_err:
                                    logger.error(f"Draw text error panel {i+1}: {txt_err}", extra=extra_log_data) # MODIFIED
                 except Exception as font_err:
                     logger.error(f"Failed to set font size {font_size}: {font_err}", extra=extra_log_data) # MODIFIED
            else: logger.warning("Font not loaded, skipping text overlay.", extra=extra_log_data) # MODIFIED

            logger.info("Comic composition complete.", extra=extra_log_data) # MODIFIED
            return final_image
        except Exception as e:
            logger.exception("Error during comic composition.", extra=extra_log_data) # MODIFIED use exception
            return None

    # --- MODIFIED: Added extra_log_data argument ---
    async def _save_or_upload_image(self, image: Image.Image, comic_id: str, trace_id: Optional[str]) -> Optional[str]:
        """최종 이미지 저장 또는 업로드"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if image is None: return None
        filename = f"{comic_id}.{self.output_format.lower()}"
        content_type = f"image/{self.output_format.lower()}"
        logger.info(f"Saving/Uploading final comic: {filename}...", extra=extra_log_data) # MODIFIED
        img_byte_arr = BytesIO()
        try:
            # Ensure format is supported by Pillow and valid
            save_format = self.output_format if self.output_format in Image.SAVE else 'PNG'
            image.save(img_byte_arr, format=save_format, quality=self.output_quality)
            image_bytes = img_byte_arr.getvalue()
            if not image_bytes: raise ValueError("Image save resulted in empty bytes.")
        except Exception as e:
            logger.exception(f"Failed to save image to buffer: {e}", extra=extra_log_data) # MODIFIED use exception
            return None

        output_location: Optional[str] = None
        # Try S3 upload first if enabled and client available
        if self.upload_to_s3 and self.storage_client:
            s3_key = f"comics/{filename}" # Example path
            try:
                logger.debug(f"Attempting S3 upload: {s3_key} ({len(image_bytes)} bytes)", extra=extra_log_data) # MODIFIED
                # Pass trace_id/comic_id if client supports them
                upload_result = await self.storage_client.upload_bytes(
                    file_bytes=image_bytes, object_key=s3_key, content_type=content_type #, trace_id=trace_id
                )
                if upload_result and isinstance(upload_result, dict) and not upload_result.get("error"):
                    s3_url = upload_result.get("s3_url") # Assuming client returns URL
                    if s3_url:
                         logger.info(f"Successfully uploaded to S3: {s3_url}", extra=extra_log_data) # MODIFIED
                         output_location = s3_url
                    else:
                         logger.warning("S3 upload successful but no URL returned.", extra=extra_log_data) # MODIFIED
                         # Could fallback to local save here if needed
                else:
                     err = upload_result.get("error", "Unknown S3 error") if isinstance(upload_result, dict) else "Invalid upload result"
                     logger.error(f"S3 upload failed: {err}", extra=extra_log_data) # MODIFIED
            except AttributeError:
                 logger.error("StorageClient missing 'upload_bytes' method. S3 upload skipped.", extra=extra_log_data) # MODIFIED
            except Exception as e:
                 logger.exception("S3 upload encountered an error.", extra=extra_log_data) # MODIFIED use exception

        # Fallback to local save if S3 failed or disabled, and local dir is set
        if not output_location and self.save_dir and isinstance(self.save_dir, str):
            try:
                os.makedirs(self.save_dir, exist_ok=True)
                filepath = os.path.join(self.save_dir, filename)
                with open(filepath, "wb") as f: f.write(image_bytes)
                logger.info(f"Successfully saved locally: {filepath}", extra=extra_log_data) # MODIFIED
                output_location = filepath # Return local path
            except Exception as e:
                logger.exception(f"Local save failed: {e}", extra=extra_log_data) # MODIFIED use exception

        if not output_location:
             logger.error("Failed to save or upload final comic image.", extra=extra_log_data) # MODIFIED

        return output_location

    # --- MODIFIED: Added extra_log_data argument ---
    def _generate_alt_text(self, scenarios: List[Dict], chosen_idea: Optional[Dict], trace_id: Optional[str], comic_id: Optional[str]) -> str:
        """ALT 텍스트 생성"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        logger.debug("Generating ALT text...", extra=extra_log_data) # MODIFIED
        default_alt = "A 4-panel comic strip."
        try:
            # Ensure chosen_idea is a dict before accessing
            title = chosen_idea.get('idea_title', default_alt) if isinstance(chosen_idea, dict) else default_alt
            alt_parts = [title.rstrip('.') + "."] # Start with title

            # Ensure scenarios is a list of 4 dicts
            if scenarios and isinstance(scenarios, list) and len(scenarios) == 4 and all(isinstance(p, dict) for p in scenarios):
                for i, panel in enumerate(scenarios):
                    desc = panel.get('panel_description', '').strip()
                    dialogue = panel.get('dialogue', '').strip()
                    panel_alt = f"Panel {i+1}: {desc}" if desc else f"Panel {i+1}." # Default description if missing
                    if dialogue: panel_alt += f" Dialogue: '{dialogue}'"
                    alt_parts.append(panel_alt.rstrip('.') + ".") # Ensure period separation
            else:
                 logger.warning("Invalid scenarios data for ALT text generation.", extra=extra_log_data) # MODIFIED

            alt_text = " ".join(alt_parts)
            # Truncate using helper
            final_alt = self._truncate_text(alt_text, self.max_alt_text_len)
            logger.info(f"Generated ALT text: '{final_alt[:100]}...'", extra=extra_log_data) # MODIFIED
            return final_alt if final_alt else default_alt
        except Exception as e:
            logger.exception("ALT text generation failed.", extra=extra_log_data) # MODIFIED use exception
            return default_alt

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """이미지 다운로드, 합성, 저장/업로드 및 ALT 텍스트 생성 실행"""
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

        image_urls = getattr(state, 'image_urls', []) or []
        scenarios = getattr(state, 'scenarios', []) or []
        # Allow translated_text to be None
        translated_text = getattr(state, 'translated_text', None)
        chosen_idea = getattr(state, 'chosen_idea', None)
        config = getattr(state, 'config', {}) or {}

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        # --------------------------

        final_comic_output = {"url": None, "alt_text": None}
        error_message: Optional[str] = None
        final_image_pil: Optional[Image.Image] = None
        task_errors: List[str] = [] # Collect errors from steps

        # --- Input Validation and Dependency Check ---
        if not PILLOW_AVAILABLE:
            error_message = "Pillow library unavailable, cannot compose image."
            logger.error(error_message, extra=extra_log_data)
        elif not image_urls or not isinstance(image_urls, list) or len(image_urls) != 4 or not all(isinstance(url, str) for url in image_urls):
            error_message = f"Requires 4 valid image URLs, found {len(image_urls)}."
            logger.error(error_message, extra=extra_log_data)
        elif not scenarios or not isinstance(scenarios, list) or len(scenarios) != 4 or not all(isinstance(p, dict) for p in scenarios):
            error_message = "Requires 4 valid scenario panel dictionaries."
            logger.error(error_message, extra=extra_log_data)
        # ---------------------------------------------

        if not error_message:
            try:
                # --- 1. Download Images ---
                logger.info("Downloading panel images...", extra=extra_log_data)
                panel_images_bytes: List[Optional[bytes]] = [None] * 4
                download_errors = 0
                timeout = aiohttp.ClientTimeout(total=self.http_timeout * 1.5) # Slightly longer timeout for downloads
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    tasks = [self._download_image(url, session, trace_id, comic_id) for url in image_urls] # Pass IDs
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for i, res in enumerate(results):
                        if isinstance(res, bytes):
                            panel_images_bytes[i] = res
                        else:
                            download_errors += 1
                            err_msg = f"Download failed for panel {i+1}: {res}"
                            logger.error(err_msg, exc_info=isinstance(res, Exception) and res, extra=extra_log_data)
                            task_errors.append(f"Image DL Panel {i+1} Error: {res}")

                if download_errors > 0:
                    # Proceed if some images downloaded, but log errors
                    logger.warning(f"Failed to download {download_errors} panel images.", extra=extra_log_data)
                    if download_errors == 4: # If all failed, cannot continue
                         raise ValueError("All image downloads failed.")

                # --- 2. Compose Image ---
                # Proceed only if we have exactly 4 byte arrays (even if some are None initially, but filtered later)
                if len(panel_images_bytes) == 4:
                    logger.info("Composing final image...", extra=extra_log_data)
                    # Select text *before* potentially slow composition
                    panel_texts = [self._select_panel_text(i, scenarios, translated_text, trace_id, comic_id) for i in range(4)] # Pass IDs
                    loop = asyncio.get_running_loop()
                    # Pass IDs
                    final_image_pil = await loop.run_in_executor(None, self._compose_comic, panel_images_bytes, panel_texts, trace_id, comic_id)
                    if final_image_pil is None:
                        task_errors.append("Comic composition failed.")
                        logger.error("Comic composition failed.", extra=extra_log_data)
                else:
                    task_errors.append("Incorrect number of images available for composition.")
                    logger.error("Incorrect number of images available for composition after download attempt.", extra=extra_log_data)


                # --- 3. Save/Upload Image ---
                if final_image_pil: # Check if composition succeeded
                    logger.info("Saving/Uploading final image...", extra=extra_log_data)
                    # Pass IDs
                    output_url = await self._save_or_upload_image(final_image_pil, comic_id, trace_id)
                    if output_url:
                        final_comic_output['url'] = output_url
                    else:
                        task_errors.append("Failed to save or upload final comic.")
                        logger.error("Failed to save or upload final comic image.", extra=extra_log_data)
                elif "Comic composition failed." not in task_errors: # Avoid duplicate error if composition already failed
                     task_errors.append("Skipping save/upload due to composition failure.")


                # --- 4. Generate ALT Text ---
                # Generate ALT text even if image failed, might still be useful info
                # Pass IDs
                final_comic_output['alt_text'] = self._generate_alt_text(scenarios, chosen_idea, trace_id, comic_id)

            except Exception as e:
                # Catch unexpected errors during the process
                logger.exception("Unexpected error during postprocessing.", extra=extra_log_data) # Use exception
                error_message = f"Unexpected postprocessing error: {e}"

        # Aggregate final error message
        if task_errors and not error_message: error_message = "; ".join(task_errors)
        elif task_errors and error_message: error_message = f"{error_message}; {'; '.join(task_errors)}"

        if error_message:
             logger.error(f"Postprocessing finished with errors: {error_message}", extra=extra_log_data)

        end_time = datetime.now(timezone.utc)
        node19_processing_stats = (end_time - start_time).total_seconds()

        update_data: Dict[str, Any] = {
            "final_comic": final_comic_output, # Return dict even if values are None
            "node19_processing_stats": node19_processing_stats,
            "error_message": error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if error_message else logger.info
        log_level(f"Postprocessing result: Output URL {'Generated' if final_comic_output['url'] else 'Failed'}. Errors: {error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node19_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}