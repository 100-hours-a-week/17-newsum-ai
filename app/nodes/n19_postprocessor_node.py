# app/nodes/19_postprocessor_node.py (Improved Version)

import asyncio
import os
import hashlib
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple # Tuple 임포트
import aiohttp
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 기본값 참조용
from app.services.storage_client_v2 import StorageClient # S3 업로드용
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# 로거 설정
logger = get_logger(__name__)

# Pillow 의존성 처리
PILLOW_AVAILABLE = False
try:
    from PIL import Image, ImageDraw, ImageFont
    PILLOW_AVAILABLE = True
    logger.info("Pillow library found.")
except ImportError:
    Image, ImageDraw, ImageFont = None, None, None # type: ignore
    logger.error("Pillow library not installed. Image composition disabled.")


class PostprocessorNode:
    """
    생성된 이미지와 텍스트를 최종 만화 형식으로 후처리합니다.
    - 이미지 다운로드, 텍스트 선택, 이미지 합성(Pillow), 저장/업로드(StorageClient), ALT 텍스트 생성.
    - 설정은 `state.config` 우선, 없으면 `settings` 기본값 사용.
    """
    inputs: List[str] = ["image_urls", "scenarios", "translated_text", "comic_id", "chosen_idea", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["final_comic", "processing_stats", "error_message"]

    def __init__(self, storage_client: Optional[StorageClient] = None):
        self.storage_client = storage_client
        self.font_cache: Dict[Tuple[str, int], Optional[ImageFont.FreeTypeFont]] = {}
        logger.info(f"PostprocessorNode initialized {'with' if storage_client else 'without'} StorageClient.")
        if not PILLOW_AVAILABLE: logger.error("Image composition disabled.")

    def _load_runtime_config(self, config: Dict[str, Any]):
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
        logger.debug("Postprocessor runtime config loaded.")

    def _load_font(self, trace_id: Optional[str]) -> Optional[ImageFont.FreeTypeFont]:
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not PILLOW_AVAILABLE or not self.font_path: return None
        font_key = (self.font_path, self.font_size_ratio)
        if font_key in self.font_cache: return self.font_cache[font_key]
        if not os.path.exists(self.font_path):
            logger.error(f"{log_prefix} Font not found: {self.font_path}")
            self.font_cache[font_key] = None; return None
        try:
            font = ImageFont.truetype(self.font_path, size=20) # Load with base size
            logger.info(f"{log_prefix} Loaded font template: {self.font_path}")
            self.font_cache[font_key] = font; return font
        except Exception as e:
            logger.exception(f"{log_prefix} Failed to load font {self.font_path}: {e}")
            self.font_cache[font_key] = None; return None

    @tenacity.retry(
        stop=stop_after_attempt(settings.IMAGE_DOWNLOAD_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=tenacity.retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True
    )
    async def _download_image(self, url: str, session: aiohttp.ClientSession, trace_id: Optional[str]) -> Optional[bytes]:
        if not url: return None
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Downloading image: {url[:80]}...")
        try:
             async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.http_timeout)) as response:
                  response.raise_for_status(); content = await response.read()
                  logger.debug(f"{log_prefix} Downloaded {len(content)} bytes from {url[:80]}")
                  return content
        except RetryError as e: logger.error(f"{log_prefix} Download failed for {url[:80]} after retries: {e}"); raise
        except Exception as e: logger.error(f"{log_prefix} Download failed for {url[:80]}: {e}"); raise

    def _select_panel_text(self, panel_index: int, scenarios: List[Dict], translated_text: Optional[List[Dict]], trace_id: Optional[str]) -> str:
        # Node 18 로직과 동일/유사
        log_prefix = f"[{trace_id}]" if trace_id else ""
        try:
            original_dialogue = scenarios[panel_index].get('dialogue', '')
            scene_num = panel_index + 1
            if translated_text:
                for item in translated_text:
                    if isinstance(item, dict) and item.get('scene') == scene_num:
                        return item.get('translated_dialogue') if item.get('translated_dialogue') is not None else original_dialogue
                logger.warning(f"{log_prefix} Translation not found for panel {scene_num}, using original.")
            return original_dialogue
        except IndexError: logger.error(f"{log_prefix} Error accessing scenario panel {panel_index}."); return ""

    def _compose_comic(self, panel_images_bytes: List[Optional[bytes]], panel_texts: List[str], trace_id: Optional[str]) -> Optional[Image.Image]:
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not PILLOW_AVAILABLE: logger.error(f"{log_prefix} Pillow unavailable."); return None
        panel_images_pil = []
        for i, img_bytes in enumerate(panel_images_bytes):
            if img_bytes:
                try: panel_images_pil.append(Image.open(BytesIO(img_bytes)))
                except Exception as img_err: logger.error(f"{log_prefix} Load failed panel {i+1}: {img_err}"); return None
            else: logger.error(f"{log_prefix} Missing data panel {i+1}."); return None
        if len(panel_images_pil) != 4: logger.error(f"{log_prefix} Need 4 images, got {len(panel_images_pil)}."); return None

        logger.info(f"{log_prefix} Composing 2x2 comic...")
        try:
            panel_width = self.final_width // 2
            aspect_ratio = panel_images_pil[0].height / panel_images_pil[0].width if panel_images_pil[0].width > 0 else 1.0
            panel_height = int(panel_width * aspect_ratio)
            final_height = panel_height * 2
            resized_panels = [img.resize((panel_width, panel_height), Image.Resampling.LANCZOS) for img in panel_images_pil]
            final_image = Image.new('RGB', (self.final_width, final_height), color='white')
            final_image.paste(resized_panels[0], (0, 0)); final_image.paste(resized_panels[1], (panel_width, 0))
            final_image.paste(resized_panels[2], (0, panel_height)); final_image.paste(resized_panels[3], (panel_width, panel_height))

            font_template = self._load_font(trace_id)
            if font_template:
                 font_size = max(15, panel_height // self.font_size_ratio)
                 try: font = font_template.font_variant(size=font_size)
                 except AttributeError: font = ImageFont.truetype(font_template.path, size=font_size)
                 draw = ImageDraw.Draw(final_image)
                 positions = [(0, panel_height), (panel_width, panel_height), (0, final_height), (panel_width, final_height)]
                 padding = font_size // 2
                 for i, text in enumerate(panel_texts):
                      if text:
                           # TODO: Implement text wrapping for long dialogues
                           px, py_bottom = positions[i]
                           try:
                                bbox = draw.textbbox((0, 0), text, font=font)
                                tx = px + max(0, (panel_width - (bbox[2] - bbox[0])) // 2)
                                ty = py_bottom - (bbox[3] - bbox[1]) - padding
                                draw.text((tx, ty), text, fill=self.font_color, font=font)
                           except Exception as txt_err: logger.error(f"{log_prefix} Draw text err panel {i+1}: {txt_err}")
            else: logger.warning(f"{log_prefix} Font not loaded, skipping text overlay.")
            logger.info(f"{log_prefix} Comic composition complete.")
            return final_image
        except Exception as e: logger.exception(f"{log_prefix} Composition error: {e}"); return None

    async def _save_or_upload_image(self, image: Image.Image, comic_id: str, trace_id: Optional[str]) -> Optional[str]:
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if image is None: return None
        filename = f"{comic_id}.{self.output_format.lower()}"
        content_type = f"image/{self.output_format.lower()}"
        logger.info(f"{log_prefix} Saving/Uploading final comic: {filename}...")
        img_byte_arr = BytesIO()
        try:
            image.save(img_byte_arr, format=self.output_format, quality=self.output_quality)
            image_bytes = img_byte_arr.getvalue()
        except Exception as e: logger.exception(f"{log_prefix} Save to buffer failed: {e}"); return None

        if self.upload_to_s3 and self.storage_client:
            s3_key = f"comics/{filename}"
            try:
                # WARNING: Assumes storage_client has upload_bytes method.
                # If not, use the temporary file workaround commented below.
                logger.debug(f"{log_prefix} Uploading {len(image_bytes)} bytes to S3 key: {s3_key}")
                upload_result = await self.storage_client.upload_bytes(file_bytes=image_bytes, object_key=s3_key, content_type=content_type)
                # # --- Temp file workaround ---
                # temp_dir = self.save_dir or "."
                # temp_path = os.path.join(temp_dir, f"temp_{filename}")
                # os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                # with open(temp_path, "wb") as f: f.write(image_bytes)
                # logger.debug(f"{log_prefix} Using temp file {temp_path} for S3 upload.")
                # upload_result = await self.storage_client.upload_file(file_path=temp_path, object_key=s3_key, content_type=content_type)
                # try: os.remove(temp_path)
                # except OSError as rm_err: logger.warning(f"Could not remove temp file {temp_path}: {rm_err}")
                # # --- End temp file workaround ---

                if "error" in upload_result: logger.error(f"{log_prefix} S3 upload failed: {upload_result['error']}"); return None
                s3_url = upload_result.get("s3_url") # Or generate URL based on client response
                if s3_url: logger.info(f"{log_prefix} Uploaded to S3: {s3_url}"); return s3_url
                else: logger.error(f"{log_prefix} S3 upload OK but no URL returned."); return None
            except AttributeError: logger.error(f"{log_prefix} StorageClient lacks upload_bytes. S3 upload failed."); return None # Or implement temp file
            except Exception as e: logger.exception(f"{log_prefix} S3 upload error: {e}"); return None
        elif self.save_dir:
            try:
                os.makedirs(self.save_dir, exist_ok=True); filepath = os.path.join(self.save_dir, filename)
                with open(filepath, "wb") as f: f.write(image_bytes)
                logger.info(f"{log_prefix} Saved locally: {filepath}"); return filepath
            except Exception as e: logger.exception(f"{log_prefix} Local save failed: {e}"); return None
        else: logger.error(f"{log_prefix} Cannot save: S3 disabled/failed and no local dir."); return None

    def _generate_alt_text(self, scenarios: List[Dict], chosen_idea: Optional[Dict], trace_id: Optional[str]) -> str:
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Generating ALT text...")
        try:
            title = chosen_idea.get('idea_title', '4-panel comic') if chosen_idea else '4-panel comic'
            alt_parts = [f"{title}."]
            if scenarios and len(scenarios) == 4:
                for i, panel in enumerate(scenarios):
                    desc = panel.get('panel_description', '').strip()
                    dialogue = panel.get('dialogue', '').strip()
                    panel_alt = f"Panel {i+1}: {desc}" if desc else f"Panel {i+1} description."
                    if dialogue: panel_alt += f" Dialogue: '{dialogue}'"
                    alt_parts.append(panel_alt.rstrip('.') + ".")
            alt_text = " ".join(alt_parts)
            final_alt = self._truncate_text(alt_text, self.max_alt_text_len)
            logger.info(f"{log_prefix} Generated ALT text: '{final_alt[:100]}...'")
            return final_alt
        except Exception as e: logger.exception(f"{log_prefix} ALT text generation failed: {e}"); return "A 4-panel comic strip."

    def _truncate_text(self, text: Optional[str], max_length: int) -> str: # Helper duplication ok for clarity
        if not text: return ""
        return text[:max_length - 3] + "..." if len(text) > max_length else text

    async def run(self, state: ComicState) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        comic_id = state.comic_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing PostprocessorNode...")

        image_urls = state.image_urls or []
        scenarios = state.scenarios or []
        translated_text = state.translated_text
        chosen_idea = state.chosen_idea
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        self._load_runtime_config(config)
        final_comic_output = {"url": None, "alt_text": None}
        error_message: Optional[str] = None
        final_image_pil: Optional[Image.Image] = None

        if not PILLOW_AVAILABLE: error_message = "Pillow library unavailable."
        elif len(image_urls) != 4: error_message = f"Requires 4 images, got {len(image_urls)}."
        elif len(scenarios) != 4: error_message = "Requires 4 scenarios."

        if not error_message:
            # --- 1. Download Images ---
            logger.info(f"{log_prefix} Downloading panel images...")
            panel_images_bytes: List[Optional[bytes]] = [None] * 4
            download_errors = 0
            try:
                timeout = aiohttp.ClientTimeout(total=self.http_timeout * 2)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    tasks = [self._download_image(url, session, trace_id) for url in image_urls]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for i, res in enumerate(results):
                        if isinstance(res, bytes): panel_images_bytes[i] = res
                        else: download_errors += 1; logger.error(f"{log_prefix} DL fail panel {i+1}: {res}")
            except Exception as gather_err: error_message = f"Image download error: {gather_err}"

            if download_errors > 0 and not error_message: error_message = f"Failed to download {download_errors} images."

            # --- 2. Compose Image ---
            if not error_message:
                logger.info(f"{log_prefix} Composing final image...")
                panel_texts = [self._select_panel_text(i, scenarios, translated_text, trace_id) for i in range(4)]
                loop = asyncio.get_running_loop()
                try:
                    final_image_pil = await loop.run_in_executor(None, self._compose_comic, panel_images_bytes, panel_texts, trace_id)
                    if final_image_pil is None: error_message = "Comic composition failed."
                except Exception as compose_err: error_message = f"Composition error: {compose_err}"

            # --- 3. Save/Upload Image ---
            if not error_message and final_image_pil:
                logger.info(f"{log_prefix} Saving/Uploading final image...")
                output_url = await self._save_or_upload_image(final_image_pil, comic_id, trace_id)
                if output_url: final_comic_output['url'] = output_url
                else: error_message = "Failed to save or upload final comic."

            # --- 4. Generate ALT Text ---
            if not error_message and final_comic_output.get('url'):
                 final_comic_output['alt_text'] = self._generate_alt_text(scenarios, chosen_idea, trace_id)

        if error_message: logger.error(f"{log_prefix} Postprocessing failed: {error_message}")

        end_time = datetime.now(timezone.utc)
        processing_stats['postprocessor_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} PostprocessorNode finished in {processing_stats['postprocessor_node_time']:.2f} seconds.")

        update_data: Dict[str, Any] = {
            "final_comic": final_comic_output,
            "processing_stats": processing_stats,
            "error_message": error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}