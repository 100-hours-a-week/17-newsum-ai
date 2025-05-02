# app/nodes/19_postprocessor_node.py

import asyncio
import os
import hashlib
from io import BytesIO
# --- datetime, timezone 임포트 추가 ---
from datetime import datetime, timezone
# ------------------------------------
from typing import List, Dict, Any, Optional
import aiohttp # 이미지 다운로드용
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 프로젝트 구성 요소 임포트 ---
from app.config.settings import settings # 재시도 횟수 등 참조
from app.services.storage_client_v2 import StorageClient # S3 업로드용 클라이언트
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# --- 로거 설정 ---
logger = get_logger("PostprocessorNode")

# --- Pillow 의존성 처리 ---
PILLOW_AVAILABLE = False
try:
    from PIL import Image, ImageDraw, ImageFont
    PILLOW_AVAILABLE = True
    logger.info("Pillow library found.")
except ImportError:
    logger.error("Pillow library not installed. Image composition is disabled.")

class PostprocessorNode:
    """
    (Refactored) 생성된 이미지와 텍스트를 최종 만화 형식으로 후처리합니다.
    - 패널 이미지 다운로드 (aiohttp).
    - 텍스트 선택 (원본/번역).
    - 이미지 합성 및 텍스트 오버레이 (Pillow).
    - 최종 만화 저장(로컬) 또는 업로드(S3 - StorageClient 사용).
    - ALT 텍스트 생성.
    - 설정은 상태의 config 딕셔너리에서 로드.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["image_urls", "scenarios", "translated_text", "comic_id", "chosen_idea", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["final_comic", "processing_stats", "error_message"]

    # StorageClient 인스턴스를 외부에서 주입받음 (S3 업로드 시 필요)
    def __init__(
        self,
        storage_client: Optional[StorageClient] = None, # S3 업로드 시 필수
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        self.storage_client = storage_client
        # self.langsmith_service = langsmith_service
        self.font: Optional[ImageFont.FreeTypeFont] = None # 폰트 객체 캐싱용
        logger.info(f"PostprocessorNode initialized {'with' if storage_client else 'without'} StorageClient.")
        if not PILLOW_AVAILABLE:
            logger.error("Image composition disabled due to missing Pillow library.")

    def _load_font(self, font_path: Optional[str], font_size: int, trace_id: Optional[str]) -> Optional[ImageFont.FreeTypeFont]:
        """폰트 파일 로드 (캐싱 시도)"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not PILLOW_AVAILABLE or not font_path:
            if not font_path: logger.warning(f"{log_prefix} Font path not configured. Text overlay disabled.")
            return None

        # 간단한 캐싱: 경로와 크기가 같으면 재사용
        if self.font and self.font.path == font_path and self.font.size == font_size:
             return self.font

        if not os.path.exists(font_path):
            logger.error(f"{log_prefix} Font file not found at: {font_path}. Text overlay disabled.")
            self.font = None
            return None
        try:
            self.font = ImageFont.truetype(font_path, font_size)
            logger.info(f"{log_prefix} Loaded font: {font_path} (Size: {font_size})")
            return self.font
        except Exception as e:
            logger.exception(f"{log_prefix} Failed to load font {font_path}: {e}")
            self.font = None
            return None

    # --- 이미지 다운로드 (재시도 포함) ---
    # settings.IMAGE_DOWNLOAD_RETRIES 값 사용
    @tenacity.retry(
        stop=stop_after_attempt(settings.IMAGE_DOWNLOAD_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=tenacity.retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True
    )
    async def _download_image(self, url: str, session: aiohttp.ClientSession, trace_id: Optional[str]) -> Optional[bytes]:
        """URL에서 이미지 비동기 다운로드"""
        if not url: return None
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Downloading image: {url[:80]}...")
        try:
             # 이미지 다운로드 타임아웃 설정 (예: 20초)
             async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as response:
                  response.raise_for_status() # HTTP 오류 시 예외 발생
                  content = await response.read()
                  logger.debug(f"{log_prefix} Downloaded {len(content)} bytes from {url[:80]}")
                  return content
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
             logger.error(f"{log_prefix} Failed to download {url[:80]}: {e}")
             raise # tenacity 재시도 위해 예외 다시 발생
        except Exception as e:
             logger.exception(f"{log_prefix} Unexpected error downloading {url[:80]}: {e}")
             return None # 최종 실패 시 None 반환

    # --- 텍스트 선택 ---
    def _select_panel_text(self, panel_index: int, scenarios: List[Dict], translated_text: Optional[List[Dict]]) -> str:
        """패널에 맞는 텍스트 선택 (원본 또는 번역)"""
        try:
            original_dialogue = scenarios[panel_index].get('dialogue', '')
            scene_num_to_find = panel_index + 1

            if translated_text and isinstance(translated_text, list):
                for item in translated_text:
                    if isinstance(item, dict) and item.get('scene') == scene_num_to_find:
                        # 번역 실패(None)가 아닌 경우 번역본 우선 사용
                        if item.get('translated_dialogue') is not None:
                            return item['translated_dialogue']
                        else: # 번역 실패 시 원본 사용
                            logger.debug(f"Panel {scene_num_to_find}: Translation was None, using original.")
                            return original_dialogue
                # 매칭되는 번역 결과 못 찾음 (오류 상황일 수 있음)
                logger.warning(f"Translated text entry not found for panel {scene_num_to_find}, using original.")
                return original_dialogue
            else:
                # 번역 비활성화 또는 결과 없음
                return original_dialogue
        except IndexError:
             logger.error(f"Error accessing scenario panel index {panel_index}.")
             return ""

    # --- 이미지 합성 (CPU 바운드) ---
    def _compose_comic(self, panel_images: List[Image.Image], panel_texts: List[str], config: Dict, trace_id: Optional[str]) -> Optional[Image.Image]:
        """패널 이미지 합성 및 텍스트 오버레이 (Pillow 사용)"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not PILLOW_AVAILABLE:
            logger.error(f"{log_prefix} Cannot compose comic: Pillow library not available.")
            return None

        # 입력 이미지 개수 확인
        if len(panel_images) != 4:
             logger.error(f"{log_prefix} Expected 4 panel images, got {len(panel_images)}. Cannot compose.")
             return None

        logger.info(f"{log_prefix} Composing 2x2 comic strip...")
        try:
            final_width = config.get("final_image_width", 1024)
            panel_width = final_width // 2
            # 원본 비율 유지를 위해 높이 계산 (첫 번째 이미지 기준)
            img0 = panel_images[0]
            aspect_ratio = img0.height / img0.width
            panel_height = int(panel_width * aspect_ratio)
            final_height = panel_height * 2

            logger.debug(f"{log_prefix} Final size: {final_width}x{final_height}. Panel size: {panel_width}x{panel_height}")

            # 모든 패널 리사이즈
            resized_panels = [img.resize((panel_width, panel_height), Image.Resampling.LANCZOS) for img in panel_images]

            # 최종 캔버스 생성
            final_image = Image.new('RGB', (final_width, final_height), color='white')

            # 패널 붙여넣기 (2x2 그리드)
            final_image.paste(resized_panels[0], (0, 0))
            final_image.paste(resized_panels[1], (panel_width, 0))
            final_image.paste(resized_panels[2], (0, panel_height))
            final_image.paste(resized_panels[3], (panel_width, panel_height))

            # --- 텍스트 오버레이 ---
            font_path = config.get("default_font_path")
            font_size_ratio = config.get("text_overlay_font_size_ratio", 20)
            font_color = config.get("text_overlay_color", "black")
            font_size = max(15, panel_height // int(font_size_ratio)) # 패널 높이 기반 폰트 크기

            font = self._load_font(font_path, font_size, trace_id)
            if font:
                 draw = ImageDraw.Draw(final_image)
                 # 텍스트 위치 (패널 하단 중앙 정렬 예시)
                 positions = [ (0, panel_height), (panel_width, panel_height),
                               (0, final_height), (panel_width, final_height) ]
                 padding = font_size // 2 # 하단 여백

                 for i, text in enumerate(panel_texts):
                      if text:
                           pos_x_start, pos_y_bottom = positions[i]
                           # 텍스트 바운딩 박스 계산
                           try:
                                bbox = draw.textbbox((0, 0), text, font=font)
                                text_width = bbox[2] - bbox[0]
                                text_height = bbox[3] - bbox[1]
                                # 중앙 정렬 X 좌표 계산
                                draw_x = pos_x_start + max(0, (panel_width - text_width) // 2)
                                # 하단 정렬 Y 좌표 계산
                                draw_y = pos_y_bottom - text_height - padding
                                # TODO: 텍스트 래핑, 배경 박스 추가 등 개선 필요
                                draw.text((draw_x, draw_y), text, fill=font_color, font=font)
                           except Exception as text_err:
                                logger.error(f"{log_prefix} Error drawing text for panel {i+1}: {text_err}")
            else:
                 logger.warning(f"{log_prefix} Font not loaded, skipping text overlay.")
            # --- 텍스트 오버레이 종료 ---

            logger.info(f"{log_prefix} Comic composition complete.")
            return final_image

        except Exception as e:
            logger.exception(f"{log_prefix} Error during comic composition: {e}")
            return None

    # --- 이미지 저장/업로드 ---
    async def _save_or_upload_image(self, image: Image.Image, format_str: str, comic_id: str, config: Dict, trace_id: Optional[str]) -> Optional[str]:
        """최종 만화 이미지를 저장 또는 업로드하고 URL/경로 반환"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if image is None: return None

        upload_to_s3 = config.get("upload_to_s3", False)
        save_dir = config.get("final_comic_save_dir")
        quality = config.get("final_image_quality", 85)
        filename = f"{comic_id}.{format_str.lower()}"
        content_type = f"image/{format_str.lower()}"

        logger.info(f"{log_prefix} Processing final comic image (Format: {format_str}, UploadS3: {upload_to_s3})...")

        # BytesIO를 사용하여 이미지 데이터를 메모리에 저장
        img_byte_arr = BytesIO()
        try:
            image.save(img_byte_arr, format=format_str.upper(), quality=quality)
            img_byte_arr.seek(0) # 포인터를 시작으로 이동
            image_bytes = img_byte_arr.getvalue() # 바이트 데이터 가져오기
        except Exception as e:
             logger.exception(f"{log_prefix} Failed to save image to memory buffer: {e}")
             return None

        # S3 업로드 시도 (설정된 경우 및 클라이언트 사용 가능 시)
        if upload_to_s3 and self.storage_client:
            logger.debug(f"{log_prefix} Attempting S3 upload for {filename}...")
            s3_key = f"comics/{filename}" # S3 내 저장 경로
            # StorageClient의 upload_file 메서드는 로컬 파일 경로를 받음
            # 따라서 메모리 버퍼를 직접 업로드하는 기능이 StorageClient에 필요하거나,
            # 임시 파일로 저장 후 업로드해야 함.
            # 여기서는 StorageClient에 바이트 업로드 기능(예: upload_bytes)이 있다고 가정.
            try:
                # StorageClient에 upload_bytes 와 같은 메서드가 필요함
                # upload_result = await self.storage_client.upload_bytes(
                #     file_bytes=image_bytes,
                #     object_key=s3_key,
                #     content_type=content_type
                # )

                # --- 임시 방편: 로컬 저장 후 StorageClient.upload_file 사용 ---
                # 이 방식은 비효율적일 수 있음. StorageClient 개선 권장.
                temp_local_path = os.path.join(save_dir or ".", filename) # 임시 저장 경로
                os.makedirs(os.path.dirname(temp_local_path), exist_ok=True)
                with open(temp_local_path, "wb") as f: f.write(image_bytes)
                logger.debug(f"{log_prefix} Temporarily saved to {temp_local_path} for S3 upload.")

                upload_result = await self.storage_client.upload_file(
                     file_path=temp_local_path,
                     object_key=s3_key,
                     content_type=content_type
                )
                os.remove(temp_local_path) # 임시 파일 삭제
                # -----------------------------------------------------------

                if "error" in upload_result:
                    logger.error(f"{log_prefix} S3 upload failed: {upload_result['error']}")
                    return None # 업로드 실패
                else:
                    # s3_uri 또는 presigned_url 등 StorageClient 반환값에 따라 URL 생성/반환
                    # 여기서는 s3_uri를 반환한다고 가정
                    s3_uri = upload_result.get("s3_uri")
                    if s3_uri:
                         logger.info(f"{log_prefix} Final comic uploaded to S3: {s3_uri}")
                         return s3_uri
                    else:
                         logger.error(f"{log_prefix} S3 upload succeeded but no URI returned.")
                         return None
            except Exception as e:
                logger.exception(f"{log_prefix} Error during S3 upload: {e}")
                return None # 업로드 중 예외 발생
        # S3 업로드 안하거나 실패 시 로컬 저장 시도
        elif save_dir:
            logger.debug(f"{log_prefix} Attempting local save to {save_dir}...")
            try:
                os.makedirs(save_dir, exist_ok=True)
                filepath = os.path.join(save_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(image_bytes) # 메모리 버퍼에서 직접 쓰기
                logger.info(f"{log_prefix} Final comic saved locally to: {filepath}")
                return filepath # 로컬 파일 경로 반환
            except Exception as e:
                logger.exception(f"{log_prefix} Failed to save final comic locally: {e}")
                return None
        else:
            logger.error(f"{log_prefix} Cannot save final comic: S3 upload disabled/failed and local save directory not configured.")
            return None

    # --- ALT 텍스트 생성 ---
    def _generate_alt_text(self, scenarios: List[Dict], chosen_idea: Optional[Dict], max_len: int, trace_id: Optional[str]) -> str:
        """최종 만화 이미지용 ALT 텍스트 생성"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Generating ALT text...")
        try:
            # TODO: 필요시 LLM을 사용하여 더 자연스러운 ALT 텍스트 생성
            alt_parts = []
            title = chosen_idea.get('idea_title', 'Comic') if chosen_idea else 'Comic'
            alt_parts.append(f"A 4-panel comic titled '{title}'.")

            if scenarios and len(scenarios) == 4:
                for i, panel in enumerate(scenarios):
                    desc = panel.get('panel_description', '')
                    dialogue = panel.get('dialogue', '')
                    panel_alt = f"Panel {i+1}: {desc}" if desc else f"Panel {i+1}"
                    if dialogue: panel_alt += f" Dialogue says '{dialogue}'"
                    alt_parts.append(panel_alt + ".")

            alt_text = " ".join(alt_parts)
            final_alt = alt_text[:max_len] + ("..." if len(alt_text) > max_len else "")
            logger.info(f"{log_prefix} Generated ALT text: '{final_alt[:100]}...'")
            return final_alt
        except Exception as e:
             logger.exception(f"{log_prefix} Failed to generate ALT text: {e}")
             return "A 4-panel comic." # 기본 ALT 텍스트

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """후처리 프로세스 실행: 다운로드, 합성, 저장/업로드, ALT 텍스트"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing PostprocessorNode...")

        # 상태 및 설정 로드
        image_urls = state.image_urls or []
        scenarios = state.scenarios or []
        translated_text = state.translated_text # Optional
        chosen_idea = state.chosen_idea
        comic_id = state.comic_id or "unknown_comic"
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        final_comic_output = {"png_url": None, "webp_url": None, "alt_text": None}
        error_message: Optional[str] = None

        # 기본 요구사항 확인
        if not PILLOW_AVAILABLE: error_message = "Pillow library not available."
        elif len(image_urls) != 4: error_message = f"Requires 4 image URLs, found {len(image_urls)}."
        elif len(scenarios) != 4: error_message = "Requires 4 scenario panels."

        if error_message:
            logger.error(f"{log_prefix} {error_message} Cannot proceed with postprocessing.")
        else:
            try:
                # --- 1. 이미지 다운로드 ---
                logger.info(f"{log_prefix} Downloading 4 panel images...")
                http_timeout = config.get('http_timeout', 15) # 다운로드 타임아웃
                panel_images_pil: List[Optional[Image.Image]] = [None] * 4
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=http_timeout * 2)) as session:
                    download_tasks = [self._download_image(url, session, state.trace_id) for url in image_urls]
                    image_bytes_list = await asyncio.gather(*download_tasks, return_exceptions=True)

                loaded_count = 0
                for i, img_bytes in enumerate(image_bytes_list):
                    if isinstance(img_bytes, bytes):
                        try:
                            panel_images_pil[i] = Image.open(BytesIO(img_bytes))
                            loaded_count += 1
                        except Exception as img_err: logger.error(f"{log_prefix} Failed to load image bytes for panel {i+1}: {img_err}")
                    elif isinstance(img_bytes, Exception): logger.error(f"{log_prefix} Failed to download image for panel {i+1}: {img_bytes}")

                if loaded_count != 4:
                    error_message = f"Failed to download or load all 4 panel images (loaded {loaded_count})."
                    logger.error(f"{log_prefix} {error_message}")
                else:
                    logger.info(f"{log_prefix} Successfully downloaded and loaded 4 panel images.")

                    # --- 2. 텍스트 선택 ---
                    panel_texts = [self._select_panel_text(i, scenarios, translated_text) for i in range(4)]

                    # --- 3. 이미지 합성 (CPU 바운드 -> Executor) ---
                    logger.info(f"{log_prefix} Starting image composition...")
                    loop = asyncio.get_running_loop()
                    # Pass PIL images list
                    final_image_pil = await loop.run_in_executor(
                        None, self._compose_comic, panel_images_pil, panel_texts, config, state.trace_id
                    )

                    # --- 4. 저장 / 업로드 ---
                    if final_image_pil:
                        output_format = config.get("final_image_format", "WEBP").upper()
                        output_url = await self._save_or_upload_image(final_image_pil, output_format, comic_id, config, state.trace_id)
                        if output_url:
                             if output_format == "PNG": final_comic_output['png_url'] = output_url
                             elif output_format == "WEBP": final_comic_output['webp_url'] = output_url
                             else: final_comic_output['webp_url'] = output_url # 기본 WEBP
                        else: error_message = "Failed to save or upload the final comic image."
                    else:
                        error_message = "Comic composition failed."

                    # --- 5. ALT 텍스트 생성 ---
                    if not error_message: # 최종 이미지 생성 성공 시에만 생성
                         max_alt_len = config.get("max_alt_text_len", 300)
                         final_comic_output['alt_text'] = self._generate_alt_text(scenarios, chosen_idea, max_alt_len, state.trace_id)

            except Exception as e:
                 error_message = f"Error during postprocessing: {str(e)}"
                 logger.exception(f"{log_prefix} {error_message}")


        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['postprocessor_node_time'] = node_processing_time
        logger.info(f"{log_prefix} PostprocessorNode finished in {node_processing_time:.2f} seconds.")

        # TODO: LangSmith 로깅

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "final_comic": final_comic_output, # 결과 URL 및 ALT 텍스트 포함
            "processing_stats": processing_stats,
            "error_message": error_message # 최종 오류 메시지
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}