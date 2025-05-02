# app/nodes/17_imager_node.py (Improved Version)

import asyncio
import re
import json
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 기본값 참조용
from app.services.image_server_client_v2 import ImageGenerationClient
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# 로거 설정
logger = get_logger(__name__)

class ImagerNode:
    """
    시나리오 각 패널에 대한 이미지를 생성합니다.
    - ImageGenerationClient를 사용하여 이미지 생성 API 호출 (재시도 포함).
    - ControlNet 전략 적용 (Placeholder).
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["scenarios", "trace_id", "comic_id", "config", "processing_stats"]
    outputs: List[str] = ["image_urls", "processing_stats", "error_message"]

    # ImageGenerationClient 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        image_client: ImageGenerationClient,
        # langsmith_service: Optional[LangSmithService] = None
    ):
        if not image_client: raise ValueError("ImageGenerationClient is required.")
        self.image_client = image_client
        # self.langsmith = langsmith_service
        logger.info("ImagerNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.image_model = config.get("image_model", settings.DEFAULT_IMAGE_MODEL)
        self.img_height = int(config.get("image_height", settings.DEFAULT_IMAGE_HEIGHT))
        self.img_width = int(config.get("image_width", settings.DEFAULT_IMAGE_WIDTH))
        self.negative_prompt = config.get("image_negative_prompt", settings.DEFAULT_IMAGE_NEGATIVE_PROMPT)
        self.style_preset = config.get("image_style_preset", settings.DEFAULT_IMAGE_STYLE_PRESET) # API가 지원하는 경우
        self.default_style = config.get("image_default_style", settings.DEFAULT_IMAGE_STYLE)
        self.max_prompt_len = int(config.get("max_image_prompt_len", settings.DEFAULT_MAX_IMAGE_PROMPT_LEN))
        # ControlNet 관련 설정 (Placeholder)
        self.enable_controlnet = config.get("enable_controlnet", settings.ENABLE_CONTROLNET)
        self.controlnet_type = config.get("controlnet_type", settings.DEFAULT_CONTROLNET_TYPE)
        self.controlnet_weight = float(config.get("controlnet_weight", settings.DEFAULT_CONTROLNET_WEIGHT))

        logger.debug(f"Runtime config loaded. Model: {self.image_model}, Size: {self.img_width}x{self.img_height}")
        logger.debug(f"Style Preset: {self.style_preset}, Default Style: {self.default_style}")
        logger.debug(f"ControlNet Enabled: {self.enable_controlnet}, Type: {self.controlnet_type}, Weight: {self.controlnet_weight}")

    # --- 이미지 프롬프트 구성 ---
    def _construct_image_prompt(self, panel_data: Dict[str, Any]) -> str:
        """이미지 생성 API를 위한 프롬프트 구성"""
        description = panel_data.get('panel_description', '').strip()
        # Node 15에서 생성된 seed_tags 사용
        tags = panel_data.get('seed_tags', [])

        prompt_parts = [description]
        if tags and isinstance(tags, list):
            # 태그 순서 무작위화 시도 (결과 다양성)
            shuffled_tags = random.sample(tags, len(tags))
            prompt_parts.append(", ".join(tag for tag in shuffled_tags if tag))
        if self.default_style:
            prompt_parts.append(self.default_style)

        # 공백 정리 및 최대 길이 제한
        full_prompt = re.sub(r'\s+', ' ', ", ".join(part for part in prompt_parts if part)).strip()
        return full_prompt[:self.max_prompt_len]

    # --- ControlNet 전략 (Placeholder) ---
    # TODO: 실제 ControlNet 사용 시 이 부분 구현 필요
    def _get_controlnet_input(self, panel_index: int, previous_image_url: Optional[str], panel_data: Dict[str, Any], trace_id: Optional[str]) -> Optional[Dict]:
        """ControlNet 입력 결정 (현재는 Placeholder)"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not self.enable_controlnet: return None

        # 예시: 이전 패널 이미지 사용 (첫 패널 제외)
        if self.controlnet_type == "previous_panel" and panel_index > 0 and previous_image_url:
            logger.debug(f"{log_prefix} Using previous panel image for ControlNet.")
            # ImageGenerationClient가 이해하는 형식으로 반환 필요
            return {"control_type": "reference", "image_url": previous_image_url, "weight": self.controlnet_weight}
        # 예시: OpenPose 등 다른 타입 구현
        # elif self.controlnet_type == "openpose":
        #     pose_data = generate_pose(panel_data) # 포즈 생성 로직
        #     logger.debug(f"{log_prefix} Using generated pose for ControlNet.")
        #     return {"control_type": "openpose", "pose_data": pose_data, "weight": self.controlnet_weight}
        else:
            # logger.debug(f"{log_prefix} No applicable ControlNet strategy found for panel {panel_index + 1}.")
            return None

    # --- 패널 이미지 생성 (재시도 포함) ---
    @tenacity.retry(
        stop=stop_after_attempt(settings.IMAGE_API_RETRIES),
        wait=wait_fixed(2) + wait_exponential(multiplier=1.5, max=15), # 대기 시간 약간 늘림
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying image gen for panel idx {retry_state.args[1]} (Attempt {retry_state.attempt_number}). Last exception: {retry_state.outcome.exception()}"
        ),
        reraise=True
    )
    async def _generate_panel_image(
        self, panel_data: Dict[str, Any], panel_index: int, previous_image_url: Optional[str], trace_id: Optional[str]
        ) -> Optional[str]:
        """단일 패널 이미지 생성 (실제 클라이언트 사용)"""
        log_prefix = f"[{trace_id}][Panel {panel_index + 1}]"
        logger.info(f"{log_prefix} Starting image generation...")

        # 1. 프롬프트 구성
        prompt = self._construct_image_prompt(panel_data)
        logger.debug(f"{log_prefix} Prompt: {prompt}")
        if not prompt:
             logger.warning(f"{log_prefix} Prompt is empty. Skipping image generation.")
             return None # 프롬프트 없으면 생성 불가

        # 2. ControlNet 입력 결정
        controlnet_params = self._get_controlnet_input(panel_index, previous_image_url, panel_data, trace_id)

        # 3. 이미지 생성 API 호출 (클라이언트 사용)
        try:
            # ImageGenerationClient.generate_image 는 비동기로 가정
            result = await self.image_client.generate_image(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                model=self.image_model,
                height=self.img_height,
                width=self.img_width,
                style_preset=self.style_preset,
                controlnet_params=controlnet_params,
                seed=random.randint(0, 2**32 - 1) # 각 패널에 랜덤 시드 부여
                # 기타 필요한 파라미터 (예: steps, cfg_scale 등) 전달
            )

            # 클라이언트 반환값 처리 (URL 또는 경로)
            if "error" in result:
                 error_msg = result['error']
                 logger.error(f"{log_prefix} Image generation API error: {error_msg}")
                 raise RuntimeError(f"Image generation failed: {error_msg}") # 재시도 유발
            elif "image_url" in result and result["image_url"]:
                 image_url = result["image_url"]
                 logger.info(f"{log_prefix} Image generated successfully (URL): {image_url[:80]}...")
                 return image_url
            elif "image_path" in result and result["image_path"]:
                 # 로컬 경로 반환 시, 후처리 노드에서 접근 가능해야 함
                 image_path = result["image_path"]
                 logger.info(f"{log_prefix} Image generated successfully (Path): {image_path}")
                 return image_path
            else:
                 logger.error(f"{log_prefix} Image generation API returned unexpected result: {result}")
                 raise ValueError("Image generation returned no valid URL or path") # 재시도 유발

        except RetryError as e: # 모든 재시도 실패 시
             logger.error(f"{log_prefix} Image generation failed after multiple retries: {e}")
             raise # run 메서드에서 처리하도록 예외 다시 발생
        except Exception as e:
            logger.error(f"{log_prefix} Image generation attempt failed: {e.__class__.__name__}: {e}", exc_info=True)
            raise # 재시도 또는 run 메서드 처리 위해 예외 다시 발생

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """모든 시나리오 패널에 대한 이미지 생성 실행"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        comic_id = state.comic_id # 로깅 등에 사용 가능
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing ImagerNode...")

        scenarios = state.scenarios or []
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        if not scenarios or not isinstance(scenarios, list) or len(scenarios) != 4:
            logger.error(f"{log_prefix} Invalid or missing 4-panel scenario data.")
            processing_stats['imager_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "image_urls": [], "processing_stats": processing_stats,
                "error_message": "Valid 4-panel scenario is required."
            }

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        # --------------------------

        logger.info(f"{log_prefix} Starting image generation for {len(scenarios)} panels...")

        generated_image_outputs: List[Optional[str]] = [None] * len(scenarios)
        previous_image_output: Optional[str] = None # 이전 패널 결과 (URL 또는 경로)
        task_errors: List[str] = []

        # 패널 순서대로 생성 (ControlNet 의존성 고려)
        for i, panel_data in enumerate(scenarios):
            try:
                image_output = await self._generate_panel_image(
                    panel_data, i, previous_image_output, trace_id
                )
                generated_image_outputs[i] = image_output
                previous_image_output = image_output # 다음 패널 위해 업데이트
            except Exception as e: # _generate_panel_image 에서 최종 실패 시
                msg = f"Panel {i+1} generation ultimately failed: {e.__class__.__name__}"
                logger.error(f"{log_prefix} {msg}")
                task_errors.append(msg)
                generated_image_outputs[i] = None
                # 실패 시 이전 결과 사용 여부 결정 (현재는 사용 안함)
                # previous_image_output = None

        # 유효한 결과만 필터링
        final_image_outputs = [url for url in generated_image_outputs if url is not None]
        failed_panels = len(scenarios) - len(final_image_outputs)
        logger.info(f"{log_prefix} Image generation complete. Success: {len(final_image_outputs)}, Failed: {failed_panels}.")

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"{log_prefix} Errors during image generation: {final_error_message}")

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['imager_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} ImagerNode finished in {processing_stats['imager_node_time']:.2f} seconds.")

        # ComicState 업데이트
        update_data: Dict[str, Any] = {
            "image_urls": final_image_outputs, # 성공한 이미지 URL/경로 리스트
            "processing_stats": processing_stats,
            "error_message": final_error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}