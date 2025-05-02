# app/nodes/17_imager_node.py

import asyncio
import re
import json
import random # Seed 생성을 위해 유지
# --- datetime, timezone 임포트 추가 ---
from datetime import datetime, timezone
# ------------------------------------
from typing import List, Dict, Any, Optional
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed

# --- 프로젝트 구성 요소 임포트 ---
from app.config.settings import settings # 재시도 횟수 등 참조
# 실제 이미지 생성 클라이언트 임포트 (경로 및 클래스명 확인 필요)
from app.services.image_server_client_v2 import ImageGenerationClient
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# --- 로거 설정 ---
logger = get_logger("ImagerNode")

class ImagerNode:
    """
    (Refactored) 시나리오 각 패널에 대한 이미지를 생성합니다.
    - ImageGenerationClient를 사용하여 이미지 생성 API 호출.
    - ControlNet 전략 적용 (플레이스홀더, 필요시 구현).
    - 패널별 재시도 로직 포함.
    - 설정은 상태의 config 딕셔너리에서 로드.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["scenarios", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["image_urls", "processing_stats", "error_message"]

    # ImageGenerationClient 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        image_client: ImageGenerationClient,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        self.image_client = image_client
        # self.langsmith_service = langsmith_service
        logger.info("ImagerNode initialized with ImageGenerationClient.")

    # --- 이미지 프롬프트 구성 ---
    def _construct_image_prompt(self, panel_data: Dict[str, Any], default_style: str, max_len: int) -> str:
        """이미지 생성 API를 위한 프롬프트 구성"""
        description = panel_data.get('panel_description', '')
        tags = panel_data.get('seed_tags', [])

        prompt_parts = [description]
        if tags and isinstance(tags, list):
            # 태그 순서 섞기 (다양성)
            shuffled_tags = random.sample(tags, len(tags))
            prompt_parts.append(", ".join(tag for tag in shuffled_tags if tag)) # 빈 태그 제외
        if default_style:
            prompt_parts.append(default_style)

        full_prompt = ", ".join(part for part in prompt_parts if part and part.strip())
        full_prompt = re.sub(r'\s+', ' ', full_prompt).strip()

        # 최대 길이 제한
        return full_prompt[:max_len]

    # --- ControlNet 전략 (구현 필요 시) ---
    def _get_controlnet_input(self, panel_index: int, previous_image_url: Optional[str], panel_data: Dict[str, Any], config: Dict, trace_id: Optional[str]) -> Optional[Dict]:
        """ControlNet 입력 결정 (플레이스홀더)"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        # enable_controlnet = config.get("enable_controlnet", False)
        # if not enable_controlnet: return None

        # TODO: 실제 ControlNet 전략 구현 (예: 이전 패널 사용, 포즈 생성 등)
        # control_type = config.get("controlnet_type")
        # control_weight = config.get("controlnet_weight")
        # if control_type == "previous_panel" and panel_index > 0 and previous_image_url:
        #     logger.debug(f"{log_prefix} Using previous panel image for ControlNet ({control_type}).")
        #     return {"control_image_url": previous_image_url, "type": control_type, "weight": control_weight}
        # elif control_type == "openpose":
        #     # 포즈 생성 로직 필요
        #     logger.debug(f"{log_prefix} Using generated pose for ControlNet ({control_type}).")
        #     # return {"pose_data": ..., "type": control_type, "weight": control_weight}
        # else:
        #     logger.debug(f"{log_prefix} No applicable ControlNet strategy found for panel {panel_index + 1}.")
        return None # 현재는 ControlNet 사용 안 함

    # --- 패널 이미지 생성 (재시도 포함) ---
    # settings의 IMAGE_API_RETRIES 값으로 재시도 횟수 설정
    @tenacity.retry(
        stop=stop_after_attempt(settings.IMAGE_API_RETRIES),
        wait=wait_fixed(2) + wait_exponential(multiplier=1, max=10), # 고정+지수 대기
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying image generation for panel index {retry_state.args[1]} (Attempt {retry_state.attempt_number})... Last exception: {retry_state.outcome.exception()}"
        ),
        reraise=True
    )
    async def _generate_panel_image(
        self, panel_data: Dict[str, Any], panel_index: int, previous_image_url: Optional[str], config: Dict, trace_id: Optional[str]
        ) -> Optional[str]:
        """단일 패널 이미지 생성 (실제 클라이언트 사용)"""
        log_prefix = f"[{trace_id}][Panel {panel_index + 1}]"
        logger.info(f"{log_prefix} Starting image generation...")

        # 설정 로드
        image_model = config.get("image_model", "default_model")
        img_height = config.get("image_height", 1024)
        img_width = config.get("image_width", 1024)
        negative_prompt = config.get("image_negative_prompt", "")
        style_preset = config.get("image_style_preset") # API가 지원하는 경우
        default_style = config.get("image_default_style", "")
        max_prompt_len = config.get("max_image_prompt_len", 500)

        # 1. 프롬프트 구성
        prompt = self._construct_image_prompt(panel_data, default_style, max_prompt_len)
        logger.debug(f"{log_prefix} Prompt: {prompt}")

        # 2. ControlNet 입력 결정 (필요시 구현)
        controlnet_params = self._get_controlnet_input(panel_index, previous_image_url, panel_data, config, trace_id)

        # 3. 이미지 생성 API 호출 (실제 클라이언트 사용)
        try:
            # generate_image 가 async 라고 가정 (클라이언트 구현 확인 필요)
            # 필요한 모든 파라미터를 kwargs 또는 명시적 인자로 전달
            result = await self.image_client.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                # API 클라이언트가 모델, 크기 등을 어떻게 받는지 확인 필요
                # 아래는 예시이며, 실제 클라이언트 인터페이스에 맞춰야 함
                model=image_model, # 클라이언트가 지원하는 경우
                height=img_height, # 클라이언트가 지원하는 경우
                width=img_width,   # 클라이언트가 지원하는 경우
                style_preset=style_preset, # 클라이언트가 지원하는 경우
                controlnet_params=controlnet_params, # 클라이언트가 지원하는 경우
                seed=random.randint(0, 2**32 - 1) # 시드값 생성
                # 기타 필요한 파라미터 전달 ...
            )

            # 클라이언트 반환값 처리 (URL 또는 경로 추출)
            # ImageGenerationClient v2는 {"image_path": ..., "image_filename": ...} 또는 {"image_url": ...} 반환 가정
            if "error" in result:
                 error_msg = result['error']
                 logger.error(f"{log_prefix} Image generation API returned error: {error_msg}")
                 raise RuntimeError(f"Image generation failed: {error_msg}") # 재시도 유발
            elif "image_url" in result:
                 image_url = result["image_url"]
                 logger.info(f"{log_prefix} Image generated successfully (URL): {image_url}")
                 return image_url
            elif "image_path" in result:
                 # 로컬 경로 반환 시, 외부 접근 가능한 URL로 변환 필요할 수 있음
                 # 여기서는 일단 로컬 경로/파일명 자체를 반환
                 image_path = result["image_path"]
                 logger.info(f"{log_prefix} Image generated successfully (Path): {image_path}")
                 return image_path # 또는 파일명: result["image_filename"]
            else:
                 logger.error(f"{log_prefix} Image generation API returned unexpected result: {result}")
                 raise ValueError("Image generation returned no valid URL or path") # 재시도 유발

        except Exception as e:
            # tenacity 재시도를 위해 예외 다시 발생
            logger.error(f"{log_prefix} Image generation attempt failed: {e.__class__.__name__}: {e}")
            raise

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """모든 시나리오 패널에 대한 이미지 생성 실행"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing ImagerNode...")

        # 상태 및 설정 로드
        scenarios = state.scenarios or []
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        # 입력 유효성 검사
        if not scenarios or not isinstance(scenarios, list) or len(scenarios) != 4:
            logger.error(f"{log_prefix} Invalid or missing 4-panel scenario data.")
            processing_stats['imager_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "image_urls": [],
                "processing_stats": processing_stats,
                "error_message": "Valid 4-panel scenario is required for image generation."
            }

        logger.info(f"{log_prefix} Starting image generation for {len(scenarios)} panels...")

        generated_image_urls: List[Optional[str]] = [None] * len(scenarios)
        previous_image_url: Optional[str] = None # 이전 패널 이미지 URL (ControlNet용)
        task_errors: List[str] = [] # 개별 패널 오류 기록

        # ControlNet 등 순차 의존성 고려하여 순차 실행 (필요시 비동기 병렬 처리로 변경 가능)
        for i, panel_data in enumerate(scenarios):
            try:
                # 개별 패널 이미지 생성 (재시도 로직 포함)
                image_url_or_path = await self._generate_panel_image(
                    panel_data, i, previous_image_url, config, state.trace_id
                )
                generated_image_urls[i] = image_url_or_path
                # 다음 패널의 ControlNet 입력을 위해 현재 결과 저장 (URL 또는 경로)
                previous_image_url = image_url_or_path
            except Exception as e:
                # 최종 재시도 실패 시
                msg = f"Panel {i+1} generation failed after retries: {e.__class__.__name__}"
                logger.error(f"{log_prefix} {msg}")
                task_errors.append(msg)
                generated_image_urls[i] = None # 실패 표시
                # 실패 시 previous_image_url을 어떻게 할지 결정 (None으로 리셋?)
                # previous_image_url = None

        # 성공한 URL/경로만 필터링
        final_image_outputs = [url for url in generated_image_urls if url is not None]
        failed_panels = len(scenarios) - len(final_image_outputs)

        logger.info(f"{log_prefix} Image generation complete. Success: {len(final_image_outputs)}, Failed: {failed_panels}.")

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"{log_prefix} Errors occurred during image generation: {final_error_message}")

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['imager_node_time'] = node_processing_time
        logger.info(f"{log_prefix} ImagerNode finished in {node_processing_time:.2f} seconds.")

        # TODO: LangSmith 로깅

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "image_urls": final_image_outputs, # 성공한 이미지 URL/경로 리스트
            "processing_stats": processing_stats,
            "error_message": final_error_message # 부분 오류 요약
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}