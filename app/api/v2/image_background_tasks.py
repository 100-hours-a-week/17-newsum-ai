# ai/app/api/v2/image_background_tasks.py
import os
import asyncio
import time
from typing import List, Dict, Any
from app.utils.logger import get_logger
from app.services.image_service import ImageService
from app.services.storage_service import StorageService
from app.services.backend_client import BackendApiClient
from .schemas import BatchImageGenerationRequest, ImageUploadResult, BatchImageGenerationResponse

logger = get_logger(__name__)


# generate_and_upload_single_image 함수는 변경할 필요가 없습니다.
async def generate_and_upload_single_image(
        item_payload: Dict[str, Any],
        image_service: ImageService,
        storage_service: StorageService,
        request_id: str,
        image_index: int
) -> Dict[str, Any]:
    """단일 프롬프트에 대한 이미지 생성, S3 업로드, 로컬 파일 삭제를 수행하는 보조 함수"""
    prompt = item_payload.get("prompt", "")
    log_prompt = prompt[:50] + '...' if prompt else "N/A"
    logger.info(f"이미지 생성 시작: request_id='{request_id}', index={image_index}, prompt='{log_prompt}'")

    gen_result = await image_service.generate_image(
        **item_payload,
        request_id=request_id,
        image_index=image_index
    )
    if gen_result.get("error") or not gen_result.get("image_path"):
        error_msg = gen_result.get("error", "Generated image path not found.")
        logger.error(f"이미지 생성 실패 '{log_prompt}': {error_msg}")
        raise ValueError(f"Generation failed: {error_msg}")

    local_path = gen_result["image_path"]
    try:
        logger.info(f"S3 업로드 시작: '{local_path}'")
        # object_key를 local_path에서 storage_base_path 기준 상대경로로 추출
        object_key = os.path.relpath(local_path, image_service.storage_base_path)
        upload_result = await storage_service.upload_file_with_cloudfront_url(
            file_path=local_path,
            object_key=object_key,
            content_type="image/png"
        )
        if not upload_result.get("cloudfront_url"):
            error_msg = upload_result.get("error", "CloudFront URL 반환 실패")
            logger.error(f"S3 업로드 실패 '{local_path}': {error_msg}")
            raise IOError(f"S3 upload failed: {error_msg}")

        return {
            "s3_uri": upload_result["cloudfront_url"],
            "object_key": object_key,
            "original_prompt": prompt,
        }
    finally:
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
                logger.info(f"로컬 파일 정리 성공: '{local_path}'")
            except OSError as e:
                logger.error(f"로컬 파일 정리 실패 '{local_path}': {e}")


async def generate_images_in_background(
        payload: BatchImageGenerationRequest,
        image_service: ImageService,
        storage_service: StorageService,
        backend_client: BackendApiClient,
):
    """
    [수정] 배치 이미지 생성 요청을 순차적으로 처리하고,
    완료 후 백엔드 서버에 결과를 콜백으로 전송합니다.
    """
    extra_log = {"request_id": payload.id}
    logger.info("배치 이미지 생성 백그라운드 작업 시작 (순차 처리 모드).", extra=extra_log)

    image_service.is_running = True  # 배치 시작 시 True
    try:
        success_uploads: List[ImageUploadResult] = []
        has_errors = False
        num_prompts = len(payload.image_prompts)
        for i, item in enumerate(payload.image_prompts):
            try:
                single_result_dict = await generate_and_upload_single_image(
                    item_payload=item.model_dump(),
                    image_service=image_service,
                    storage_service=storage_service,
                    request_id=payload.id,
                    image_index=i
                )
                success_uploads.append(ImageUploadResult(**single_result_dict))
            except Exception as e:
                has_errors = True
                logger.error(f"배경 작업 중 개별 작업(index: {i}) 실패: {e}", extra=extra_log, exc_info=True)
            if i < num_prompts - 1:
                logger.info(f"다음 이미지 생성을 위해 2초 대기합니다. (현재 {i + 1}/{num_prompts} 완료)")
                await asyncio.sleep(2)
        final_status = "COMPLETED"
        if has_errors:
            final_status = "COMPLETED_WITH_ERRORS" if success_uploads else "FAILED"

        if len(success_uploads) != num_prompts:
            error_msg = f"요청한 프롬프트 개수({num_prompts})와 생성된 이미지 개수({len(success_uploads)})가 일치하지 않습니다. 콜백을 전송하지 않습니다."
            logger.error(error_msg, extra=extra_log)
            raise RuntimeError(error_msg)

        try:
            image_links = [result.s3_uri for result in success_uploads]
            logger.info(f"백그라운드 작업 결과 콜백 전송 시도. 전송할 링크 수: {len(image_links)}", extra=extra_log)
            await backend_client.backend_send_ai_response(
                request_id=payload.id,
                image_links=image_links
            )
            logger.info("콜백 전송 성공.", extra=extra_log)
        except Exception as e:
            logger.error(f"콜백 전송 중 예외 발생: {e}", exc_info=True, extra=extra_log)
    finally:
        image_service.is_running = False  # 배치 끝나면 False