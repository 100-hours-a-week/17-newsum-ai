# ai/app/api/v2/image_background_tasks.py
import os
import asyncio
import time
from typing import List, Dict, Any
from app.utils.logger import get_logger
from app.services.postgresql_service import PostgreSQLService
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
    """단일 프롬프트에 대한 이미지 생성, S3 업로드, 로컬 파일 삭제를 수행하는 보조 함수 (재시도 포함)"""
    prompt = item_payload.get("prompt", "")
    log_prompt = prompt[:50] + '...' if prompt else "N/A"
    logger.info(f"이미지 생성 시작: request_id='{request_id}', index={image_index}, prompt='{log_prompt}'")

    max_retries = 2
    for attempt in range(max_retries + 1):
        gen_result = await image_service.generate_image(
            **item_payload,
            request_id=request_id,
            image_index=image_index
        )
        if not gen_result.get("error") and gen_result.get("image_path"):
            break
        else:
            error_msg = gen_result.get("error", "Generated image path not found.")
            logger.error(f"이미지 생성 실패 '{log_prompt}' (시도 {attempt+1}/{max_retries+1}): {error_msg}")
            if attempt == max_retries:
                raise ValueError(f"Generation failed after {max_retries+1} attempts: {error_msg}")
            await asyncio.sleep(1)  # 재시도 전 1초 대기

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
        pg_service: PostgreSQLService,
        image_service: ImageService,
        storage_service: StorageService,
        backend_client: BackendApiClient,
):
    """
    [수정] 배치 이미지 생성 요청을 순차적으로 처리하고,
    완료 후 백엔드 서버에 결과를 콜백으로 전송합니다.
    실패 시 ai_image_job_tracking에서 job_id를 삭제합니다.
    """
    extra_log = {"request_id": payload.id}
    logger.info(f"배치 이미지 생성 백그라운드 작업 시작 (순차 처리 모드).", extra=extra_log)

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
                logger.info(
                    f"[{payload.id}] 이미지 생성 및 업로드 성공 (index: {i+1}/{num_prompts})",
                    extra={**extra_log, "image_index": i, "success_count": len(success_uploads)}
                )
            except Exception as e:
                has_errors = True
                logger.error(
                    f"[{payload.id}] 배경 작업 중 개별 작업(index: {i}) 실패: {e}",
                    extra={**extra_log, "image_index": i, "error": str(e)},
                    exc_info=True
                )
            if i < num_prompts - 1:
                logger.info(
                    f"[{payload.id}] 다음 이미지 생성을 위해 2초 대기 (현재 {i + 1}/{num_prompts} 완료)",
                    extra=extra_log
                )
                await asyncio.sleep(2)
        final_status = "COMPLETED"
        if has_errors:
            final_status = "COMPLETED_WITH_ERRORS" if success_uploads else "FAILED"

        if len(success_uploads) != num_prompts:
            error_msg = f"[{payload.id}] 요청한 프롬프트 개수({num_prompts})와 생성된 이미지 개수({len(success_uploads)})가 일치하지 않습니다. 콜백을 전송하지 않습니다. DB에서 job_id 제거."
            logger.error(error_msg, extra={**extra_log, "success_count": len(success_uploads), "expected_count": num_prompts})
            await pg_service.delete_image_job_by_id(payload.id)
            raise RuntimeError(error_msg)

        try:
            image_links = [result.s3_uri for result in success_uploads]
            logger.info(
                f"[{payload.id}] 백그라운드 작업 결과 콜백 전송 시도. 전송할 링크 수: {len(image_links)}",
                extra={**extra_log, "image_links": image_links, "success_count": len(success_uploads), "expected_count": num_prompts}
            )
            await backend_client.backend_send_ai_response(
                request_id=payload.id,
                image_links=image_links
            )
            logger.info(
                f"[{payload.id}] 콜백 전송 성공. (전송된 링크 수: {len(image_links)})",
                extra={**extra_log, "image_links": image_links, "success_count": len(success_uploads), "expected_count": num_prompts}
            )
        except Exception as e:
            logger.error(
                f"[{payload.id}] 콜백 전송 중 예외 발생: {e}. DB에서 job_id 제거.",
                exc_info=True,
                extra={**extra_log, "image_links": image_links if 'image_links' in locals() else [], "error": str(e)}
            )
            await pg_service.delete_image_job_by_id(payload.id)
    finally:
        image_service.is_running = False  # 배치 끝나면 False