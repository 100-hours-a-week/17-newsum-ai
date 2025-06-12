# ai/app/api/v2/image_background_tasks.py
import os
import asyncio
from typing import List, Dict, Any
from app.utils.logger import get_logger
from app.services.image_service import ImageService
from app.services.storage_service import StorageService
from app.services.backend_client import BackendApiClient
from .schemas import BatchImageGenerationRequest, ImageUploadResult

logger = get_logger(__name__)

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

    # 1. 이미지 생성
    # item_payload는 ImagePromptItem.model_dump()의 결과로, generate_image가 필요로 하는 모든 키를 포함합니다.
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
        # 2. S3에 파일 업로드
        logger.info(f"S3 업로드 시작: '{local_path}'")
        upload_result = await storage_service.upload_file(file_path=local_path, prefix="generated_images/")
        if upload_result.get("error"):
            error_msg = upload_result.get("error")
            logger.error(f"S3 업로드 실패 '{local_path}': {error_msg}")
            raise IOError(f"S3 upload failed: {error_msg}")

        return {
            "s3_uri": upload_result["s3_uri"],
            "object_key": upload_result["object_key"],
            "original_prompt": prompt,
        }
    finally:
        # 3. 로컬 임시 파일 정리
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
    배치 이미지 생성 요청을 비동기적으로 처리하고,
    완료 후 백엔드 서버에 결과를 콜백으로 전송합니다.
    """
    extra_log = {"request_id": payload.request_id}
    logger.info("배치 이미지 생성 백그라운드 작업 시작.", extra=extra_log)

    # [수정됨] 스키마 변경에 따라 payload.prompts -> payload.image_prompts 로 변경
    tasks = [
        generate_and_upload_single_image(
            item_payload=item.model_dump(),
            image_service=image_service,
            storage_service=storage_service,
            request_id=payload.request_id,
            image_index=i
        )
        for i, item in enumerate(payload.image_prompts)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_uploads: List[ImageUploadResult] = []
    has_errors = False
    for res in results:
        if isinstance(res, BaseException):
            has_errors = True
            logger.error(f"배경 작업 중 개별 작업 실패: {res}", extra=extra_log)
        else:
            success_uploads.append(ImageUploadResult(**res))

    # 최종 결과 구성 (이 부분은 수정 없음)
    final_status = "COMPLETED"
    if has_errors:
        final_status = "COMPLETED_WITH_ERRORS" if success_uploads else "FAILED"

    # [수정됨] 백엔드 서버로 결과 콜백 전송 로직 변경
    try:
        # 1. 성공한 결과에서 이미지 링크(S3 URI) 목록을 추출합니다.
        image_links = [result.s3_uri for result in success_uploads]

        logger.info(f"백그라운드 작업 결과 콜백 전송 시도. 전송할 링크 수: {len(image_links)}", extra=extra_log)

        # 2. backend_send_ai_response 함수 시그니처에 맞게 인자를 전달합니다.
        await backend_client.backend_send_ai_response(
            request_id=payload.request_id,
            image_links=image_links
        )
        logger.info("콜백 전송 성공.", extra=extra_log)
    except Exception as e:
        logger.error(f"콜백 전송 중 예외 발생: {e}", exc_info=True, extra=extra_log)