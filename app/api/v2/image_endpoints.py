# ai/app/api/v2/image_endpoints.py

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from app.utils.logger import get_logger
from .schemas import (
    ImageHealthResponse,
    BatchImageGenerationRequest,
    AcceptedResponse
)
# 의존성 주입을 위한 서비스 및 클라이언트 임포트 (dependencies.py에 정의되어야 함)
from app.dependencies import (
    ImageServiceDep,
    StorageServiceDep,
    BackendApiClientDep
)
from .image_background_tasks import generate_images_in_background

router = APIRouter(prefix="/images", tags=["Image Generation"])
logger = get_logger(__name__)

@router.get(
    "/health",
    response_model=ImageHealthResponse,
    summary="이미지 생성 서비스 건강 상태 확인",
    description="이미지 생성 서비스가 현재 요청을 처리할 수 있는 정상 상태인지 확인합니다."
)
async def check_image_service_health(image_service: ImageServiceDep):
    """
    ImageService의 is_ready 상태를 확인하여 서비스 가용성을 반환합니다.
    """
    await image_service.initialize_service()
    if image_service.is_ready:
        return ImageHealthResponse(status="healthy")
    else:
        logger.warning("이미지 서비스 상태 확인 실패: 서비스가 준비되지 않음.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unhealthy", "message": "Image generation service is not available."},
        )

@router.post(
    "/generate/batch",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AcceptedResponse,
    summary="이미지 배치 생성 및 S3 업로드 요청 (비동기)",
    description="최대 5개의 프롬프트를 받아 이미지 생성을 백그라운드에서 처리하고, 완료 시 콜백으로 S3 경로를 전송합니다."
)
async def batch_generate_images(
    payload: BatchImageGenerationRequest,
    background_tasks: BackgroundTasks,
    image_service: ImageServiceDep,
    storage_service: StorageServiceDep,
    backend_client: BackendApiClientDep,
):
    """
    요청을 즉시 수락(202 Accepted)하고, 실제 작업은 백그라운드 태스크로 위임합니다.
    """
    logger.info(f"배치 이미지 생성 요청 수신: {payload.request_id}")

    background_tasks.add_task(
        generate_images_in_background,
        payload=payload,
        image_service=image_service,
        storage_service=storage_service,
        backend_client=backend_client,
    )

    return AcceptedResponse(request_id=payload.request_id)