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
    PostgreSQLServiceDep,
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
    ImageService의 실시간 헬스체크 결과를 반환합니다.
    """
    is_healthy = await image_service.check_health()
    if is_healthy:
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
    pg_service: PostgreSQLServiceDep,
    image_service: ImageServiceDep,
    storage_service: StorageServiceDep,
    backend_client: BackendApiClientDep,
):
    """
    요청을 즉시 수락(202 Accepted)하고, 실제 작업은 백그라운드 태스크로 위임합니다.
    이미지 생성 전 실시간 헬스체크를 수행합니다.
    """
    logger.info(f"배치 이미지 생성 요청 수신: {payload.id}")

    # 1. job_id 중복 확인
    is_running = await pg_service.is_image_job_running(payload.id)
    if is_running:
        logger.warning(f"중복된 job_id: {payload.id}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"status": "duplicate", "message": f"Job {payload.id} is already running."},
        )

    # 2. 기존 health check
    is_healthy = await image_service.check_health()
    if not is_healthy:
        logger.warning("이미지 생성 요청 시 이미지 서버가 준비되지 않음.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unhealthy", "message": "Image generation service is not available."},
        )

    background_tasks.add_task(
        generate_images_in_background,
        payload=payload,
        pg_service=pg_service,
        image_service=image_service,
        storage_service=storage_service,
        backend_client=backend_client,
    )

    return AcceptedResponse(request_id=payload.id)