# image_service.py
import os
import httpx
import asyncio
from typing import Dict, Any, Optional

from app.config.settings import Settings
from app.utils.logger import get_logger, summarize_for_logging

settings = Settings()

class ImageService:
    def __init__(
        self,
        timeout: float = 120.0,
        logger_name: str = "ImageGenerationClient"
    ):
        self.logger = get_logger(logger_name)

        # 설정에서 고정 엔드포인트를 가져옵니다.
        self.endpoint = settings.IMAGE_SERVER_URL
        if not self.endpoint:
            self.logger.error("IMAGE_SERVER_URL이 설정되지 않았습니다.")
            raise ValueError("IMAGE_SERVER_URL must be set in settings.")

        # 헬스 체크용 URL을 구성합니다.
        self.health_check_url = self.endpoint.rstrip("/") + "/health"

        self.client: Optional[httpx.AsyncClient] = None
        self.timeout = timeout

        # 서비스 준비 상태 및 헬스 체크 태스크
        self.is_ready = False
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_running = False  # 배치 전체 생성 중 True

        # 이미지 저장 경로 설정
        self.storage_base_path = settings.IMAGE_STORAGE_PATH
        if self.storage_base_path:
            os.makedirs(self.storage_base_path, exist_ok=True)
        else:
            self.logger.warning("IMAGE_STORAGE_PATH가 설정되지 않았습니다. 이미지를 로컬에 저장할 수 없습니다.")

    async def initialize_service(self):
        """
        서비스를 시작하고, 초기 헬스 체크를 수행한 뒤 백그라운드 헬스 체크를 시작합니다.
        중복 호출 방어 로직 추가
        """
        # 중복 태스크 방어: 이미 실행 중이면 재생성 안 함
        if self.health_check_task and not self.health_check_task.done():
            self.logger.info("헬스 체크 루프가 이미 실행 중입니다. 새로운 태스크를 생성하지 않습니다.")
            return

        self.logger.info("ImageService 초기화 시작...")
        self.client = httpx.AsyncClient(timeout=self.timeout)

        self.logger.info(f"초기 헬스 체크 수행: {self.health_check_url}")
        is_initially_healthy = await self._check_health()

        if is_initially_healthy:
            self.is_ready = True
            self.logger.info("초기 헬스 체크 성공. 서비스가 준비되었습니다.")
        else:
            self.is_ready = False
            self.logger.warning("초기 헬스 체크 실패. 서버가 응답하지 않습니다. 백그라운드에서 계속 확인합니다.")

        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.logger.info("백그라운드 헬스 체크 루프가 시작되었습니다.")

    async def _check_health(self) -> bool:
        """단일 헬스 체크를 수행하고 성공 여부를 반환합니다."""
        if not self.client:
            return False
        try:
            response = await self.client.get(self.health_check_url, timeout=30)
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.warning(f"헬스 체크 실패: {e}")
            return False

    async def _health_check_loop(self):
        """
        최초 1분 대기 후, 정상 시 5분, 장애 시 1분, 배치 중 스킵 30초
        """
        await asyncio.sleep(60)
        interval = 300
        skip_interval = 30
        first_success = False

        try:
            while True:
                if self.is_running:
                    self.logger.info("현재 이미지 배치 생성이 진행중입니다. health check를 skip합니다.")
                    await asyncio.sleep(skip_interval)
                    continue

                is_healthy = await self._check_health()
                if is_healthy and not self.is_ready:
                    self.logger.info("헬스 체크 성공: 서비스가 다시 준비 상태가 되었습니다.")
                    self.is_ready = True
                elif not is_healthy and self.is_ready:
                    self.logger.error("헬스 체크 실패: 서비스가 응답하지 않아 '준비되지 않음' 상태로 변경됩니다.")
                    self.is_ready = False

                if not is_healthy:
                    interval = 60
                    first_success = False
                elif not first_success:
                    interval = 300
                    first_success = True

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            self.logger.info("헬스 체크 루프가 CancelledError로 안전하게 종료되었습니다.")
            return

    # generate_image, close 등은 기존과 동일하게 유지