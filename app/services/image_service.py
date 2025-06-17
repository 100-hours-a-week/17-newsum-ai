# ai/app/services/image_service.py

import os
import httpx
import time
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

        # 헬스 체크용 URL을 구성합니다. (Colab 서버에 /health 경로가 있다고 가정)
        self.health_check_url = f"{self.endpoint}".replace("/generate/text-to-image","")

        self.client: Optional[httpx.AsyncClient] = None
        self.timeout = timeout

        # 서비스의 준비 상태와 헬스 체크 태스크를 관리합니다.
        self.is_ready = False
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_running = False  # 배치 전체 생성 중 True, 끝나면 False

        # 이미지 저장 경로 설정
        self.storage_base_path = settings.IMAGE_STORAGE_PATH
        if self.storage_base_path:
            os.makedirs(self.storage_base_path, exist_ok=True)
        else:
            self.logger.warning("IMAGE_STORAGE_PATH가 설정되지 않았습니다. 이미지를 로컬에 저장할 수 없습니다.")

    async def initialize_service(self):
        """
        서비스를 시작하고, 초기 헬스 체크를 수행한 뒤 백그라운드 헬스 체크를 시작합니다.
        중복 호출 방어 로직(if문)만 유지 (lock 미적용)
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
        """단일 헬스 체크를 수행하고 성공 여부를 반환합니다. 예외 종류별 상세 로깅."""
        if not self.client:
            return False
        try:
            response = await self.client.get(self.health_check_url, timeout=30)
            response.raise_for_status()
            return True
        except httpx.TimeoutException as e:
            self.logger.warning(f"헬스 체크 타임아웃: {e}")
            return False
        except httpx.ConnectError as e:
            self.logger.warning(f"헬스 체크 연결 오류(DNS/네트워크): {e}")
            return False
        except httpx.RequestError as e:
            self.logger.warning(f"헬스 체크 기타 네트워크 오류: {e}")
            return False
        except httpx.HTTPStatusError as e:
            self.logger.warning(f"헬스 체크 실패: 서버가 비정상 상태 코드를 반환했습니다. Status: {e.response.status_code}")
            return False
        except Exception as e:
            self.logger.error(f"헬스 체크 중 예기치 않은 오류 발생: {e}", exc_info=True)
            return False

    async def _health_check_loop(self):
        """
        정상(healthy) 시 5분(300초), 비정상(unhealthy) 시 1분(60초) 간격으로 헬스체크를 반복합니다.
        """
        healthy_interval = 300  # 5분
        unhealthy_interval = 60  # 1분
        while True:
            try:
                ok = await self._check_health()
                self.is_ready = ok
            except Exception as e:
                self.logger.error(f"헬스 체크 루프 예외: {e}", exc_info=True)
                self.is_ready = False
            interval = healthy_interval if self.is_ready else unhealthy_interval
            await asyncio.sleep(interval)

    # [수정됨] 메서드 시그니처에 request_id와 image_index 추가
    async def generate_image(self, model_name: str, prompt: str, request_id: str, image_index: int, **kwargs) -> \
    Dict[str, Any]:
        """이미지 생성을 요청합니다. 서비스가 준비된 경우에만 작동합니다."""
        # 1차 안전장치: 서비스 준비 상태 체크
        if not self.is_ready or not self.client:
            self.logger.error("서비스가 준비되지 않아 이미지 생성 요청을 거부합니다.")
            return {"error": "Image generation service is not available or unhealthy."}

        payload = {
            "model_name": model_name,
            "prompt": prompt,
        }
        if "negative_prompt" in kwargs and kwargs["negative_prompt"]:
            payload["negative_prompt"] = kwargs["negative_prompt"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            payload["seed"] = kwargs["seed"]

        headers = {"Content-Type": "application/json", "Accept": "*/*"}
        api_url = str(self.endpoint)

        try:
            # 2차 안전장치: 요청 직전에도 is_ready 체크
            if not self.is_ready:
                self.logger.error("이미지 생성 직전 is_ready가 False로 변경됨. 요청 중단.")
                return {"error": "Image generation service became unavailable just before request."}

            self.logger.debug(f"이미지 생성 요청: URL={api_url}, Payload={summarize_for_logging(payload)}")
            response = await self.client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if 'image' in content_type and self.storage_base_path:
                # [수정됨] 요청에 따라 저장 경로와 파일명을 새로 구성합니다.
                save_directory = os.path.join(self.storage_base_path, request_id)
                os.makedirs(save_directory, exist_ok=True)

                filename = f"{image_index}.png"
                filepath = os.path.join(save_directory, filename)

                with open(filepath, 'wb') as f:
                    f.write(response.content)
                self.logger.info(f"이미지 저장 성공: {filepath}")
                return {"image_path": filepath, "image_filename": filename}
            elif 'application/json' in content_type:
                return response.json()
            else:
                return {"error": f"Unexpected response content-type: {content_type}"}

        except httpx.HTTPStatusError as e:
            self.logger.error(f"API 오류: Status={e.response.status_code}, Response={e.response.text[:200]}")
            return {"error": f"API call failed (Status {e.response.status_code})", "details": e.response.text}
        except Exception as e:
            self.logger.error(f"이미지 생성 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}

    async def close(self):
        """서비스를 종료하고 모든 자원을 정리합니다."""
        self.logger.info("ImageService 종료 절차 시작...")
        if self.health_check_task:
            if not self.health_check_task.done():
                self.health_check_task.cancel()
            self.logger.info("헬스 체크 태스크가 취소되었습니다.")

        if self.client:
            await self.client.aclose()
            self.logger.info("httpx 클라이언트가 성공적으로 닫혔습니다.")