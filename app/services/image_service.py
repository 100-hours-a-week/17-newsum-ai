# ai/app/services/image_service.py

import os
import httpx
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
        self.endpoint = settings.IMAGE_SERVER_URL
        if not self.endpoint:
            self.logger.error("IMAGE_SERVER_URL이 설정되지 않았습니다.")
            raise ValueError("IMAGE_SERVER_URL must be set in settings.")
        self.health_check_url = f"{self.endpoint}".replace("/generate/text-to-image","")
        self.client: Optional[httpx.AsyncClient] = httpx.AsyncClient(timeout=timeout)
        if not self.client:
            self.logger.error("httpx.AsyncClient가 설정되지 않았습니다.")
            raise ValueError("httpx.AsyncClient must be set in settings.")
        else:
            self.logger.info("httpx.AsyncClient가 성공적으로 설정되었습니다.")
        self.timeout = timeout
        self.storage_base_path = settings.IMAGE_STORAGE_PATH
        if self.storage_base_path:
            os.makedirs(self.storage_base_path, exist_ok=True)
        else:
            self.logger.warning("IMAGE_STORAGE_PATH가 설정되지 않았습니다. 이미지를 로컬에 저장할 수 없습니다.")

    async def check_health(self) -> bool:
        """이미지 서버의 헬스 상태를 실시간으로 확인합니다."""
        if not self.client:
            return False
        try:
            response = await self.client.get(self.health_check_url, timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.warning(f"헬스체크 실패: {e}")
            return False

    async def generate_image(self, model_name: str, prompt: str, request_id: str, image_index: int, **kwargs) -> \
    Dict[str, Any]:
        """이미지 생성을 요청합니다. 서비스가 준비된 경우에만 작동합니다."""
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
            self.logger.debug(f"이미지 생성 요청: URL={api_url}, Payload={summarize_for_logging(payload)}")
            response = await self.client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if 'image' in content_type and self.storage_base_path:
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
        if self.client:
            await self.client.aclose()
            self.logger.info("httpx 클라이언트가 성공적으로 닫혔습니다.")