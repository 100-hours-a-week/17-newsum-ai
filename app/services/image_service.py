# ai/app/services/image_service.py

import os
import httpx
import time
import json
from typing import Dict, Any, Optional

# settings.py 와 logger.py 는 이미 app.config 와 app.utils 에 있다고 가정
from app.config.settings import Settings
from app.utils.logger import get_logger, summarize_for_logging

settings = Settings()


class ImageService:
    def __init__(
            self,
            endpoint: Optional[str] = None,  # settings.IMAGE_SERVER_URL이 AnyHttpUrl 타입
            api_token: Optional[str] = None,
            storage_base_path: Optional[str] = None,
            timeout: float = 120.0,
            logger_name: str = "ImageGenerationClient"
    ):
        self.logger = get_logger(logger_name)

        raw_endpoint = endpoint or settings.IMAGE_SERVER_URL
        if not raw_endpoint:
            self.logger.error("Image generation service endpoint가 설정되지 않았습니다 (settings 또는 인자 필요).")
            raise ValueError(
                "Image generation service endpoint must be provided via argument or settings.IMAGE_SERVER_URL")

        self.endpoint = str(raw_endpoint)  # <<< TypeError 방지를 위해 문자열로 변환

        self.api_token = api_token or settings.IMAGE_SERVER_API_TOKEN
        # IMAGE_STORAGE_PATH는 settings.py에서 str 타입으로 정의되어 있음
        self.storage_base_path = storage_base_path or settings.IMAGE_STORAGE_PATH

        try:
            # storage_base_path가 None이 아닐 경우에만 디렉토리 생성 시도
            if self.storage_base_path:
                os.makedirs(self.storage_base_path, exist_ok=True)
            else:
                # storage_base_path가 설정되지 않은 경우 로깅 (파일 저장 기능 사용 불가)
                self.logger.warning("IMAGE_STORAGE_PATH가 설정되지 않았습니다. 이미지를 로컬에 저장할 수 없습니다.")

        except OSError as e:
            self.logger.error(f"이미지 저장 디렉토리 생성 실패: {self.storage_base_path}. 오류: {e}", exc_info=True)
            # 필요 시 여기서 예외 발생

        if not self.endpoint:  # 이미 위에서 체크했지만, 이중 확인
            self.logger.error("Image generation service endpoint가 없습니다.")  # 이 로그는 거의 발생 안 함
            raise ValueError("Endpoint is required.")

        try:
            self.client = httpx.AsyncClient(timeout=timeout)
            self.logger.info(
                f"ImageGenerationClient 초기화 완료. Endpoint: {self.endpoint}, Storage: {self.storage_base_path or '저장 경로 미설정'}")
        except Exception as e:
            self.logger.error(f"httpx.AsyncClient 초기화 실패: {e}", exc_info=True)
            self.client = None  # type: ignore
            raise ConnectionError(f"Failed to initialize httpx client: {e}") from e

    # ... (generate_image, close 메소드는 이전 제공 코드와 동일하게 유지) ...
    async def generate_image(
            self,
            model_name: str,
            prompt: str,
            negative_prompt: Optional[str] = None,
            **kwargs  # 추가적인 API 파라미터를 위한 kwargs
    ) -> Dict[str, Any]:
        if not self.client:
            self.logger.error("ImageGenerationClient가 초기화되지 않아 generate_image 작업을 수행할 수 없습니다.")
            return {"error": "ImageGenerationClient is not initialized."}

        if not prompt:
            self.logger.error("빈 프롬프트가 이미지 생성 서비스에 제공되었습니다.")
            return {"error": "Prompt cannot be empty"}

        payload = {"model_name": model_name, "prompt": prompt, **kwargs}
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*"
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        try:
            self.logger.debug(
                f"이미지 생성 요청: URL={self.endpoint}, Headers={list(headers.keys())}, Payload={summarize_for_logging(payload)}")  # summarize_for_logging 임포트 필요

            response = await self.client.post(
                self.endpoint,  # self.endpoint는 이미 str 타입
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            self.logger.debug(f"이미지 생성 응답 수신: Status={response.status_code}, Content-Type='{content_type}'")

            timestamp = int(time.time() * 1000)
            filename = f"generated_{timestamp}.png"

            # storage_base_path가 설정된 경우에만 파일 경로 구성 및 저장 시도
            filepath = None
            if self.storage_base_path:
                filepath = os.path.join(self.storage_base_path, filename)

            if 'image' in content_type:
                image_bytes = response.content
                if filepath:
                    try:
                        with open(filepath, 'wb') as f:
                            f.write(image_bytes)
                        self.logger.info(f"이미지 로컬 저장 성공: {filepath}")
                        return {
                            "image_path": filepath,
                            "image_filename": filename,
                            "content_type": content_type
                        }
                    except IOError as e:
                        self.logger.error(f"이미지 파일 저장 실패: {e}", exc_info=True)
                        return {"error": f"Failed to save image file: {e}", "image_url": None}  # URL 반환도 고려
                else:  # 저장 경로 없으면 URL만 반환 (만약 API가 URL도 준다면) 또는 오류
                    self.logger.warning("이미지 바이트를 받았으나 로컬 저장 경로(IMAGE_STORAGE_PATH)가 설정되지 않았습니다.")
                    # 이 경우, API가 이미지 URL을 별도로 제공하지 않는다면 처리 곤란.
                    # 여기서는 오류로 처리하거나, 임시 저장 후 경로 반환 등의 로직이 필요할 수 있음.
                    # 지금은 image_path 없이 반환. N09에서 이 경우를 처리해야 함.
                    return {"error": "Image data received but no storage path configured.",
                            "image_bytes_size": len(image_bytes)}

            elif 'application/json' in content_type:
                try:
                    result = response.json()
                    image_url = result.get("image_url")
                    if image_url:
                        self.logger.info(f"이미지 URL 수신: {image_url}")
                        return {
                            "image_url": image_url,  # N09에서 이 URL을 사용할 수 있음
                            "image_path": None,  # 로컬 저장은 안됨 (또는 URL 다운로드 후 저장 로직 추가)
                            "raw_response": result
                        }
                    else:
                        self.logger.error(f"JSON 응답에 'image_url' 필드가 없습니다. Response: {result}")
                        return {"error": "No 'image_url' found in JSON response"}
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON 응답 파싱 실패: {e}. Response text: {response.text[:200]}...", exc_info=True)
                    return {"error": f"Failed to parse JSON response: {e}"}
            else:
                self.logger.error(f"예상치 못한 응답 Content-Type: {content_type}. Response text: {response.text[:200]}...")
                return {"error": f"Unexpected response content-type: {content_type}"}

        except httpx.TimeoutException as e:
            self.logger.error(f"이미지 생성 요청 타임아웃 발생: {e}", exc_info=True)
            return {"error": f"Request timed out: {e}"}
        except httpx.RequestError as e:  # 네트워크 연결 오류 등
            self.logger.error(f"이미지 생성 중 네트워크 오류 발생: {e}", exc_info=True)
            return {"error": f"Network error: {e}"}
        except httpx.HTTPStatusError as e:  # 4xx, 5xx 오류
            self.logger.error(
                f"이미지 생성 API HTTP 오류: Status={e.response.status_code}, Response={e.response.text[:200]}...",
                exc_info=False  # 스택 트레이스 없이 간단히 로깅
            )
            return {
                "error": f"API call failed (Status {e.response.status_code})",
                "details": e.response.text  # 전체 오류 메시지 포함
            }
        except Exception as e:
            self.logger.error(f"이미지 생성 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return {"error": f"Unexpected error: {e}"}

    async def close(self):
        if self.client:
            try:
                await self.client.aclose()
                self.logger.info("ImageGenerationClient의 httpx 클라이언트가 성공적으로 닫혔습니다.")
                self.client = None  # type: ignore
            except Exception as e:
                self.logger.error(f"ImageGenerationClient의 httpx 클라이언트 닫기 실패: {e}", exc_info=True)
        else:
            self.logger.info("ImageGenerationClient의 httpx 클라이언트가 이미 닫혀 있거나 초기화되지 않았습니다.")

# ImageService의 summarize_for_logging 임포트를 위해 app.utils.logger에서 가져와야 함
# 이 파일에서 직접 사용하지 않으면 필요 없지만, 로깅에서 사용한다면 필요.
# from app.utils.logger import summarize_for_logging # 필요시 추가