# app/services/image_server_client_v2.py

import os
import httpx
import time
import json # JSON 응답 처리 시 필요할 수 있음
from typing import Dict, Any, Optional
from app.config.settings import settings
from app.utils.logger import get_logger


class ImageGenerationClient: # 클래스 이름은 파일명과 유사하게 유지하거나 ImageGenerationService로 변경 가능
    """
    이미지 생성 API와 상호작용하는 서비스 (ImageGenerationService 스타일)
    환경 변수 또는 인자를 통한 설정, 유연한 파라미터 및 응답 처리, 오류 반환 방식을 지원합니다.
    """
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_token: Optional[str] = None,
        storage_base_path: Optional[str] = None,
        timeout: float = 120.0, # 타임아웃 늘림
        logger_name: str = "ImageGenerationClient"
    ):
        """
        이미지 생성 클라이언트 초기화

        Args:
            endpoint (Optional[str]): API 엔드포인트 URL. 기본값: settings.IMAGE_SERVER_URL.
            api_token (Optional[str]): 인증 토큰. 기본값: settings.IMAGE_SERVER_API_TOKEN.
            storage_base_path (Optional[str]): 생성된 이미지 저장 기본 경로. 기본값: settings.IMAGE_STORAGE_PATH.
            timeout (float): 요청 타임아웃(초). 기본값: 120.0.
            logger_name (str): 로거 인스턴스 이름.
        """
        # 변경: os.getenv 대신 settings 객체 사용
        self.endpoint = endpoint or settings.IMAGE_SERVER_URL
        self.api_token = api_token or settings.IMAGE_SERVER_API_TOKEN
        self.storage_base_path = storage_base_path or settings.IMAGE_STORAGE_PATH

        # 로거 설정
        self.logger = get_logger(logger_name)

        # 저장 디렉토리 생성 (settings에서 이미 생성하지만, 인자로 경로 지정 시 대비해 유지)
        try:
            os.makedirs(self.storage_base_path, exist_ok=True)
        except OSError as e:
             self.logger.error(f"이미지 저장 디렉토리 생성 실패: {self.storage_base_path}. 오류: {e}", exc_info=True)
             # 디렉토리 생성 실패 시에도 일단 진행은 하되, 파일 저장 시 오류 발생 가능성 있음
             # raise e # 필요 시 여기서 예외 발생시켜 초기화 중단

        # 필수 설정 값 확인 (Endpoint)
        if not self.endpoint:
            self.logger.error("Image generation service endpoint가 설정되지 않았습니다 (settings 또는 인자 필요).")
            raise ValueError("Image generation service endpoint must be provided via argument or settings.IMAGE_SERVER_URL")

        # 비동기 HTTP 클라이언트 생성
        try:
            self.client = httpx.AsyncClient(timeout=timeout)
            self.logger.info(f"ImageGenerationClient 초기화 완료. Endpoint: {self.endpoint}, Storage: {self.storage_base_path}")
        except Exception as e:
            self.logger.error(f"httpx.AsyncClient 초기화 실패: {e}", exc_info=True)
            self.client = None
            raise ConnectionError(f"Failed to initialize httpx client: {e}") from e



    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        **kwargs # 추가적인 API 파라미터를 위한 kwargs
    ) -> Dict[str, Any]:
        """
        설정된 서비스를 사용하여 이미지를 생성합니다.

        Args:
            prompt (str): 이미지 생성을 위한 텍스트 프롬프트.
            negative_prompt (Optional[str]): 네거티브 프롬프트.
            **kwargs: API에 전달할 추가 생성 파라미터 (예: seed, num_inference_steps 등).

        Returns:
            Dict[str, Any]: 성공 시 이미지 경로/파일명 또는 URL이 포함된 딕셔너리.
                           실패 시 {"error": "..."} 형태의 딕셔너리.
        """
        if not self.client:
            self.logger.error("ImageGenerationClient가 초기화되지 않아 generate_image 작업을 수행할 수 없습니다.")
            return {"error": "ImageGenerationClient is not initialized."}

        # 입력값 검증
        if not prompt:
            self.logger.error("빈 프롬프트가 이미지 생성 서비스에 제공되었습니다.")
            return {"error": "Prompt cannot be empty"}

        # 요청 페이로드 준비 (kwargs 포함)
        payload = {
            "prompt": prompt,
            **kwargs # kwargs를 페이로드에 병합
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        # 요청 헤더 준비 (인증 토큰 포함)
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*" # 다양한 응답 타입을 받을 수 있도록 설정 (image/*, application/json 등)
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        try:
            # 요청 상세 로깅
            self.logger.debug(f"이미지 생성 요청: URL={self.endpoint}, Headers={list(headers.keys())}, Payload={payload}")

            # API 호출
            response = await self.client.post(
                self.endpoint,
                json=payload,
                headers=headers
            )
            response.raise_for_status() # HTTP 오류 시 예외 발생 (아래에서 처리)

            # 응답 Content-Type에 따른 처리
            content_type = response.headers.get('content-type', '').lower()
            self.logger.debug(f"이미지 생성 응답 수신: Status={response.status_code}, Content-Type='{content_type}'")

            # 파일명/경로 생성 (타임스탬프 기반)
            timestamp = int(time.time() * 1000) # 밀리초까지 사용하여 동시 요청 시 충돌 방지 강화
            filename = f"generated_{timestamp}.png" # 우선 png로 가정
            filepath = os.path.join(self.storage_base_path, filename)

            if 'image' in content_type:
                # Case 1: 응답이 이미지 바이트인 경우
                image_bytes = response.content
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
                     return {"error": f"Failed to save image file: {e}"}

            elif 'application/json' in content_type:
                # Case 2: 응답이 JSON인 경우 (이미지 URL 포함 기대)
                try:
                    result = response.json()
                    image_url = result.get("image_url") # API 응답 형식에 따라 키 조정 필요

                    if image_url:
                        self.logger.info(f"이미지 URL 수신: {image_url}")
                        return {
                            "image_url": image_url,
                            "raw_response": result # 원본 JSON 응답도 포함
                        }
                    else:
                        self.logger.error(f"JSON 응답에 'image_url' 필드가 없습니다. Response: {result}")
                        return {"error": "No 'image_url' found in JSON response"}
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON 응답 파싱 실패: {e}. Response text: {response.text[:200]}...", exc_info=True)
                    return {"error": f"Failed to parse JSON response: {e}"}

            else:
                # Case 3: 예상치 못한 Content-Type
                self.logger.error(f"예상치 못한 응답 Content-Type: {content_type}. Response text: {response.text[:200]}...")
                return {"error": f"Unexpected response content-type: {content_type}"}

        except httpx.TimeoutException as e:
            self.logger.error(f"이미지 생성 요청 타임아웃 발생: {e}", exc_info=True)
            return {"error": f"Request timed out: {e}"}
        except httpx.RequestError as e:
            self.logger.error(f"이미지 생성 중 네트워크 오류 발생: {e}", exc_info=True)
            return {"error": f"Network error: {e}"}
        except httpx.HTTPStatusError as e:
            # response.raise_for_status()에서 발생한 예외 처리
            self.logger.error(
                f"이미지 생성 API HTTP 오류: Status={e.response.status_code}, Response={e.response.text[:200]}...",
                exc_info=True # 스택 트레이스 없이 간단히 로깅
            )
            return {
                "error": f"API call failed (Status {e.response.status_code}): {e.response.text}"
            }
        except Exception as e:
            # 기타 예상치 못한 오류
            self.logger.error(f"이미지 생성 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return {"error": f"Unexpected error: {e}"}

    async def close(self):
        """
        HTTP 클라이언트 세션을 닫습니다.
        """
        if self.client:
            try:
                await self.client.aclose()
                self.logger.info("ImageGenerationClient의 httpx 클라이언트가 성공적으로 닫혔습니다.")
                self.client = None
            except Exception as e:
                self.logger.error(f"ImageGenerationClient의 httpx 클라이언트 닫기 실패: {e}", exc_info=True)
        else:
            self.logger.info("ImageGenerationClient의 httpx 클라이언트가 이미 닫혀 있거나 초기화되지 않았습니다.")


# --- 선택적 싱글턴 인스턴스 ---
# 필요에 따라 주석 해제하여 사용
# image_generation_client = ImageGenerationClient()

# --- 사용 예시 ---
# async def main():
#     try:
#         # 환경 변수 기반 초기화
#         client = ImageGenerationClient()
#     except (ValueError, ConnectionError) as e:
#         logger.critical(f"클라이언트 초기화 실패: {e}")
#         return

#     # 이미지 생성 요청
#     result = await client.generate_image(
#         prompt="A serene landscape with mountains and a lake",
#         negative_prompt="ugly, deformed, noisy",
#         seed=12345, # kwargs로 전달
#         num_inference_steps=25 # kwargs로 전달
#     )

#     if "error" in result:
#         print(f"Error: {result['error']}")
#     else:
#         print(f"Success! Result: {result}")

#     # 종료 시 클라이언트 닫기
#     await client.close()

# import asyncio
# if __name__ == "__main__":
#      # 로깅 설정 확인
#      if not get_logger("ImageGenerationClient").handlers:
#           logging.basicConfig(level=logging.DEBUG) # 디버그 레벨로 변경하여 상세 로그 확인
#      asyncio.run(main())