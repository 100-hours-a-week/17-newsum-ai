# app/services/storage_client_v2.py

import os
import uuid
import mimetypes
from typing import Dict, Any, Optional
from app.config.settings import settings
from app.utils.logger import get_logger

# aiobotocore 및 관련 라이브러리 import
try:
    import aiobotocore.session
    from aiobotocore.config import AioConfig
    from botocore.exceptions import ClientError, BotoCoreError
    import aiofiles # 비동기 파일 읽기용
    AIOBOTOCORE_AVAILABLE = True
except ImportError:
    AIOBOTOCORE_AVAILABLE = False
    logger_temp = get_logger("StorageClient_Init")
    logger_temp.warning("aiobotocore 또는 aiofiles 라이브러리가 설치되지 않았습니다. StorageClient 기능이 비활성화됩니다.")
    # Define dummy classes if not available
    class ClientError(Exception): pass
    class BotoCoreError(Exception): pass

class StorageClient:
    """
    비동기 AWS S3 스토리지 클라이언트 (aiobotocore 사용).
    파일 업로드 및 Presigned URL 생성을 지원합니다.
    AWS 자격 증명은 표준 검색 경로(환경 변수, IAM 역할 등)를 통해 자동으로 로드됩니다.
    """
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None,
        logger_name: str = "StorageClient"
    ):
        self.logger = get_logger(logger_name)
        self.s3_client = None
        self._session = None
        self.use_local_fallback = False # 로컬 폴백 사용 여부 플래그
        self.local_storage_path = settings.LOCAL_STORAGE_PATH # 설정에서 로컬 경로 읽기

        if not AIOBOTOCORE_AVAILABLE:
            self.logger.error("필수 라이브러리(aiobotocore, aiofiles)가 없어 StorageClient를 초기화할 수 없습니다.")
            # 로컬 폴백을 강제하거나, 여기서 에러를 발생시킬지 결정 필요
            # 예: self.use_local_fallback = True 또는 raise ImportError(...)
            # 여기서는 S3 시도 자체를 못하므로 에러 발생이 더 적합할 수 있음
            raise ImportError("aiobotocore and aiofiles must be installed to use StorageClient.")

        # 설정 값 로드
        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME
        self.region_name = region_name or settings.AWS_REGION

        # S3 클라이언트 생성 시도
        try:
            if not self.bucket_name:
                 # 버킷 이름이 없으면 S3는 어차피 사용 불가
                 raise ValueError("S3 bucket name must be provided for S3 mode.")

            self._session = aiobotocore.session.get_session()
            aio_config = AioConfig(connect_timeout=10, read_timeout=60)
            self.s3_client = self._session.create_client(
                's3',
                region_name=self.region_name,
                config=aio_config
            )
            self.logger.info(f"StorageClient S3 모드로 초기화 완료. Bucket: {self.bucket_name}, Region: {self.region_name or 'default'}")

        except Exception as e:
            self.logger.warning(f"aiobotocore S3 클라이언트 생성 실패: {e}. 로컬 스토리지 폴백 모드로 전환합니다 (경로: {self.local_storage_path}).", exc_info=True)
            self.s3_client = None # S3 클라이언트 없음 명시
            self.use_local_fallback = True # 로컬 폴백 활성화
            if not self.local_storage_path:
                self.logger.error("S3 초기화 실패 및 로컬 스토리지 경로(LOCAL_STORAGE_PATH)가 설정되지 않아 StorageClient 사용 불가.")
                # 로컬 경로도 없으면 동작 불가하므로 에러 발생
                raise ValueError("S3 initialization failed and LOCAL_STORAGE_PATH is not configured.")
            # 로컬 경로가 있으면 경고만 하고 객체 생성은 계속 진행

    async def upload_file(
        self,
        file_path: str,
        object_key: Optional[str] = None,
        prefix: str = "uploads/", # 기본 업로드 경로 접두사
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        로컬 파일을 S3 버킷에 비동기적으로 업로드합니다.

        Args:
            file_path (str): 업로드할 로컬 파일 경로.
            object_key (Optional[str]): S3에 저장될 객체 키 (전체 경로). 지정하지 않으면 자동 생성됩니다.
            prefix (str): object_key 자동 생성 시 사용될 접두사. 기본값: "uploads/".
            content_type (Optional[str]): 파일의 MIME 타입. 지정하지 않으면 추측합니다.

        Returns:
            Dict[str, Any]: 성공 시 {"s3_uri": str, "object_key": str}, 실패 시 {"error": str}
        """
        if not self.s3_client:
            self.logger.error("StorageClient가 초기화되지 않아 upload_file 작업을 수행할 수 없습니다.")
            return {"error": "StorageClient is not initialized."}

        if not os.path.exists(file_path):
            self.logger.error(f"업로드할 파일을 찾을 수 없습니다: {file_path}")
            return {"error": f"File not found at path: {file_path}"}

        # 객체 키 생성 (지정되지 않은 경우)
        if not object_key:
            filename = os.path.basename(file_path)
            # 파일 확장자 유지하며 UUID 추가 (충돌 방지)
            name, ext = os.path.splitext(filename)
            unique_id = uuid.uuid4().hex
            object_key = f"{prefix.strip('/')}/{name}_{unique_id}{ext}"

        # ContentType 추측 (지정되지 않은 경우)
        if not content_type:
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = 'application/octet-stream' # 기본값

        try:
            self.logger.info(f"S3 업로드 시작: '{file_path}' -> 's3://{self.bucket_name}/{object_key}' (ContentType: {content_type})")

            # aiofiles를 사용하여 비동기적으로 파일 읽기
            async with aiofiles.open(file_path, mode='rb') as f:
                content = await f.read()

            # S3에 파일 업로드 (put_object 사용)
            # 참고: 매우 큰 파일(GB 단위)의 경우, put_object 대신 aiobotocore의
            #      저수준 API를 사용하여 멀티파트 업로드를 직접 구현해야 할 수 있습니다.
            response = await self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=content,
                ContentType=content_type
                # ACL='public-read' # 필요 시 접근 권한 설정
            )

            # 응답 확인 (선택 사항)
            http_status_code = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
            if http_status_code == 200:
                s3_uri = f"s3://{self.bucket_name}/{object_key}"
                self.logger.info(f"S3 업로드 성공: {s3_uri}")
                return {
                    "s3_uri": s3_uri,
                    "object_key": object_key,
                    "content_type": content_type
                    # "etag": response.get('ETag') # ETag 등 추가 정보 포함 가능
                }
            else:
                 self.logger.error(f"S3 업로드 실패: HTTP Status {http_status_code}, Response: {response}")
                 return {"error": f"S3 upload failed with status code {http_status_code}"}

        except FileNotFoundError:
            self.logger.error(f"업로드 중 파일을 찾을 수 없음: {file_path}", exc_info=True)
            return {"error": f"File not found during upload: {file_path}"}
        except (ClientError, BotoCoreError) as e:
            self.logger.error(f"S3 업로드 중 AWS 오류 발생: {e}", exc_info=True)
            # ClientError의 경우 더 구체적인 오류 코드 확인 가능 (e.g., e.response['Error']['Code'])
            return {"error": f"AWS S3 Error: {e}"}
        except Exception as e:
            self.logger.error(f"S3 업로드 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return {"error": f"Unexpected error during S3 upload: {e}"}

    async def generate_presigned_url(
        self,
        object_key: str,
        expiration: int = 3600 # URL 유효 시간(초), 기본값 1시간
    ) -> Dict[str, Any]:
        """
        S3 객체에 접근하기 위한 Presigned URL을 생성합니다.

        Args:
            object_key (str): URL을 생성할 S3 객체 키.
            expiration (int): URL의 유효 시간(초). 기본값: 3600.

        Returns:
            Dict[str, Any]: 성공 시 {"presigned_url": str}, 실패 시 {"error": str}
        """
        if not self.s3_client:
            self.logger.error("StorageClient가 초기화되지 않아 Presigned URL 생성을 수행할 수 없습니다.")
            return {"error": "StorageClient is not initialized."}

        try:
            self.logger.debug(f"Presigned URL 생성 요청: bucket='{self.bucket_name}', key='{object_key}', expiration={expiration}s")
            url = await self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            self.logger.info(f"Presigned URL 생성 성공: key='{object_key}'")
            return {"presigned_url": url}
        except (ClientError, BotoCoreError) as e:
            self.logger.error(f"Presigned URL 생성 중 AWS 오류 발생: key='{object_key}', error={e}", exc_info=True)
            return {"error": f"AWS S3 Presigned URL Generation Error: {e}"}
        except Exception as e:
            self.logger.error(f"Presigned URL 생성 중 예상치 못한 오류 발생: key='{object_key}', error={e}", exc_info=True)
            return {"error": f"Unexpected error during Presigned URL generation: {e}"}


    async def close(self):
        """
        aiobotocore S3 클라이언트 세션을 닫습니다.
        """
        if self.s3_client:
            try:
                await self.s3_client.close()
                self.logger.info("StorageClient의 aiobotocore S3 클라이언트가 성공적으로 닫혔습니다.")
                self.s3_client = None
            except Exception as e:
                self.logger.error(f"StorageClient의 S3 클라이언트 닫기 실패: {e}", exc_info=True)
        else:
            self.logger.info("StorageClient의 S3 클라이언트가 이미 닫혀 있거나 초기화되지 않았습니다.")


# --- 선택적 싱글턴 인스턴스 ---
# 필요에 따라 주석 해제하여 사용
# storage_client = StorageClient()

# --- 사용 예시 ---
# async def main():
#     # 로깅 레벨 설정 (디버그 메시지 확인용)
#     logging.getLogger("StorageClient").setLevel(logging.DEBUG)

#     try:
#         # 환경 변수 기반 초기화 (S3_BUCKET_NAME 설정 필요)
#         client = StorageClient()
#     except (ValueError, ConnectionError, ImportError) as e:
#         logging.critical(f"클라이언트 초기화 실패: {e}")
#         return

#     # 임시 테스트 파일 생성
#     test_file = "my_test_file.txt"
#     async with aiofiles.open(test_file, "w") as f:
#         await f.write("Hello from StorageClient test!")

#     # 파일 업로드
#     upload_result = await client.upload_file(test_file, prefix="test_uploads/")

#     if "error" in upload_result:
#         print(f"Upload Error: {upload_result['error']}")
#     else:
#         print(f"Upload Success: {upload_result}")
#         object_key = upload_result.get("object_key")

#         # Presigned URL 생성
#         if object_key:
#             url_result = await client.generate_presigned_url(object_key, expiration=600) # 10분 유효
#             if "error" in url_result:
#                 print(f"URL Gen Error: {url_result['error']}")
#             else:
#                 print(f"Presigned URL: {url_result['presigned_url']}")

#     # 임시 파일 삭제
#     if os.path.exists(test_file):
#         os.remove(test_file)

#     # 종료 시 클라이언트 닫기
#     await client.close()

# import asyncio
# if __name__ == "__main__":
#      # 로깅 설정 확인
#      if not get_logger("StorageClient").handlers:
#           logging.basicConfig(level=logging.DEBUG) # 기본 로깅 설정
#      asyncio.run(main())