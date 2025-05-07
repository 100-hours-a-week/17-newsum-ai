# ai/app/services/storage_service.py

import os
import uuid
import mimetypes
import asyncio  # run_in_executor 사용 위해 추가
from typing import Dict, Any, Optional, Union
from functools import partial  # run_in_executor 에 인자 전달 위해 추가
from types import SimpleNamespace  # main 테스트용 SimpleNamespace

from app.config.settings import Settings
from app.utils.logger import get_logger

settings = Settings()  # StorageService 클래스 로딩 시 settings 변수가 필요해서 임시 사용

# boto3 및 관련 예외 import
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger_temp = get_logger("StorageClient_Init")
    logger_temp.warning("boto3 라이브러리가 설치되지 않았습니다. StorageClient 기능이 비활성화됩니다.")
    class ClientError(Exception):
        pass
    class BotoCoreError(Exception):
        pass
try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    logger_temp = get_logger("StorageClient_Init_Aiofiles")
    logger_temp.warning("aiofiles 라이브러리가 설치되지 않았습니다. 로컬 폴백 시 비동기 파일 처리가 비활성화될 수 있습니다.")

logger = get_logger("StorageClientV2_Boto3_Updated")


class StorageService:
    """
    동기 boto3를 사용하여 비동기 환경에서 AWS S3 스토리지와 상호작용하는 클라이언트.
    파일 업로드 및 Presigned URL 생성을 지원합니다. run_in_executor 사용.
    AWS 자격 증명은 settings 객체를 통해 명시적으로 제공되거나, 표준 검색 경로를 통해 자동 로드될 수 있습니다.
    """

    def __init__(
            self,
            bucket_name: Optional[str] = None,
            region_name: Optional[str] = None,
            logger_name: str = "StorageClientV2_Boto3_Updated",
            settings_override: Optional[Any] = None  # 테스트 또는 외부 설정 주입용
    ):
        self.logger = get_logger(logger_name)

        self.s3_client = None
        self.use_local_fallback = False
        self.local_storage_path = settings.LOCAL_STORAGE_PATH

        if not BOTO3_AVAILABLE:
            self.logger.error("필수 라이브러리(boto3)가 없어 StorageClient를 초기화할 수 없습니다.")
            raise ImportError("boto3 must be installed to use StorageClient.")

        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME
        self.region_name = region_name or settings.AWS_REGION

        s3_client_args = {}
        if self.region_name:
            s3_client_args['region_name'] = self.region_name

        # 설정에서 명시적 AWS 키 사용 (제안 내용 반영)
        # 실제 Settings 클래스에 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY 필드가 있다고 가정
        if hasattr(settings, 'ACCESS_KEY') and \
                hasattr(settings, 'SECRET_KEY') and \
                settings.ACCESS_KEY and \
                settings.SECRET_KEY:

            self.logger.warning("명시적인 AWS Access Key ID와 Secret Access Key를 사용하여 S3 클라이언트를 초기화합니다. ")
            s3_client_args['aws_access_key_id'] = settings.ACCESS_KEY
            s3_client_args['aws_secret_access_key'] = settings.SECRET_KEY
        else:
            self.logger.info(
                "명시적 AWS 키가 제공되지 않았습니다. AWS SDK 기본 자격 증명 체인(IAM 역할, 환경 변수, 공유 자격 증명 파일 등)을 사용합니다."
            )

        try:
            if not self.bucket_name:
                raise ValueError(
                    "S3 bucket name must be provided either as an argument or in settings (S3_BUCKET_NAME).")

            self.s3_client = boto3.client('s3', **s3_client_args)
            # 간단한 테스트 호출 (선택 사항) - 초기화 시 자격 증명 확인 등 (예: list_buckets - 권한 필요)
            # self.s3_client.head_bucket(Bucket=self.bucket_name) # 버킷 존재 및 접근 가능 여부 확인
            self.logger.info(
                f"StorageClient (boto3) S3 모드로 초기화 시도. Bucket: {self.bucket_name}, Region: {self.region_name or 'default'}")
            # 실제로 head_bucket 등을 호출하여 연결을 확인하는 것이 좋습니다.
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                self.logger.info(f"S3 Bucket '{self.bucket_name}'에 성공적으로 연결되었습니다.")
            except (ClientError, BotoCoreError) as e:
                self.logger.warning(
                    f"S3 Bucket '{self.bucket_name}' 연결 테스트 실패 (AWS 오류): {e}. 로컬 스토리지 폴백 모드로 전환합니다 (경로: {self.local_storage_path}).",
                    exc_info=True)
                self.s3_client = None  # 클라이언트 사용 불가 처리
                self.use_local_fallback = True


        except ValueError as e:  # bucket_name 누락 등의 설정 오류
            self.logger.error(f"StorageClient 설정 오류: {e}. 로컬 스토리지 폴백 모드로 전환합니다 (경로: {self.local_storage_path}).",
                                exc_info=True)
            self.s3_client = None
            self.use_local_fallback = True
        except (ClientError, BotoCoreError) as e:  # AWS 관련 초기화 실패
            self.logger.warning(
                f"boto3 S3 클라이언트 생성 실패 (AWS 오류): {e}. 로컬 스토리지 폴백 모드로 전환합니다 (경로: {self.local_storage_path}).",
                exc_info=True)
            self.s3_client = None
            self.use_local_fallback = True
        except Exception as e:  # 기타 예상치 못한 오류
            self.logger.warning(
                f"boto3 S3 클라이언트 생성 중 예상치 못한 오류: {e}. 로컬 스토리지 폴백 모드로 전환합니다 (경로: {self.local_storage_path}).",
                exc_info=True)
            self.s3_client = None
            self.use_local_fallback = True

        if self.use_local_fallback and not self.local_storage_path:
            self.logger.error("S3 초기화 실패 및 로컬 스토리지 경로(LOCAL_STORAGE_PATH)가 설정되지 않아 StorageClient 사용 불가.")
            # 이 경우 StorageService 사용이 불가능하므로 더 강력한 에러 발생 또는 상태 표시 필요
            raise ValueError("S3 initialization failed and LOCAL_STORAGE_PATH is not configured for fallback.")

    async def upload_file(
            self,
            file_path: str,
            object_key: Optional[str] = None,
            prefix: str = "uploads/",
            content_type: Optional[str] = None,
            acl: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.use_local_fallback:
            return await self._upload_file_local(file_path, object_key, prefix)

        if not self.s3_client:
            self.logger.error("StorageClient(S3)가 초기화되지 않아 upload_file 작업을 수행할 수 없습니다.")
            return {"error": "StorageClient S3 client is not initialized."}

        if not os.path.exists(file_path):
            self.logger.error(f"업로드할 파일을 찾을 수 없습니다: {file_path}")
            return {"error": f"File not found at path: {file_path}"}

        if not object_key:
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            unique_id = uuid.uuid4().hex
            object_key = f"{prefix.strip('/')}/{name}_{unique_id}{ext}"

        if not content_type:
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = 'application/octet-stream'

        try:
            self.logger.info(
                f"S3 업로드 시작 (boto3/executor): '{file_path}' -> 's3://{self.bucket_name}/{object_key}' (ContentType: {content_type})")
            loop = asyncio.get_running_loop()

            extra_args = {'ContentType': content_type}
            if acl:
                extra_args['ACL'] = acl

            upload_func = partial(
                self.s3_client.upload_file,
                Filename=file_path,
                Bucket=self.bucket_name,
                Key=object_key,
                ExtraArgs=extra_args
            )
            await loop.run_in_executor(None, upload_func)

            s3_uri = f"s3://{self.bucket_name}/{object_key}"
            self.logger.info(f"S3 업로드 성공: {s3_uri}")
            return {
                "s3_uri": s3_uri,
                "object_key": object_key,
                "content_type": content_type
            }
        except FileNotFoundError:
            self.logger.error(f"업로드 중 파일을 찾을 수 없음: {file_path}", exc_info=True)
            return {"error": f"File not found during upload: {file_path}"}
        except (ClientError, BotoCoreError) as e:
            self.logger.error(f"S3 업로드 중 AWS 오류 발생: {e}", exc_info=True)
            return {"error": f"AWS S3 Error: {e}"}
        except Exception as e:
            self.logger.error(f"S3 업로드 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return {"error": f"Unexpected error during S3 upload: {e}"}

    async def generate_presigned_url(
            self,
            object_key: str,
            expiration: int = 3600,
            http_method: str = 'GET'
    ) -> Dict[str, Any]:
        if self.use_local_fallback:
            self.logger.warning("로컬 폴백 모드에서는 Presigned URL을 생성할 수 없습니다.")
            return {"error": "Cannot generate presigned URL in local fallback mode."}

        if not self.s3_client:
            self.logger.error("StorageClient(S3)가 초기화되지 않아 Presigned URL 생성을 수행할 수 없습니다.")
            return {"error": "StorageClient S3 client is not initialized."}

        try:
            self.logger.debug(
                f"Presigned URL 생성 요청 (boto3/executor): bucket='{self.bucket_name}', key='{object_key}', expiration={expiration}s, method='{http_method}'")
            loop = asyncio.get_running_loop()

            client_method_map = {
                'GET': 'get_object',
                'PUT': 'put_object',
                # 필요한 경우 'DELETE': 'delete_object' 등 추가
            }
            client_method = client_method_map.get(http_method.upper())

            if not client_method:
                return {
                    "error": f"Unsupported HTTP method for presigned URL: {http_method}. Supported methods: {list(client_method_map.keys())}"}

            params = {'Bucket': self.bucket_name, 'Key': object_key}
            # PUT의 경우 ContentType 등을 Params에 추가할 수도 있지만, 보통 클라이언트가 요청 시 헤더에 포함
            # if http_method.upper() == 'PUT' and content_type_for_put_url:
            #    params['ContentType'] = content_type_for_put_url

            presign_func = partial(
                self.s3_client.generate_presigned_url,
                ClientMethod=client_method,
                Params=params,
                ExpiresIn=expiration,
                HttpMethod=http_method.upper()  # Boto3는 HttpMethod 파라미터도 대문자를 기대할 수 있음
            )
            url = await loop.run_in_executor(None, presign_func)
            self.logger.info(f"Presigned URL 생성 성공: key='{object_key}' ({http_method})")
            return {"presigned_url": url}
        except (ClientError, BotoCoreError) as e:
            self.logger.error(f"Presigned URL 생성 중 AWS 오류 발생: key='{object_key}', error={e}", exc_info=True)
            return {"error": f"AWS S3 Presigned URL Generation Error: {e}"}
        except Exception as e:
            self.logger.error(f"Presigned URL 생성 중 예상치 못한 오류 발생: key='{object_key}', error={e}", exc_info=True)
            return {"error": f"Unexpected error during Presigned URL generation: {e}"}

    async def _upload_file_local(self, file_path: str, object_key: Optional[str], prefix: str) -> Dict[str, Any]:
        if not self.local_storage_path:
            self.logger.error("로컬 폴백: 로컬 스토리지 경로(LOCAL_STORAGE_PATH)가 설정되지 않았습니다.")
            return {"error": "Local storage path is not configured."}

        if not AIOFILES_AVAILABLE:
            self.logger.error("로컬 폴백: aiofiles 라이브러리가 없어 비동기 로컬 파일 저장이 불가능합니다.")
            return {"error": "aiofiles library is not available for local fallback."}

        try:
            if not object_key:
                filename = os.path.basename(file_path)
                name, ext = os.path.splitext(filename)
                unique_id = uuid.uuid4().hex
                local_key = os.path.join(prefix.strip('/'), f"{name}_{unique_id}{ext}")
            else:
                local_key = object_key

            target_dir = os.path.join(self.local_storage_path, os.path.dirname(local_key))
            # 로컬 폴백 디렉토리 생성 시도, 실패 시 에러 반환
            try:
                os.makedirs(target_dir, exist_ok=True)
            except OSError as e:
                self.logger.error(f"로컬 폴백: 대상 디렉토리 생성 실패 '{target_dir}': {e}", exc_info=True)
                return {"error": f"Failed to create local fallback directory: {e}"}

            target_path = os.path.join(self.local_storage_path, local_key)

            async with aiofiles.open(file_path, mode='rb') as src:
                async with aiofiles.open(target_path, mode='wb') as dest:
                    while True:
                        chunk = await src.read(1024 * 1024)
                        if not chunk: break
                        await dest.write(chunk)

            self.logger.info(f"로컬 폴백: 파일 저장 성공 '{file_path}' -> '{target_path}'")
            local_uri = f"file://{os.path.abspath(target_path)}"
            return {
                "s3_uri": local_uri,
                "object_key": local_key,
                "content_type": mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
            }
        except Exception as e:
            self.logger.error(f"로컬 파일 저장 실패: {e}", exc_info=True)
            return {"error": f"Local file saving failed: {e}"}


# --- main 테스트 함수 ---
async def main():

    from app.config.settings import Settings
    settings = Settings()
    """로컬 이미지를 S3에 업로드하는 테스트 함수"""
    print("--- StorageService S3 Upload Test ---")

    # 테스트용 설정값 (실제 키는 환경 변수에서 로드)
    # 이 테스트를 실행하기 전에 아래 환경 변수를 설정해야 합니다:
    # TEST_S3_BUCKET_NAME, TEST_AWS_REGION
    # 선택적으로 명시적 키 테스트를 위해: TEST_AWS_ACCESS_KEY_ID, TEST_AWS_SECRET_ACCESS_KEY
    s3_bucket = settings.S3_BUCKET_NAME
    aws_region = settings.AWS_REGION
    access_key = settings.ACCESS_KEY  # 없으면 None
    secret_key = settings.SECRET_KEY  # 없으면 None

    if not s3_bucket or not aws_region:
        print("테스트를 위해 TEST_S3_BUCKET_NAME 와 TEST_AWS_REGION 환경 변수를 설정해야 합니다.")
        print("예: export TEST_S3_BUCKET_NAME=your-s3-bucket-name")
        print("    export TEST_AWS_REGION=your-aws-region")
        if access_key:
            print("    export TEST_AWS_ACCESS_KEY_ID=your_access_key_id (선택 사항)")
            print("    export TEST_AWS_SECRET_ACCESS_KEY=your_secret_access_key (선택 사항)")
        return

    # 테스트용 로컬 폴백 디렉토리 생성
    # os.makedirs(settings.LOCAL_STORAGE_PATH, exist_ok=True)

    # StorageService 인스턴스 생성 (테스트 설정 주입)
    storage_service = StorageService()

    # 테스트용 더미 이미지 파일 생성 (Pillow 사용)
    dummy_image_path = settings.LOCAL_STORAGE_PATH+"/generated_ip_adapter_test.png"
    print(dummy_image_path)
    # # 파일 업로드 테스트
    if os.path.exists(dummy_image_path):
        upload_prefix = "main_uploads/"
        custom_object_key = f"{upload_prefix}custom_name_image.png"  # 사용자 정의 키 사용 테스트

        print(f"\n1. 자동 생성된 object_key로 업로드 테스트 (ACL 기본값)...")
        result1 = await storage_service.upload_file(
            file_path=dummy_image_path,
            prefix=upload_prefix
        )
        print(f"업로드 결과 1: {result1}")

    else:
        print(f"테스트 파일 '{dummy_image_path}' 생성 실패로 업로드 테스트를 진행할 수 없습니다.")

    print("\n--- 테스트 완료 ---")


if __name__ == "__main__":
    asyncio.run(main())
