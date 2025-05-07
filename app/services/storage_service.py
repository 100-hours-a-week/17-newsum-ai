# ai/app/services/storage_service.py

import os
import uuid
import mimetypes
import asyncio # run_in_executor 사용 위해 추가
from typing import Dict, Any, Optional
from functools import partial # run_in_executor 에 인자 전달 위해 추가

from app.config.settings import Settings
from app.utils.logger import get_logger

settings = Settings()
# boto3 및 관련 예외 import
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger_temp = get_logger("StorageClient_Init")
    logger_temp.warning("boto3 라이브러리가 설치되지 않았습니다. StorageClient 기능이 비활성화됩니다.")
    # Define dummy classes if not available
    class ClientError(Exception): pass
    class BotoCoreError(Exception): pass

# aiofiles는 이제 upload_file에서는 직접 사용하지 않지만,
# 다른 곳에서 비동기 파일 처리를 위해 남겨둘 수 있음.
# 만약 다른 곳에서도 사용하지 않는다면 제거 가능.
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False


logger = get_logger("StorageClientV2_Boto3") # 로거 이름 변경

class StorageService:
    """
    동기 boto3를 사용하여 비동기 환경에서 AWS S3 스토리지와 상호작용하는 클라이언트.
    파일 업로드 및 Presigned URL 생성을 지원합니다. run_in_executor 사용.
    AWS 자격 증명은 표준 검색 경로(환경 변수, IAM 역할 등)를 통해 자동으로 로드됩니다.
    """
    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None,
        logger_name: str = "StorageClientV2_Boto3" # 로거 이름 업데이트
    ):
        self.logger = get_logger(logger_name)
        self.s3_client = None
        self.use_local_fallback = False # 로컬 폴백 사용 여부 플래그
        self.local_storage_path = settings.LOCAL_STORAGE_PATH # 설정에서 로컬 경로 읽기

        if not BOTO3_AVAILABLE:
            self.logger.error("필수 라이브러리(boto3)가 없어 StorageClient를 초기화할 수 없습니다.")
            raise ImportError("boto3 must be installed to use StorageClient.")

        # 설정 값 로드
        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME
        self.region_name = region_name or settings.AWS_REGION

        # S3 클라이언트 생성 시도 (boto3 사용)
        try:
            if not self.bucket_name:
                 raise ValueError("S3 bucket name must be provided for S3 mode.")

            # 동기 boto3 클라이언트 생성
            self.s3_client = boto3.client(
                's3',
                region_name=self.region_name
                # 필요 시 추가 설정 전달 가능: config=Config(...)
            )
            # 간단한 테스트 호출 (선택 사항) - 초기화 시 자격 증명 확인 등
            # self.s3_client.list_buckets() # 예시: 버킷 리스트 요청 (권한 필요)
            self.logger.info(f"StorageClient (boto3) S3 모드로 초기화 완료. Bucket: {self.bucket_name}, Region: {self.region_name or 'default'}")

        except (ClientError, BotoCoreError) as e:
            self.logger.warning(f"boto3 S3 클라이언트 생성 실패 (AWS 오류): {e}. 로컬 스토리지 폴백 모드로 전환합니다 (경로: {self.local_storage_path}).", exc_info=True)
            self.s3_client = None
            self.use_local_fallback = True
            if not self.local_storage_path:
                self.logger.error("S3 초기화 실패 및 로컬 스토리지 경로(LOCAL_STORAGE_PATH)가 설정되지 않아 StorageClient 사용 불가.")
                raise ValueError("S3 initialization failed and LOCAL_STORAGE_PATH is not configured.")
        except Exception as e:
             self.logger.warning(f"boto3 S3 클라이언트 생성 중 예상치 못한 오류: {e}. 로컬 스토리지 폴백 모드로 전환합니다 (경로: {self.local_storage_path}).", exc_info=True)
             self.s3_client = None
             self.use_local_fallback = True
             if not self.local_storage_path:
                 self.logger.error("S3 초기화 실패 및 로컬 스토리지 경로(LOCAL_STORAGE_PATH)가 설정되지 않아 StorageClient 사용 불가.")
                 raise ValueError("S3 initialization failed and LOCAL_STORAGE_PATH is not configured.")

    async def upload_file(
        self,
        file_path: str,
        object_key: Optional[str] = None,
        prefix: str = "uploads/", # 기본 업로드 경로 접두사
        content_type: Optional[str] = None,
        acl: Optional[str] = None # ACL 설정 추가 (선택 사항)
    ) -> Dict[str, Any]:
        """
        로컬 파일을 S3 버킷에 비동기적으로 업로드합니다 (run_in_executor 사용).

        Args:
            file_path (str): 업로드할 로컬 파일 경로.
            object_key (Optional[str]): S3에 저장될 객체 키 (전체 경로). 지정하지 않으면 자동 생성됩니다.
            prefix (str): object_key 자동 생성 시 사용될 접두사. 기본값: "uploads/".
            content_type (Optional[str]): 파일의 MIME 타입. 지정하지 않으면 추측합니다.
            acl (Optional[str]): 적용할 객체 ACL (예: 'public-read'). 기본값은 버킷 설정 따름.

        Returns:
            Dict[str, Any]: 성공 시 {"s3_uri": str, "object_key": str}, 실패 시 {"error": str}
        """
        # 로컬 폴백 모드 처리 (선택적 구현)
        if self.use_local_fallback:
             return await self._upload_file_local(file_path, object_key, prefix)

        # S3 클라이언트 확인
        if not self.s3_client:
            self.logger.error("StorageClient(S3)가 초기화되지 않아 upload_file 작업을 수행할 수 없습니다.")
            return {"error": "StorageClient S3 client is not initialized."}

        if not os.path.exists(file_path):
            self.logger.error(f"업로드할 파일을 찾을 수 없습니다: {file_path}")
            return {"error": f"File not found at path: {file_path}"}

        # 객체 키 생성 (지정되지 않은 경우)
        if not object_key:
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            unique_id = uuid.uuid4().hex
            object_key = f"{prefix.strip('/')}/{name}_{unique_id}{ext}"

        # ContentType 추측 (지정되지 않은 경우)
        if not content_type:
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = 'application/octet-stream' # 기본값

        try:
            self.logger.info(f"S3 업로드 시작 (boto3/executor): '{file_path}' -> 's3://{self.bucket_name}/{object_key}' (ContentType: {content_type})")
            loop = asyncio.get_running_loop()

            # upload_file 메서드에 전달할 추가 인자 구성
            extra_args = {'ContentType': content_type}
            if acl:
                extra_args['ACL'] = acl

            # functools.partial 사용하여 upload_file 함수와 인자 준비
            # upload_file(Filename, Bucket, Key, ExtraArgs=None, Callback=None, Config=None)
            upload_func = partial(
                self.s3_client.upload_file,
                Filename=file_path,
                Bucket=self.bucket_name,
                Key=object_key,
                ExtraArgs=extra_args
            )

            # 동기 함수인 upload_file을 executor에서 실행
            await loop.run_in_executor(None, upload_func)

            s3_uri = f"s3://{self.bucket_name}/{object_key}"
            # 퍼블릭 URL 생성 (ACL='public-read' 설정 시 또는 버킷 정책에 따라)
            # object_url = f"https://{self.bucket_name}.s3.{self.s3_client.meta.region_name}.amazonaws.com/{object_key}"
            self.logger.info(f"S3 업로드 성공: {s3_uri}")
            return {
                "s3_uri": s3_uri,
                "object_key": object_key,
                "content_type": content_type
                # "public_url": object_url # 필요시 반환
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
        expiration: int = 3600, # URL 유효 시간(초), 기본값 1시간
        http_method: str = 'GET' # Presigned URL 용도 (GET, PUT 등)
    ) -> Dict[str, Any]:
        """
        S3 객체에 접근하기 위한 Presigned URL을 생성합니다 (run_in_executor 사용).

        Args:
            object_key (str): URL을 생성할 S3 객체 키.
            expiration (int): URL의 유효 시간(초). 기본값: 3600.
            http_method (str): URL로 허용할 HTTP 메소드 ('GET', 'PUT' 등). 기본값 'GET'.

        Returns:
            Dict[str, Any]: 성공 시 {"presigned_url": str}, 실패 시 {"error": str}
        """
        if self.use_local_fallback:
             # 로컬 모드에서는 Presigned URL 의미 없음
             self.logger.warning("로컬 폴백 모드에서는 Presigned URL을 생성할 수 없습니다.")
             return {"error": "Cannot generate presigned URL in local fallback mode."}

        if not self.s3_client:
            self.logger.error("StorageClient(S3)가 초기화되지 않아 Presigned URL 생성을 수행할 수 없습니다.")
            return {"error": "StorageClient S3 client is not initialized."}

        try:
            self.logger.debug(f"Presigned URL 생성 요청 (boto3/executor): bucket='{self.bucket_name}', key='{object_key}', expiration={expiration}s, method='{http_method}'")
            loop = asyncio.get_running_loop()

            # generate_presigned_url 함수와 인자 준비
            # generate_presigned_url(ClientMethod, Params=None, ExpiresIn=3600, HttpMethod=None)
            client_method = 'get_object' if http_method.upper() == 'GET' else \
                            'put_object' if http_method.upper() == 'PUT' else None # 다른 메소드 지원 추가 가능
            if not client_method:
                return {"error": f"Unsupported HTTP method for presigned URL: {http_method}"}

            params = {'Bucket': self.bucket_name, 'Key': object_key}
            # PUT 요청 시 추가 파라미터 필요할 수 있음 (예: ContentType)

            presign_func = partial(
                self.s3_client.generate_presigned_url,
                ClientMethod=client_method,
                Params=params,
                ExpiresIn=expiration,
                HttpMethod=http_method.upper()
            )

            # 동기 함수인 generate_presigned_url을 executor에서 실행
            url = await loop.run_in_executor(None, presign_func)

            self.logger.info(f"Presigned URL 생성 성공: key='{object_key}' ({http_method})")
            return {"presigned_url": url}

        except (ClientError, BotoCoreError) as e:
            self.logger.error(f"Presigned URL 생성 중 AWS 오류 발생: key='{object_key}', error={e}", exc_info=True)
            return {"error": f"AWS S3 Presigned URL Generation Error: {e}"}
        except Exception as e:
            self.logger.error(f"Presigned URL 생성 중 예상치 못한 오류 발생: key='{object_key}', error={e}", exc_info=True)
            return {"error": f"Unexpected error during Presigned URL generation: {e}"}

    # --- 로컬 폴백 메서드 (선택적 구현) ---
    async def _upload_file_local(self, file_path: str, object_key: Optional[str], prefix: str) -> Dict[str, Any]:
         """로컬 파일 시스템에 파일을 저장하는 폴백 메서드"""
         if not self.local_storage_path:
              return {"error": "Local storage path is not configured."}

         try:
              if not object_key:
                   filename = os.path.basename(file_path)
                   name, ext = os.path.splitext(filename)
                   unique_id = uuid.uuid4().hex
                   # 로컬 경로는 OS에 맞게 처리
                   local_key = os.path.join(prefix.strip('/'), f"{name}_{unique_id}{ext}")
              else:
                   local_key = object_key

              # 대상 디렉토리 생성
              target_dir = os.path.join(self.local_storage_path, os.path.dirname(local_key))
              os.makedirs(target_dir, exist_ok=True)
              target_path = os.path.join(self.local_storage_path, local_key)

              # aiofiles로 비동기 복사
              async with aiofiles.open(file_path, mode='rb') as src:
                   async with aiofiles.open(target_path, mode='wb') as dest:
                        while True:
                             chunk = await src.read(1024 * 1024) # 1MB씩 읽기
                             if not chunk: break
                             await dest.write(chunk)

              self.logger.info(f"로컬 폴백: 파일 저장 성공 '{file_path}' -> '{target_path}'")
              # 로컬 파일 URI 반환 (file:// 스키마)
              local_uri = f"file://{os.path.abspath(target_path)}"
              return {
                   "s3_uri": local_uri, # 로컬 파일 URI
                   "object_key": local_key, # 로컬 경로 기준 키
                   "content_type": mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
              }
         except Exception as e:
              self.logger.error(f"로컬 파일 저장 실패: {e}", exc_info=True)
              return {"error": f"Local file saving failed: {e}"}

    # boto3 클라이언트는 명시적인 close 불필요
    # async def close(self):
    #     pass