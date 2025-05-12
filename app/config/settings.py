# ai/app/config/settings.py
from pathlib import Path # <<< Path 객체 사용 위해 임포트
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AnyHttpUrl
from typing import Optional, List

# --- 프로젝트 루트 경로 계산 ---
# settings.py 파일의 위치 (ai/app/config/settings.py) 를 기준으로 상위 2단계 디렉토리 (ai/) 를 프로젝트 루트로 가정합니다.
# 다른 기준 (예: 특정 파일 존재 여부)으로 프로젝트 루트를 찾을 수도 있습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- .env 파일 경로 설정 ---
# 프로젝트 루트를 기준으로 .env 파일 경로 지정
ENV_FILE_PATH = PROJECT_ROOT / '.env'


class Settings(BaseSettings):
    """애플리케이션 설정 모델"""
    # .env 파일 로드 설정, 환경 변수 이름 대소문자 구분 안 함
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH, extra='ignore', case_sensitive=False)

    # --- 애플리케이션 기본 정보 ---
    APP_NAME: str = "NewSum AI Service"
    APP_VERSION: str = "0.1.0"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8090 # <<< 실행 로그에서 사용된 포트로 수정
    APP_RELOAD: bool = True

    # --- 로깅 설정 ---
    LOG_LEVEL: str = "INFO"
    # <<< 수정: validation_alias 제거 (필드명과 동일하여 불필요), 기본값 명확화
    LOG_CONFIG_PATH: Optional[str] = Field(str(PROJECT_ROOT / "logging_config.yaml"))


    # --- 데이터베이스 (Redis 예시) ---
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = Field(None) 

    # --- LLM 서비스 ---
    LLM_API_ENDPOINT: Optional[AnyHttpUrl] = Field(None) 
    LLM_API_TIMEOUT: int = 60
    LLM_API_RETRIES: int = 3
    DEFAULT_LLM_MODEL: str = "default-model-name"

    # --- 번역 서비스 (Papago 예시) ---
    NAVER_CLIENT_ID: Optional[str] = Field(None) 
    NAVER_CLIENT_SECRET: Optional[str] = Field(None) 

    # --- 스팸 탐지 서비스 ---
    SPAM_KEYWORDS: List[str] = Field(default_factory=lambda: ["광고", "홍보", "클릭"]) 
    SPAM_MAX_URL_COUNT: int = 2
    SPAM_MAX_UPPERCASE_RATIO: float = 0.7

    # --- 스토리지 서비스 (S3 예시) ---
    S3_BUCKET_NAME: Optional[str] = Field(None) 
    AWS_REGION: Optional[str] = Field(None) 
    ACCESS_KEY: Optional[str] = Field(None) 
    SECRET_KEY: Optional[str] = Field(None) 
    LOCAL_STORAGE_PATH: str = Field("local_storage_path") 
    S3_STORAGE_PATH: str = Field("s3_storage_path") 

    # --- 이미지 생성 서비스 ---
    IMAGE_SERVER_URL: Optional[AnyHttpUrl] = Field(None) 
    IMAGE_SERVER_API_TOKEN: Optional[str] = Field(None) 
    IMAGE_STORAGE_PATH: str = Field("generated_images") 
    LLM_MAX_TOKENS_SCENARIO: int = 2000
    IMAGE_DEFAULT_NEGATIVE_PROMPT: str = Field("(worst quality, low quality, normal quality:1.2), deformed, blurry, text, signature")

    # --- 검색 도구 (Google 예시) ---
    GOOGLE_API_KEY: Optional[str] = Field(None) 
    GOOGLE_CSE_ID: Optional[str] = Field(None) 
    TARGET_COMMUNITY_DOMAINS: List[str] = Field(default_factory=list) 
    EXTERNAL_NOTIFICATION_API_URL: Optional[str] = Field(None) 
    EXTERNAL_API_TIMEOUT_SECONDS: int = 30

    # --- 도구 공통 설정 ---
    TOOL_HTTP_TIMEOUT: int = 30
    TOOL_RETRY_ATTEMPTS: int = 2
    TOOL_RETRY_WAIT_MIN: int = 1
    TOOL_RETRY_WAIT_MAX: int = 3

    # --- LangSmith ---
    LANGCHAIN_TRACING_V2: Optional[str] = Field("true") 
    LANGCHAIN_ENDPOINT: Optional[AnyHttpUrl] = Field("https://api.smith.langchain.com") 
    LANGCHAIN_API_KEY: Optional[str] = Field(None) 
    LANGCHAIN_PROJECT: Optional[str] = Field("NewSum-Project") 

    SCENARIO_TARGET_SCENES_COUNT: int = 4
    IMAGE_MAX_PARALLEL_TASKS: int = 1
    S3_MAX_PARALLEL_UPLOADS: int = 5

    IMAGE_DEFAULT_STYLE_PROMPT: str = "vibrant webtoon art style, dynamic camera angle"

    TRANSLATION_BATCH_SIZE: int = 60
    TRANSLATION_BATCH_CHAR_LIMIT: int = 25000
    ORIGINAL_REPORT_LANGUAGE: str = "en"
    N06_REPORT_TRANSLATION_TARGET_LANG: str = "ko"

# 설정 객체 인스턴스 생성
settings = Settings()

# --- 설정 로드 확인용 (디버깅 시 사용) ---
# print("Settings loaded:")
# print(f"  LOG_CONFIG_PATH: {getattr(settings, 'LOG_CONFIG_PATH', 'NOT FOUND')}")
# print(f"  APP_PORT: {settings.APP_PORT}")