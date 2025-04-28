# app/config/settings.py

from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    서비스 전역 설정 관리 (환경 변수 기반 로딩)
    """
    llm_server_url: str
    image_server_url: str
    s3_bucket_name: str
    aws_access_key_id: str
    aws_secret_access_key: str

    redis_url: str = "redis://localhost:6379/0"
    langsmith_project_name: str = "newsom-project"

    class Config:
        env_file = ".env"

settings = Settings()
