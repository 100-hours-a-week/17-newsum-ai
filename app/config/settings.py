# app/config/settings.py
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional

# .env 파일 로드 (프로젝트 루트에 .env 파일이 있다고 가정)
# 또는 애플리케이션 시작 시점에서 한 번만 로드
load_dotenv()

class Settings:
    # Google Search
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID: Optional[str] = os.getenv("GOOGLE_CSE_ID")

    # LLM API
    LLM_API_ENDPOINT: Optional[str] = os.getenv("LLM_API_ENDPOINT", "http://localhost:8000/v1/llm/generate") # 기본값 설정 예시
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY") # 필요시
    LLM_API_TIMEOUT: int = int(os.getenv("LLM_API_TIMEOUT", "60")) # 기본 타임아웃 60초
    LLM_API_RETRIES: int = int(os.getenv("LLM_API_RETRIES", "2")) # 기본 재시도 2회
    LLM_API_RETRY_DELAY: int = int(os.getenv("LLM_API_RETRY_DELAY", "1")) # 기본 재시도 간격 1초

    # Image API (이후 사용)
    # IMAGE_API_ENDPOINT: Optional[str] = os.getenv("IMAGE_API_ENDPOINT")
    # IMAGE_API_KEY: Optional[str] = os.getenv("IMAGE_API_KEY")

    # 기타 설정...

# 설정 객체 생성 (싱글톤처럼 사용 가능)
settings = Settings()

# .env 파일 예시 (프로젝트 루트에 생성)
"""
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
GOOGLE_CSE_ID=YOUR_GOOGLE_CSE_ID_HERE
LLM_API_ENDPOINT=http://YOUR_LLM_API_SERVER/generate # 실제 LLM API 주소
# LLM_API_KEY=YOUR_LLM_API_KEY_IF_NEEDED
"""