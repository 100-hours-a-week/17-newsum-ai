# app/config/settings.py
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Optional

# .env 파일 로드 (프로젝트 루트에 .env 파일이 있다고 가정)
# 또는 애플리케이션 시작 시점에서 한 번만 로드
load_dotenv()

class Settings:

    # LLM API
    LLM_API_ENDPOINT: Optional[str] = os.getenv("LLM_API_ENDPOINT", "http://localhost:8000/v1/chat/completions") # 기본값 설정 예시
    LLM_API_MODEL: Optional[str] = os.getenv("LLM_API_MODEL", "./merged_model") # 기본 모델 예시
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY") # 필요시
    LLM_API_TIMEOUT: int = int(os.getenv("LLM_API_TIMEOUT", "60")) # 기본 타임아웃 60초
    LLM_API_RETRIES: int = int(os.getenv("LLM_API_RETRIES", "2")) # 기본 재시도 2회
    LLM_API_RETRY_DELAY: int = int(os.getenv("LLM_API_RETRY_DELAY", "1")) # 기본 재시도 간격 1초

    # Image API (이후 사용)
    IMAGE_SERVER_URL: Optional[str] = os.getenv("IMAGE_SERVER_URL")
    # IMAGE_API_ENDPOINT: Optional[str] = os.getenv("IMAGE_API_ENDPOINT")
    # IMAGE_API_KEY: Optional[str] = os.getenv("IMAGE_API_KEY")

    # Google
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID: Optional[str] = os.getenv("GOOGLE_CSE_ID")

    # Tavily
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")

    # Bing Search
    BING_SUBSCRIPTION_KEY: Optional[str] = os.getenv("BING_SUBSCRIPTION_KEY")
    BING_SEARCH_URL: Optional[str] = os.getenv("BING_SEARCH_URL", "https://api.bing.microsoft.com/v7.0/search")
    
    # YouTube API
    YOUTUBE_API_KEY: Optional[str] = os.getenv("YOUTUBE_API_KEY")
    YOUTUBE_MAX_VIDEOS: int = int(os.getenv("YOUTUBE_MAX_VIDEOS", "3"))
    YOUTUBE_MAX_COMMENTS: int = int(os.getenv("YOUTUBE_MAX_COMMENTS", "50"))

    # 검색 기본값
    DEFAULT_SEARCH_ENGINE: str = os.getenv("DEFAULT_SEARCH_ENGINE", "google")
    DEFAULT_NUM_RESULTS: int = int(os.getenv("DEFAULT_NUM_RESULTS", "5"))
    # Tavily 용 시간 범위 (None, 'day', 'week', 'month', 'year')
    SEARCH_TIME_RANGE: Optional[str] = os.getenv("SEARCH_TIME_RANGE")
    # Bing 용 마켓 코드
    SEARCH_MARKET: str = os.getenv("SEARCH_MARKET", "en-US")
    # 특정 도메인 포함/제외 (Tavily 등 지원) - 필요시 .env 에서 리스트 형식 로드
    SEARCH_INCLUDE_DOMAINS: Optional[List[str]] = os.getenv("SEARCH_INCLUDE_DOMAINS",
                                                            None)  # 예: "example.com,anothersite.net"
    SEARCH_EXCLUDE_DOMAINS: Optional[List[str]] = os.getenv("SEARCH_EXCLUDE_DOMAINS", None)
    
    # 에이전트 결과 저장 설정
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results'))
    SAVE_AGENT_RESULTS: bool = os.getenv("SAVE_AGENT_RESULTS", "True").lower() in ["true", "1", "yes"]
    SAVE_AGENT_INPUTS: bool = os.getenv("SAVE_AGENT_INPUTS", "False").lower() in ["true", "1", "yes"]
    SAVE_DEBUG_INFO: bool = os.getenv("SAVE_DEBUG_INFO", "True").lower() in ["true", "1", "yes"]

    # --- Helper to parse list from env var ---
    @staticmethod
    def _parse_list(env_var: Optional[str]) -> Optional[List[str]]:
        if env_var:
            return [item.strip() for item in env_var.split(',')]
        return None

    def __init__(self):
        # 문자열로 로드된 도메인 리스트를 실제 리스트로 변환
        self.SEARCH_INCLUDE_DOMAINS = self._parse_list(os.getenv("SEARCH_INCLUDE_DOMAINS"))
        self.SEARCH_EXCLUDE_DOMAINS = self._parse_list(os.getenv("SEARCH_EXCLUDE_DOMAINS"))
        
        # 결과 디렉토리 생성
        os.makedirs(self.RESULTS_DIR, exist_ok=True)


# 설정 객체 생성 (싱글톤처럼 사용 가능)
settings = Settings()

# .env 파일 예시 (프로젝트 루트에 생성)
"""
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
GOOGLE_CSE_ID=YOUR_GOOGLE_CSE_ID_HERE
LLM_API_ENDPOINT=http://YOUR_LLM_API_SERVER/generate # 실제 LLM API 주소
# LLM_API_KEY=YOUR_LLM_API_KEY_IF_NEEDED
YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY # YouTube API 키
SAVE_AGENT_RESULTS=True # 에이전트 결과 저장 활성화
SAVE_AGENT_INPUTS=False # 입력 상태 저장 비활성화
SAVE_DEBUG_INFO=True # 디버그 정보 저장 활성화
RESULTS_DIR=/path/to/results # 결과 저장 디렉토리 커스텀 (선택 사항)
"""