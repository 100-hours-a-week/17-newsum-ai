# src/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() # .env 파일 로드

BASE_DIR = Path(__file__).resolve().parent.parent

# --- 외부 API 엔드포인트 설정 ---
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://your-llm-service.trycloudflare.com/v1") # 예시: vLLM Cloudflare 주소
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY") # vLLM 기본값 또는 실제 키
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Llama-3.2-3B-Instruct") # API에서 사용할 모델 식별자

SD_API_BASE_URL = os.getenv("SD_API_BASE_URL", "https://your-sd-service.trycloudflare.com") # 예시: SD FastAPI Cloudflare 주소
# SD_API_KEY = os.getenv("SD_API_KEY") # Stable Diffusion API에 인증이 필요하면 설정

# --- 뉴스 수집 설정 (동일) ---
NEWS_SOURCES = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    # 여기에 다른 RSS 주소 추가 가능
]

# --- 저장 경로 (동일) ---
OUTPUT_DIR = BASE_DIR / "../data"
FINAL_JSON_DIR = OUTPUT_DIR / "cache/final"
IMAGE_SAVE_DIR = OUTPUT_DIR / "static/images"
FALLBACK_IMAGE_PATH = str(OUTPUT_DIR / "static/images/fallback/laughing_emoji.png") # 미리 준비

# --- 생성 기본값 (API 호출 시 사용) ---
SD_DEFAULT_STEPS = 28
SD_DEFAULT_CFG_SCALE = 7.0
SD_DEFAULT_WIDTH = 512 # 기본값을 512로 변경 (API 서버와 맞출 수 있음)
SD_DEFAULT_HEIGHT = 512

# --- LangGraph 설정 (동일) ---
MAX_RETRIES_ANALYZE = 2
MAX_RETRIES_HUMORIZE = 1
# MAX_RETRIES_RENDER = 1 # API 호출은 재시도 정책 다르게 가져갈 수 있음

# 디렉터리 생성 (동일)
FINAL_JSON_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
(IMAGE_SAVE_DIR / "fallback").mkdir(exist_ok=True)

# --- 기타 (동일) ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")