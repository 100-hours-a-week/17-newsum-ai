# app/config/settings.py (Reorganized Version)

import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

# --- .env 파일 로드 ---
# 실제 앱 실행 시점(예: main.py)에서 load_dotenv() 호출 권장
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded environment variables from: {dotenv_path}")
else:
    print(f".env file not found at {dotenv_path}. Using environment variables or defaults.")

# --- 프로젝트 루트 경로 계산 ---
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 환경 변수 파싱 헬퍼 함수 ---
def _parse_list_helper(env_var_value: Optional[str]) -> List[str]:
    """환경 변수에서 쉼표로 구분된 문자열을 리스트로 파싱합니다."""
    if env_var_value:
        return [item.strip() for item in env_var_value.split(',') if item.strip()]
    return []

def _parse_json_helper(env_var_value: Optional[str], default: Any = None) -> Any:
    """환경 변수에서 JSON 문자열을 파싱합니다."""
    _default = default if default is not None else {}
    if env_var_value:
        try:
            return json.loads(env_var_value)
        except json.JSONDecodeError as e:
             print(f"Warning: Failed to parse JSON from env var. Value: '{env_var_value[:50]}...'. Error: {e}. Using default: {_default}")
             return _default
    return _default

class Settings:
    """
    (Reorganized) 애플리케이션 설정을 관리하는 중앙 클래스.
    필수 설정과 선택적/튜닝 가능 설정으로 구분하여 관리합니다.
    """

    # ======================================================================
    # 필수 설정 (Essential Configuration)
    # ======================================================================
    # 아래 값들은 .env 파일 또는 환경 변수를 통해 설정해야 할 가능성이 높습니다.
    # 주석에 명시된 조건 또는 특정 기능 활성화 시 필수적으로 필요합니다.

    # --- LLM API ---
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY") # 필수: LLM 서비스 인증에 필요합니다.
    LLM_API_ENDPOINT: Optional[str] = os.getenv("LLM_API_ENDPOINT") # 선택적: 자체 호스팅/특정 엔드포인트 사용 시 설정합니다.

    # --- Image Generation API ---
    # IMAGE_SERVER_URL 설정 시 API 토큰이 필요할 수 있습니다.
    IMAGE_SERVER_URL: Optional[str] = os.getenv("IMAGE_SERVER_URL") # 필수: 이미지 생성 서버 주소 (외부 서비스 사용 시).
    IMAGE_SERVER_API_TOKEN: Optional[str] = os.getenv("IMAGE_SERVER_API_TOKEN") # 필수: 이미지 생성 서버 인증 토큰.

    # --- Translation API (Naver Papago) ---
    # TRANSLATION_ENABLED=True 일 경우 아래 값들이 필수입니다.
    NAVER_CLIENT_ID: Optional[str] = os.getenv("NAVER_CLIENT_ID") # 필수 (번역 사용 시): Naver Cloud Platform Client ID.
    NAVER_CLIENT_SECRET: Optional[str] = os.getenv("NAVER_CLIENT_SECRET") # 필수 (번역 사용 시): Naver Cloud Platform Client Secret.

    # --- Search / Trend Tool APIs ---
    # 사용하는 각 검색/소셜 미디어 도구에 대한 API 키/자격 증명이 필요합니다.
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")       # 필수 (Google Search 사용 시): Google API Key.
    GOOGLE_CSE_ID: Optional[str] = os.getenv("GOOGLE_CSE_ID")         # 필수 (Google Search 사용 시): Google Custom Search Engine ID.
    TWITTER_BEARER_TOKEN: Optional[str] = os.getenv("TWITTER_BEARER_TOKEN") # 필수 (Twitter 사용 시): Twitter API Bearer Token.
    REDDIT_CLIENT_ID: Optional[str] = os.getenv("REDDIT_CLIENT_ID")     # 필수 (Reddit 사용 시): Reddit App Client ID.
    REDDIT_CLIENT_SECRET: Optional[str] = os.getenv("REDDIT_CLIENT_SECRET") # 필수 (Reddit 사용 시): Reddit App Client Secret.
    YOUTUBE_API_KEY: Optional[str] = os.getenv("YOUTUBE_API_KEY")     # 필수 (Youtube 사용 시): YouTube Data API Key.
    # Reddit 스크립트 인증 방식 사용 시 추가 필요
    REDDIT_USERNAME: Optional[str] = os.getenv("REDDIT_USERNAME") # 선택적: Reddit 스크립트 인증용 사용자 이름.
    REDDIT_PASSWORD: Optional[str] = os.getenv("REDDIT_PASSWORD") # 선택적: Reddit 스크립트 인증용 비밀번호.

    # --- Redis Cache/DB ---
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))  # 기본 캐시/DB 번호

    # --- LangGraph Redis Checkpointer URL ---
    # 환경 변수 REDIS_URL 있으면 사용하고, 없으면 기본값 생성
    # 체크포인트용 DB는 캐시(DB 0)와 분리하기 위해 다른 번호(예: 1) 권장
    _default_checkpoint_db = REDIS_DB
    _redis_checkpoint_host = os.getenv("REDIS_HOST", "localhost")  # 체크포인트용 호스트가 다를 수 있다면 별도 변수 사용
    _redis_checkpoint_port = int(os.getenv("REDIS_PORT", "6379"))  # 체크포인트용 포트가 다를 수 있다면 별도 변수 사용
    _redis_checkpoint_password = os.getenv("REDIS_PASSWORD")  # 동일 비밀번호 가정

    _redis_checkpoint_base = f"redis://{_redis_checkpoint_host}:{_redis_checkpoint_port}/{_default_checkpoint_db}"
    if _redis_checkpoint_password:
        _redis_checkpoint_base = f"redis://:{_redis_checkpoint_password}@{_redis_checkpoint_host}:{_redis_checkpoint_port}/{_default_checkpoint_db}"

    REDIS_URL: str = os.getenv("REDIS_URL", _redis_checkpoint_base)

    # --- LangSmith (Optional Observability) ---
    # LangSmith 연동 시 필요합니다.
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY") # 필수 (LangSmith 사용 시): LangChain API Key.

    # --- AWS S3 Storage ---
    # UPLOAD_TO_S3=True 일 경우 아래 값들이 필수입니다.
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME") # 필수 (S3 업로드 사용 시): 대상 S3 버킷 이름.
    AWS_REGION: Optional[str] = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION")) # 필수 (S3 업로드 사용 시): 대상 S3 버킷의 AWS 리전.
    LOCAL_STORAGE_PATH: str = os.getenv(  # Optional[str] 대신 str 로 변경하고 기본값 지정
        "LOCAL_STORAGE_PATH",
        os.path.join(PROJECT_ROOT_DIR, 'storage', 'local_fallback')  # 기본 로컬 저장 경로
    )

    # ======================================================================
    # 코어 설정 및 기능 활성화 플래그 (Core Settings & Feature Flags)
    # ======================================================================
    # 애플리케이션의 기본적인 동작 모드나 선택적 기능의 활성화 여부를 제어합니다.

    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "ComicGenerator") # 선택적: 프로젝트 이름 (로깅 등에 사용 가능).
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ["true", "1", "yes"] # 선택적: 디버그 모드 활성화.
    TRANSLATION_ENABLED: bool = os.getenv("TRANSLATION_ENABLED", "false").lower() == "true" # 선택적: 결과물 번역 기능 활성화.
    UPLOAD_TO_S3: bool = os.getenv("UPLOAD_TO_S3", "false").lower() == "true" # 선택적: 최종 결과물 S3 업로드 활성화.

    # --- 결과 저장 관련 플래그 ---
    SAVE_AGENT_RESULTS: bool = os.getenv("SAVE_AGENT_RESULTS", "True").lower() in ["true", "1", "yes"] # 선택적: 각 에이전트/노드의 결과 저장 여부.
    SAVE_AGENT_INPUTS: bool = os.getenv("SAVE_AGENT_INPUTS", "False").lower() in ["true", "1", "yes"] # 선택적: 각 에이전트/노드의 입력 저장 여부.
    SAVE_DEBUG_INFO: bool = os.getenv("SAVE_DEBUG_INFO", "True").lower() in ["true", "1", "yes"] # 선택적: 디버깅 정보 저장 여부 (DEBUG 플래그와 연동 고려).

    DEFAULT_TEMPLATE_DIR: str = os.path.join(PROJECT_ROOT_DIR, "templates") # 예: app/templates
    DEFAULT_TEMPLATE_A_FILENAME: str = "template_a.md"
    DEFAULT_TEMPLATE_B_FILENAME: str = "template_b.md"

    # ======================================================================
    # 튜닝 가능 파라미터 및 기타 설정 (Tunable Parameters & Other Settings)
    # ======================================================================
    # 대부분 기본값이 있으며, 필요에 따라 성능/동작/결과 품질 튜닝을 위해 .env에서 수정 가능합니다.

    # --- 일반 도구/API 기본값 ---
    TOOL_RETRY_ATTEMPTS: int = int(os.getenv("TOOL_RETRY_ATTEMPTS", "3"))  # 도구 기본 재시도 횟수.
    TOOL_RETRY_WAIT_MIN: int = int(os.getenv("TOOL_RETRY_WAIT_MIN", "1"))  # 도구 재시도 최소 대기 시간(초).
    TOOL_RETRY_WAIT_MAX: int = int(os.getenv("TOOL_RETRY_WAIT_MAX", "10")) # 도구 재시도 최대 대기 시간(초).
    TOOL_HTTP_TIMEOUT: int = int(os.getenv("TOOL_HTTP_TIMEOUT", "15")) # 도구 일반 HTTP 요청 타임아웃(초).
    # API_FETCH_TIMEOUT is deprecated/merged into TOOL_HTTP_TIMEOUT or specific timeouts below.


    # --- LLM 동작 튜닝 ---
    LLM_API_TIMEOUT: int = int(os.getenv("LLM_API_TIMEOUT", "60")) # LLM API 호출 타임아웃(초).
    LLM_API_RETRIES: int = int(os.getenv("LLM_API_RETRIES", "3")) # LLM API 호출 재시도 횟수.
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini") # 기본 사용할 LLM 모델 ID.
    # 작업별 LLM Temperature (창의성 조절)
    DEFAULT_CACHE_TTL_TOPIC: int = int(os.getenv("DEFAULT_CACHE_TTL_TOPIC", "60")) # Node 02
    DEFAULT_KEYWORD_CONF_THRESHOLD: float = float(os.getenv("LLM_TEMPERATURE_ANALYSIS", "0.2")) # Node 02
    LLM_TEMPERATURE_ANALYSIS: float = float(os.getenv("LLM_TEMPERATURE_ANALYSIS", "0.2")) # Node 02
    DEFAULT_SEARCH_RESULTS: int = int(os.getenv("DEFAULT_SEARCH_RESULTS", "10")) # Node 02
    DEFAULT_TARGET_URLS_PER_KW: int = int(os.getenv("DEFAULT_TARGET_URLS_PER_KW", "4")) # Node 02
    DEFAULT_HTTP_TIMEOUT: int = int(os.getenv("DEFAULT_HTTP_TIMEOUT", "15")) # Node 02
    OPINION_COLLECTOR_CONCURRENCY: int = int(os.getenv("OPINION_COLLECTOR_CONCURRENCY", "3")) # Node 02
    TARGET_BLOG_DOMAINS: List[str] = _parse_list_helper(os.getenv("TARGET_BLOG_DOMAINS")) or ["medium.com", "dev.to", "hashnode.com", "substack.com"]


    MAX_OPINION_TEXT_LEN: int = 1000     # LLM 입력용 개별 의견 최대 길이 (MAX_ALT_TEXT_LEN 아님!)
    MAX_OPINIONS_STANCE: int = 20        # 입장 분석에 사용할 최대 의견 수
    DEFAULT_MAX_OPINIONS_SENTIMENT: int = 20 # 감성 분석에 사용할 최대 의견 수
    DEFAULT_SUMMARIZER_CONCURRENCY: int = 3 # Node 9 내부 동시 작업 수
    LLM_TEMPERATURE_EXTRACT: float = float(os.getenv("LLM_TEMPERATURE_EXTRACT", "0.1")) # Node 08
    LLM_TEMPERATURE_SUMMARIZE: float = float(os.getenv("LLM_TEMPERATURE_SUMMARIZE", "0.3")) # Node 08
    LLM_TEMPERATURE_QA_GEN: float = float(os.getenv("LLM_TEMPERATURE_QA_GEN", "0.1")) # Node 08
    LLM_TEMPERATURE_QA_VERIFY: float = float(os.getenv("LLM_TEMPERATURE_QA_VERIFY", "0.0")) # Node 08
    LLM_TEMPERATURE_STANCE: float = float(os.getenv("LLM_TEMPERATURE_STANCE", "0.1")) # Node 09
    LLM_TEMPERATURE_SENTIMENT: float = float(os.getenv("LLM_TEMPERATURE_SENTIMENT", "0.1")) # Node 09
    LLM_TEMPERATURE_OPINION_SUMMARIZE: float = float(os.getenv("LLM_TEMPERATURE_OPINION_SUMMARIZE", "0.4")) # Node 09
    LLM_TEMPERATURE_SYNTHESIS: float = float(os.getenv("LLM_TEMPERATURE_SYNTHESIS", "0.5")) # Node 10
    LLM_TEMPERATURE_CREATIVE: float = float(os.getenv("LLM_TEMPERATURE_CREATIVE", "0.7")) # Node 14 (Idea)
    LLM_TEMPERATURE_SCENARIO: float = float(os.getenv("LLM_TEMPERATURE_SCENARIO", LLM_TEMPERATURE_CREATIVE)) # Node 15 (Default to creative)
    LLM_TEMPERATURE_SCENARIO_EVAL: float = float(os.getenv("LLM_TEMPERATURE_SCENARIO_EVAL", "0.3")) # Node 16

    # --- 요약 평가 및 합성 (Node 10, 11) 관련 설정 ---
    DEFAULT_EVALUATION_THRESHOLDS: Dict[str, float] = {
        "rouge_l": 0.35,
        "bert_score": 0.88,
        "topic_coverage": 0.70
    }
    # DEFAULT_DECISION_THRESHOLDS는 Node 11 로직 확인 후 정의 필요 (일단 빈 딕셔너리)
    DEFAULT_DECISION_THRESHOLDS: Dict[str, Any] = {}
    DEFAULT_BERTSCORE_LANG: str = "en"
    DEFAULT_FEQA_THRESHOLD: float = 0.75
    DEFAULT_MAX_SUMMARIES_SYNTHESIS: int = 3
    MAX_ARTICLES_SUMMARIZE: int = 5 # 기본값 5로 설정

    DEFAULT_LLM_TEMP_SYNTHESIS: float = 0.5        # Node 10 용
    DEFAULT_MAX_TOKENS_SYNTHESIS: int = 512        # Node 10 용
    DEFAULT_SYNTHESIS_WORD_COUNT: int = 200        # Node 10 용
    # --------------------------------------------------
    # 작업별 최대 토큰 수 (결과 길이 및 비용 조절)
    MAX_TOKENS_EXTRACT: int = int(os.getenv("MAX_TOKENS_EXTRACT", "512")) # Node 08
    MAX_TOKENS_SUMMARIZE: int = int(os.getenv("MAX_TOKENS_SUMMARIZE", "300")) # Node 08
    MAX_TOKENS_QA_GEN: int = int(os.getenv("MAX_TOKENS_QA_GEN", "512")) # Node 08
    MAX_TOKENS_QA_VERIFY: int = int(os.getenv("MAX_TOKENS_QA_VERIFY", "100")) # Node 08
    MAX_TOKENS_STANCE: int = int(os.getenv("MAX_TOKENS_STANCE", "10")) # Node 09
    MAX_TOKENS_SENTIMENT: int = int(os.getenv("MAX_TOKENS_SENTIMENT", "10")) # Node 09
    MAX_TOKENS_OPINION_SUMMARIZE: int = int(os.getenv("MAX_TOKENS_OPINION_SUMMARIZE", "400")) # Node 09
    MAX_TOKENS_SYNTHESIS: int = int(os.getenv("MAX_TOKENS_SYNTHESIS", "300")) # Node 10
    MAX_TOKENS_IDEA: int = int(os.getenv("LLM_MAX_TOKENS_IDEA", "1024")) # Node 14
    MAX_TOKENS_SCENARIO: int = int(os.getenv("LLM_MAX_TOKENS_SCENARIO", "2048")) # Node 15
    MAX_TOKENS_SCENARIO_EVAL: int = int(os.getenv("LLM_MAX_TOKENS_SCENARIO_EVAL", "512")) # Node 16

    # --- 이미지 생성 동작 튜닝 ---
    DEFAULT_IMAGE_MODEL: str = os.getenv("DEFAULT_IMAGE_MODEL", "dall-e-3") # 기본 사용할 이미지 생성 모델 ID.
    IMAGE_DEFAULT_STYLE: str = os.getenv("IMAGE_DEFAULT_STYLE", "4-panel comic style, simple illustration, clear lines") # 기본 이미지 스타일 프롬프트.
    MAX_IMAGE_PROMPT_LEN: int = int(os.getenv("MAX_IMAGE_PROMPT_LEN", "500")) # 이미지 생성 프롬프트 최대 길이.
    IMAGE_HEIGHT: int = int(os.getenv("IMAGE_HEIGHT", "1024")) # 생성 이미지 높이.
    IMAGE_WIDTH: int = int(os.getenv("IMAGE_WIDTH", "1024")) # 생성 이미지 너비.
    IMAGE_NEGATIVE_PROMPT: str = os.getenv("IMAGE_NEGATIVE_PROMPT", "text, letters, words, signature, watermark, ugly, deformed, blurry, low quality") # 제외할 요소 프롬프트.
    IMAGE_STYLE_PRESET: Optional[str] = os.getenv("IMAGE_STYLE_PRESET") # 이미지 생성 API의 스타일 프리셋 사용 (지원 시).
    IMAGE_API_RETRIES: int = int(os.getenv("IMAGE_API_RETRIES", LLM_API_RETRIES)) # 이미지 API 재시도 횟수 (LLM 기본값 따름).

    # --- 번역 동작 튜닝 ---
    TARGET_TRANSLATION_LANGUAGE: str = os.getenv("TARGET_TRANSLATION_LANGUAGE", "en") # 번역 목표 언어 코드 (e.g., 'en', 'ko').
    SOURCE_LANGUAGE: Optional[str] = os.getenv("SOURCE_LANGUAGE") # 번역 소스 언어 코드 (미지정 시 자동 감지 시도).
    TRANSLATOR_CONCURRENCY: int = int(os.getenv("TRANSLATOR_CONCURRENCY", "3")) # 번역 작업 동시 실행 수.
    # PAPAGO_API_RETRIES: int = int(os.getenv("PAPAGO_API_RETRIES", LLM_API_RETRIES)) # 필요시 Papago 전용 재시도 설정 (TOOL_RETRY_ATTEMPTS 사용 가능).

    # --- 검색/트렌드 도구 동작 튜닝 ---
    Google_Search_API_RETRIES: int = int(os.getenv("Google_Search_API_RETRIES", TOOL_RETRY_ATTEMPTS)) # Google Search 전용 재시도 (기본 도구 재시도 따름).
    # Google Trends (PyTrends)
    TREND_GOOGLE_WEIGHT: float = float(os.getenv("TREND_GOOGLE_WEIGHT", "0.6"))
    TREND_TWITTER_WEIGHT: float = float(os.getenv("TREND_TWITTER_WEIGHT", "0.4"))
    PYTRENDS_TIMEFRAME: str = os.getenv("PYTRENDS_TIMEFRAME", "now 7-d") # Google Trends 조회 기간.
    PYTRENDS_GEO: str = os.getenv("PYTRENDS_GEO", "") # Google Trends 지역 필터 (e.g., 'US', 'KR').
    PYTRENDS_HL: str = os.getenv("PYTRENDS_HL", "en-US") # Google Trends 인터페이스 언어.
    PYTRENDS_BATCH_DELAY_SEC: float = float(os.getenv("PYTRENDS_BATCH_DELAY_SEC", "1.5")) # Pytrends 요청 간 지연 시간(초).
    # Twitter
    TWITTER_COUNTS_DELAY_SEC: float = float(os.getenv("TWITTER_COUNTS_DELAY_SEC", "3")) # Twitter API 요청 간 지연 시간(초).
    TWITTER_OPINION_MAX_RESULTS: int = int(os.getenv("TWITTER_OPINION_MAX_RESULTS", "15")) # Twitter 의견 수집 시 최대 결과 수.
    DEFAULT_MIN_OPINIONS_PER_PLATFORM: int = 1
    DEFAULT_MAX_OPINIONS_PER_PLATFORM: int = 2
    # Reddit
    REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "ComicGeneratorAgent/1.0") # Reddit API 사용 시 User Agent.
    REDDIT_TARGET_SUBREDDITS: List[str] = _parse_list_helper(
        os.getenv("REDDIT_TARGET_SUBREDDITS", "news,worldnews,technology,science,futurology,korea,AskReddit,datascience,artificial")
    ) # Reddit 의견 수집 대상 서브레딧 목록.
    REDDIT_OPINION_MAX_RESULTS: int = int(os.getenv("REDDIT_OPINION_MAX_RESULTS", "15")) # Reddit 의견 수집 시 최대 결과 수.
    # YouTube
    YOUTUBE_OPINION_MAX_RESULTS: int = int(os.getenv("YOUTUBE_OPINION_MAX_RESULTS", "25")) # YouTube 댓글/의견 수집 시 최대 결과 수.
    # Blog/Community
    BLOG_OPINION_MAX_RESULTS: int = int(os.getenv("BLOG_OPINION_MAX_RESULTS", "10")) # 블로그 의견 수집 시 최대 결과 수.
    COMMUNITY_OPINION_MAX_RESULTS: int = int(os.getenv("COMMUNITY_OPINION_MAX_RESULTS", "10")) # 커뮤니티 의견 수집 시 최대 결과 수.
    TARGET_COMMUNITY_DOMAINS: List[str] = _parse_list_helper(
        os.getenv("TARGET_COMMUNITY_DOMAINS", "dcinside.com,clien.net,ruliweb.com,fmkorea.com,theqoo.net,instiz.net,etoland.co.kr,ppomppu.co.kr,82cook.com,quora.com,stackoverflow.com")
    ) # 의견 수집 대상 커뮤니티 도메인 목록 (주로 Google 검색과 연계 사용).
    # RSS Feeds
    PREDEFINED_RSS_FEEDS: List[str] = _parse_list_helper(os.getenv("PREDEFINED_RSS_FEEDS")) or [
        "http://feeds.bbci.co.uk/news/world/rss.xml", "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "http://rss.cnn.com/rss/edition.rss", "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=01&plink=RSSREADER"
    ] # 기본 RSS 피드 목록 (Node 03 Tool).

    # --- Redis Cache 동작 튜닝 ---
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379")) # Redis 서버 포트.
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0")) # 사용할 Redis 데이터베이스 번호.
    TOPIC_ANALYZER_CACHE_TTL: int = int(os.getenv("TOPIC_ANALYZER_CACHE_TTL", "1800")) # 토픽 분석 결과 캐시 유지 시간(초, 30분).

    # --- LangSmith 동작 튜닝 ---
    LANGCHAIN_PROJECT: Optional[str] = os.getenv("LANGCHAIN_PROJECT", "ComicGenerator-Project") # LangSmith 프로젝트 이름.
    LANGCHAIN_ENDPOINT: Optional[str] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com") # LangSmith 엔드포인트 주소.

    # --- 데이터 수집 및 처리 파라미터 ---
    SEARCH_RESULT_COUNT: int = int(os.getenv("SEARCH_RESULT_COUNT", "10")) # 검색 API 호출 시 요청할 결과 수.
    TARGET_URLS_PER_KEYWORD: int = int(os.getenv("TARGET_URLS_PER_KEYWORD", "5")) # 키워드 당 수집할 목표 URL 수.
    TRUSTED_NEWS_DOMAINS: List[str] = _parse_list_helper(os.getenv("TRUSTED_NEWS_DOMAINS")) or [
        "news.kbs.co.kr", "news.sbs.co.kr", "yna.co.kr", "news.kmib.co.kr", "chosun.com", "joongang.co.kr", "donga.com", "hani.co.kr", "khan.co.kr", "hankyung.com", "mt.co.kr", "mk.co.kr", "etnews.com", "zdnet.co.kr", "bloter.net", "bbc.com", "bbc.co.uk", "cnn.com", "nytimes.com", "reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "ft.com", "economist.com", "theguardian.com", "washingtonpost.com", "npr.org", "axios.com", "politico.com"
    ] # 신뢰할 수 있는 뉴스 출처 도메인 목록.
    MAX_ARTICLES_TO_SCRAPE: int = int(os.getenv("MAX_ARTICLES_TO_SCRAPE", "5")) # 스크랩할 최대 뉴스 기사 수 (URL 기준).
    MAX_TOTAL_OPINIONS_TARGET: int = int(os.getenv("MAX_TOTAL_OPINIONS_TARGET", "60")) # 모든 플랫폼에서 수집/처리할 최대 의견 수 목표치.
    # 플랫폼별 의견 수집 목표치 (최대/최소)
    MAX_OPINIONS_PER_PLATFORM_SAMPLING: Dict[str, int] = _parse_json_helper(
        os.getenv("MAX_OPINIONS_PER_PLATFORM_SAMPLING"),
        default={"Twitter": 15, "Reddit": 15, "YouTube": 10, "Blog": 10, "Community": 10}
    )
    MIN_OPINIONS_PER_PLATFORM_SAMPLING: Dict[str, int] = _parse_json_helper(
        os.getenv("MIN_OPINIONS_PER_PLATFORM_SAMPLING"),
        default={"Twitter": 2, "Reddit": 2, "YouTube": 1, "Blog": 2, "Community": 2}
    )
    # 스크래핑 동작 튜닝
    SCRAPER_USER_AGENT: str = os.getenv("SCRAPER_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36") # 웹 스크래핑 시 사용할 User Agent.
    SCRAPER_HTTP_TIMEOUT: int = int(os.getenv("SCRAPER_HTTP_TIMEOUT", "20")) # 스크래핑 HTTP 요청 타임아웃(초).
    SCRAPER_CONCURRENCY: int = int(os.getenv("SCRAPER_CONCURRENCY", "5")) # 뉴스 기사 스크래핑 동시 실행 수.
    OPINION_SCRAPER_CONCURRENCY: int = int(os.getenv("OPINION_SCRAPER_CONCURRENCY", "3")) # 의견 스크래핑 동시 실행 수.
    MIN_EXTRACTED_TEXT_LENGTH: int = int(os.getenv("MIN_EXTRACTED_TEXT_LENGTH", "100")) # 유효한 텍스트로 간주할 최소 길이.
    MIN_LANGDETECT_TEXT_LENGTH: int = int(os.getenv("MIN_LANGDETECT_TEXT_LENGTH", "50")) # 언어 감지를 위한 최소 텍스트 길이.
    # Selenium (동적 스크래핑) 설정
    SELENIUM_GRID_URL: Optional[str] = os.getenv("SELENIUM_GRID_URL") # Selenium Grid 사용 시 허브 URL.
    SELENIUM_HEADLESS: bool = os.getenv("SELENIUM_HEADLESS", "True").lower() in ["true", "1", "yes"] # Selenium 브라우저 Headless 모드 사용 여부.
    SELENIUM_RETRY_ATTEMPTS: int = int(os.getenv("SELENIUM_RETRY_ATTEMPTS", "2")) # Selenium 작업 재시도 횟수.
    SELENIUM_RETRY_WAIT_SECONDS: int = int(os.getenv("SELENIUM_RETRY_WAIT_SECONDS", "1")) # Selenium 재시도 간 대기 시간(초).
    # 스크래핑 지연 및 프록시
    SCRAPER_MIN_DELAY_MS: int = int(os.getenv("SCRAPER_MIN_DELAY_MS", "1500")) # 스크래핑 요청 사이 최소 지연 시간(ms).
    SCRAPER_MAX_DELAY_MS: int = int(os.getenv("SCRAPER_MAX_DELAY_MS", "4000")) # 스크래핑 요청 사이 최대 지연 시간(ms).
    SCRAPER_USE_PROXY: bool = os.getenv("SCRAPER_USE_PROXY", "false").lower() == "true" # 스크래핑 시 프록시 사용 여부.
    SCRAPER_PROXY_URL: Optional[str] = os.getenv("SCRAPER_PROXY_URL") # 사용할 프록시 서버 주소 (SCRAPER_USE_PROXY=True 시).
    SCRAPER_ROTATE_UA: bool = os.getenv("SCRAPER_ROTATE_UA", "false").lower() == "true" # User Agent 로테이션 사용 여부.
    # 텍스트 필터링 및 정제 (Node 07)
    LANGUAGE_FILTER: List[str] = _parse_list_helper(os.getenv("LANGUAGE_FILTER", "en,ko")) # 허용할 언어 코드 목록.
    SPAM_KEYWORDS: List[str] = _parse_list_helper(os.getenv("SPAM_KEYWORDS", "buy now,click here,order now,special offer,limited time,discount,free,cheap,amazing,winner,prize,cash,casino,lottery,viagra,pharmacy,subscribe,loan,credit,mortgage,debt,investment,earn money,make money,weight loss,diet,guaranteed,100%,$$$")) # 스팸으로 간주할 키워드 목록.
    SPAM_MAX_URL_COUNT: int = int(os.getenv("SPAM_MAX_URL_COUNT", "2")) # 스팸으로 간주할 최대 URL 포함 개수.
    SPAM_MAX_UPPERCASE_RATIO: float = float(os.getenv("SPAM_MAX_UPPERCASE_RATIO", "0.5")) # 스팸으로 간주할 최대 대문자 비율.
    # 중복 제거 (Simhash)
    SIMHASH_THRESHOLD: int = int(os.getenv("SIMHASH_THRESHOLD", "3")) # Simhash 유사도 임계값 (낮을수록 유사).
    SIMHASH_TOKEN_WIDTH: int = int(os.getenv("SIMHASH_TOKEN_WIDTH", "64")) # Simhash 해시 비트 수 (보통 64 또는 128).
    # 군집화 (KMeans/TF-IDF)
    KMEANS_DEFAULT_CLUSTERS: int = int(os.getenv("KMEANS_DEFAULT_CLUSTERS", "5")) # KMeans 기본 클러스터 수.
    KMEANS_MIN_SAMPLES: int = int(os.getenv("KMEANS_MIN_SAMPLES", "10")) # KMeans 클러스터링을 위한 최소 샘플 수.
    TFIDF_MAX_FEATURES: int = int(os.getenv("TFIDF_MAX_FEATURES", "5000")) # TF-IDF 벡터화 시 최대 피처 수.
    TFIDF_STOP_WORDS: Optional[str] = os.getenv("TFIDF_STOP_WORDS", "english") # TF-IDF 불용어 처리 방식 ('english' 또는 None).
    TFIDF_MIN_DF: int = int(os.getenv("TFIDF_MIN_DF", "2")) # TF-IDF 단어 최소 등장 빈도.
    TFIDF_MAX_DF: float = float(os.getenv("TFIDF_MAX_DF", "0.90")) # TF-IDF 단어 최대 등장 비율.
    TFIDF_NGRAM_RANGE_MIN: int = int(os.getenv("TFIDF_NGRAM_RANGE_MIN", "1")) # TF-IDF N-gram 범위 최소값.
    TFIDF_NGRAM_RANGE_MAX: int = int(os.getenv("TFIDF_NGRAM_RANGE_MAX", "2")) # TF-IDF N-gram 범위 최대값.
    KMEANS_N_INIT: int = int(os.getenv("KMEANS_N_INIT", "10")) # KMeans 초기 중심점 시도 횟수.
    KMEANS_MAX_NO_IMPROVEMENT: int = int(os.getenv("KMEANS_MAX_NO_IMPROVEMENT", "20")) # KMeans 조기 종료 조건 (개선 없을 시 반복 횟수).

    # --- 평가 및 의사결정 기준 (Nodes 08, 10, 11) ---
    FEQA_THRESHOLD: float = float(os.getenv("FEQA_THRESHOLD", "0.5")) # FEQA 평가 임계값.
    EVALUATION_THRESHOLDS: Dict[str, float] = _parse_json_helper(
        os.getenv("EVALUATION_THRESHOLDS"), default={"rouge_l": 0.3, "bert_score": 0.7, "topic_coverage": 0.6}
    ) # 요약/생성 결과 평가 지표별 임계값.
    DECISION_LOGIC_THRESHOLDS: Dict[str, float] = _parse_json_helper(
        os.getenv("DECISION_LOGIC_THRESHOLDS"), default={"very_low_rouge": 0.1, "very_low_bertscore": 0.5, "very_low_coverage": 0.3, "low_coverage_high_metrics": 0.7}
    ) # 재시도/대체 경로 결정을 위한 임계값.
    BERTSCORE_LANG: str = os.getenv("BERTSCORE_LANG", "en") # BertScore 계산 시 사용할 언어 모델.

    # --- 요약 및 창의적 작업 튜닝 (Nodes 08-10, 14-16) ---
    SUMMARIZER_CONCURRENCY: int = int(os.getenv("SUMMARIZER_CONCURRENCY", "3")) # 요약 작업 동시 실행 수.

    # --- 보고서 및 결과물 생성 설정 (Nodes 13, 16, 19) ---
    # 보고서 템플릿
    TEMPLATE_DIR: str = os.getenv("TEMPLATE_DIR", os.path.join(PROJECT_ROOT_DIR, "app", "templates")) # 보고서 템플릿 파일 디렉토리 경로.
    PROGRESS_REPORT_TEMPLATE_A_FILENAME: str = os.getenv("PROGRESS_REPORT_TEMPLATE_A_FILENAME", "progress_report_template_a.md.j2") # 중간 보고서 템플릿 파일명.
    PROGRESS_REPORT_TEMPLATE_B_FILENAME: str = os.getenv("PROGRESS_REPORT_TEMPLATE_B_FILENAME", "scenario_report_template_b.md.j2") # 시나리오 보고서 템플릿 파일명.
    TRENDS_REPORT_TOP_N: int = int(os.getenv("TRENDS_REPORT_TOP_N", "3")) # 트렌드 보고서에 포함할 상위 N개 항목.
    # 최종 결과물 (이미지, 만화) 후처리
    IMAGE_STORAGE_PATH: str = os.getenv("IMAGE_STORAGE_PATH", os.path.join(PROJECT_ROOT_DIR, 'storage', 'images')) # 생성된 중간 이미지 저장 경로.
    DEFAULT_FONT_PATH: Optional[str] = os.getenv("DEFAULT_FONT_PATH") # 텍스트 오버레이에 사용할 기본 폰트 파일 경로 (미지정 시 시스템 기본).
    FINAL_COMIC_SAVE_DIR: str = os.getenv("FINAL_COMIC_SAVE_DIR", os.path.join(PROJECT_ROOT_DIR, "final_comics")) # 최종 만화 파일 저장 디렉토리.
    FINAL_IMAGE_FORMAT: str = os.getenv("FINAL_IMAGE_FORMAT", "WEBP").upper() # 최종 이미지 저장 포맷 (e.g., WEBP, PNG, JPEG).
    FINAL_IMAGE_WIDTH: int = int(os.getenv("FINAL_IMAGE_WIDTH", "1024")) # 최종 이미지 너비 (리사이징).
    FINAL_IMAGE_QUALITY: int = int(os.getenv("FINAL_IMAGE_QUALITY", "85")) # 최종 이미지 저장 품질 (JPEG/WEBP).
    TEXT_OVERLAY_FONT_SIZE_RATIO: float = float(os.getenv("TEXT_OVERLAY_FONT_SIZE_RATIO", "20")) # 이미지 높이 대비 텍스트 크기 비율.
    TEXT_OVERLAY_COLOR: str = os.getenv("TEXT_OVERLAY_COLOR", "black") # 텍스트 오버레이 색상.
    IMAGE_DOWNLOAD_RETRIES: int = int(os.getenv("IMAGE_DOWNLOAD_RETRIES", "3")) # 이미지 다운로드 재시도 횟수.
    MAX_ALT_TEXT_LEN: int = int(os.getenv("MAX_ALT_TEXT_LEN", "300")) # 생성할 Alt Text 최대 길이.
    # 결과 저장 경로
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", os.path.join(PROJECT_ROOT_DIR, 'results')) # 에이전트/노드 결과 및 디버그 정보 저장 기본 경로.


    def __init__(self):
        """설정 객체 초기화 시 필요한 작업 수행 (예: 디렉토리 생성)."""
        # 필요한 디렉토리 생성 (결과, 이미지 저장 등)
        dirs_to_create = [
            self.RESULTS_DIR,
            self.IMAGE_STORAGE_PATH,
            self.FINAL_COMIC_SAVE_DIR,
            # TEMPLATE_DIR 은 존재 여부 확인만 필요할 수 있음
        ]
        for dir_path in dirs_to_create:
             if isinstance(dir_path, str):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except OSError as e:
                     print(f"Warning: Failed to create directory '{dir_path}'. Error: {e}")

# --- 애플리케이션 전체에서 사용할 단일 설정 인스턴스 생성 ---
settings = Settings()