# app/config/settings.py (Merged Version)

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
             # 앱 초기 단계이므로 print 사용
             print(f"Warning: Failed to parse JSON from env var. Value: '{env_var_value[:50]}...'. Error: {e}. Using default: {_default}")
             return _default
    return _default

class Settings:
    """
    (Merged & Refactored) 애플리케이션 설정을 관리하는 중앙 클래스.
    모든 노드 및 서비스/도구에서 필요한 설정을 통합 관리합니다.
    """

    # --- Core Settings ---
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "ComicGenerator")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ["true", "1", "yes"]
    # --- 도구(Tool) 공통 설정 (추가) ---
    # 참고: LLM_API_RETRIES, IMAGE_API_RETRIES 등 구체적인 설정이 우선될 수 있습니다.
    # 이 설정은 일반적인 도구 또는 특정 설정이 없는 경우의 기본값으로 활용 가능합니다.
    TOOL_RETRY_ATTEMPTS: int = int(os.getenv("TOOL_RETRY_ATTEMPTS", "3"))  # 도구 기본 재시도 횟수
    TOOL_RETRY_WAIT_MIN: int = int(os.getenv("TOOL_RETRY_WAIT_MIN", "1"))  # 도구 재시도 최소 대기(초)
    TOOL_RETRY_WAIT_MAX: int = int(os.getenv("TOOL_RETRY_WAIT_MAX", "10"))  # 도구 재시도 최대 대기(초)
    TOOL_HTTP_TIMEOUT: int = int(os.getenv("TOOL_HTTP_TIMEOUT", "15"))  # 도구 일반 HTTP 타임아웃

    # --- LLM API Settings (Nodes 02, 08, 09, 10, 14, 15, Optional 16) ---
    LLM_API_ENDPOINT: Optional[str] = os.getenv("LLM_API_ENDPOINT")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_API_TIMEOUT: int = int(os.getenv("LLM_API_TIMEOUT", "60"))
    LLM_API_RETRIES: int = int(os.getenv("LLM_API_RETRIES", "3")) # LLM 및 기타 API 기본 재시도 횟수
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")

    # LLM Temperature & Token Settings (per task)
    LLM_TEMPERATURE_ANALYSIS: float = float(os.getenv("LLM_TEMPERATURE_ANALYSIS", "0.2")) # Node 02
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

    # --- Image Generation Settings (Node 17) ---
    IMAGE_SERVER_URL: Optional[str] = os.getenv("IMAGE_SERVER_URL")
    IMAGE_SERVER_API_TOKEN: Optional[str] = os.getenv("IMAGE_SERVER_API_TOKEN")
    DEFAULT_IMAGE_MODEL: str = os.getenv("DEFAULT_IMAGE_MODEL", "dall-e-3")
    IMAGE_DEFAULT_STYLE: str = os.getenv("IMAGE_DEFAULT_STYLE", "4-panel comic style, simple illustration, clear lines")
    MAX_IMAGE_PROMPT_LEN: int = int(os.getenv("MAX_IMAGE_PROMPT_LEN", "500"))
    IMAGE_HEIGHT: int = int(os.getenv("IMAGE_HEIGHT", "1024"))
    IMAGE_WIDTH: int = int(os.getenv("IMAGE_WIDTH", "1024"))
    IMAGE_NEGATIVE_PROMPT: str = os.getenv("IMAGE_NEGATIVE_PROMPT", "text, letters, words, signature, watermark, ugly, deformed, blurry, low quality")
    IMAGE_STYLE_PRESET: Optional[str] = os.getenv("IMAGE_STYLE_PRESET")
    IMAGE_API_RETRIES: int = int(os.getenv("IMAGE_API_RETRIES", LLM_API_RETRIES)) # Default to LLM retries

    # --- Translation Settings (Node 18) ---
    # Papago API uses Naver keys by default, or define specific PAPAGO_* keys
    NAVER_CLIENT_ID: Optional[str] = os.getenv("NAVER_CLIENT_ID")
    NAVER_CLIENT_SECRET: Optional[str] = os.getenv("NAVER_CLIENT_SECRET")
    TRANSLATION_ENABLED: bool = os.getenv("TRANSLATION_ENABLED", "false").lower() == "true"
    TARGET_TRANSLATION_LANGUAGE: str = os.getenv("TARGET_TRANSLATION_LANGUAGE", "en")
    SOURCE_LANGUAGE: Optional[str] = os.getenv("SOURCE_LANGUAGE") # Optional: Explicit source lang
    TRANSLATOR_CONCURRENCY: int = int(os.getenv("TRANSLATOR_CONCURRENCY", "3"))
    # PAPAGO_API_RETRIES: int = int(os.getenv("PAPAGO_API_RETRIES", LLM_API_RETRIES)) # Use LLM retries

    # --- Search / Trend Tool Settings (Node 03, 04 Tools, Node 12 Tools) ---
    # Google Search
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID: Optional[str] = os.getenv("GOOGLE_CSE_ID")
    # Google Search 전용 재시도 설정 추가
    Google_Search_API_RETRIES: int = int(os.getenv("Google Search_API_RETRIES", "3"))  # .env 파일 또는 기본값 3회
    # Google Search API 타임아웃 (TOOL_HTTP_TIMEOUT 또는 별도 설정 가능)
    # Google Search_API_TIMEOUT: int = int(os.getenv("Google Search_API_TIMEOUT", "10"))
    # NAVER_API_RETRIES: int = int(os.getenv("NAVER_API_RETRIES", "3"))

    # Google Trends
    PYTRENDS_TIMEFRAME: str = os.getenv("PYTRENDS_TIMEFRAME", "today 7-d")
    PYTRENDS_GEO: str = os.getenv("PYTRENDS_GEO", "")
    PYTRENDS_HL: str = os.getenv("PYTRENDS_HL", "en-US")
    PYTRENDS_BATCH_DELAY_SEC: float = float(os.getenv("PYTRENDS_BATCH_DELAY_SEC", "1.5"))
    # Twitter
    TWITTER_BEARER_TOKEN: Optional[str] = os.getenv("TWITTER_BEARER_TOKEN")
    TWITTER_COUNTS_DELAY_SEC: float = float(os.getenv("TWITTER_COUNTS_DELAY_SEC", "0.6"))
    TWITTER_OPINION_MAX_RESULTS: int = int(os.getenv("TWITTER_OPINION_MAX_RESULTS", "15")) # Node 04 Tool use
    # Reddit
    REDDIT_CLIENT_ID: Optional[str] = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET: Optional[str] = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "ComicGeneratorAgent/1.0")
    REDDIT_USERNAME: Optional[str] = os.getenv("REDDIT_USERNAME") # Optional: Script-type auth
    REDDIT_PASSWORD: Optional[str] = os.getenv("REDDIT_PASSWORD") # Optional: Script-type auth
    REDDIT_TARGET_SUBREDDITS: List[str] = _parse_list_helper(
        os.getenv("REDDIT_TARGET_SUBREDDITS", "news,worldnews,technology,science,futurology,korea,AskReddit,datascience,artificial")
    )
    REDDIT_OPINION_MAX_RESULTS: int = int(os.getenv("REDDIT_OPINION_MAX_RESULTS", "15")) # Node 04 Tool use
    # YouTube (Search only, no separate API client in services yet)
    YOUTUBE_API_KEY: Optional[str] = os.getenv("YOUTUBE_API_KEY")
    YOUTUBE_OPINION_MAX_RESULTS: int = int(os.getenv("YOUTUBE_OPINION_MAX_RESULTS", "25")) # Node 04 Tool use
    # Blog/Community Search (using Google Search CSE usually)
    BLOG_OPINION_MAX_RESULTS: int = int(os.getenv("BLOG_OPINION_MAX_RESULTS", "10")) # Node 04 Tool use
    COMMUNITY_OPINION_MAX_RESULTS: int = int(os.getenv("COMMUNITY_OPINION_MAX_RESULTS", "10")) # Node 04 Tool use
    TARGET_COMMUNITY_DOMAINS: List[str] = _parse_list_helper(
        os.getenv("TARGET_COMMUNITY_DOMAINS", "dcinside.com,clien.net,ruliweb.com,fmkorea.com,theqoo.net,instiz.net,etoland.co.kr,ppomppu.co.kr,82cook.com,quora.com,stackoverflow.com")
    )
    # RSS Feeds (Node 03 Tool)
    PREDEFINED_RSS_FEEDS: List[str] = _parse_list_helper(os.getenv("PREDEFINED_RSS_FEEDS")) or [
        "http://feeds.bbci.co.uk/news/world/rss.xml", "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "http://rss.cnn.com/rss/edition.rss", "https://news.sbs.co.kr/news/SectionRssFeed.do?sectionId=01&plink=RSSREADER"
    ]
    # General Tool Settings (Retries often use LLM_API_RETRIES)
    TOOL_HTTP_TIMEOUT: int = int(os.getenv("TOOL_HTTP_TIMEOUT", "15")) # Timeout for tool HTTP calls
    API_FETCH_TIMEOUT: int = int(os.getenv("API_FETCH_TIMEOUT", "10")) # Specific timeout for quick API fetches

    # --- Redis Cache/DB (Node 02 Cache) ---
    REDIS_HOST: Optional[str] = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    TOPIC_ANALYZER_CACHE_TTL: int = int(os.getenv("TOPIC_ANALYZER_CACHE_TTL", "1800")) # 30 min

    # --- LangSmith (Optional Observability) ---
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: Optional[str] = os.getenv("LANGCHAIN_PROJECT", "ComicGenerator-Project")
    LANGCHAIN_ENDPOINT: Optional[str] = os.getenv("LANGCHAIN_ENDPOINT")

    # --- AWS S3 Storage (Node 19 Upload, Image Storage) ---
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")
    AWS_REGION: Optional[str] = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION"))
    UPLOAD_TO_S3: bool = os.getenv("UPLOAD_TO_S3", "false").lower() == "true"

    # --- Data Collection & Processing Parameters (Nodes 03, 04, 05, 06, 07) ---
    SEARCH_RESULT_COUNT: int = int(os.getenv("SEARCH_RESULT_COUNT", "10")) # API 결과 요청 수
    TARGET_URLS_PER_KEYWORD: int = int(os.getenv("TARGET_URLS_PER_KEYWORD", "5")) # 키워드당 목표 URL 수
    TRUSTED_NEWS_DOMAINS: List[str] = _parse_list_helper(os.getenv("TRUSTED_NEWS_DOMAINS")) or [
        "news.kbs.co.kr", "news.sbs.co.kr", "yna.co.kr", "news.kmib.co.kr", "chosun.com", "joongang.co.kr", "donga.com", "hani.co.kr", "khan.co.kr", "hankyung.com", "mt.co.kr", "mk.co.kr", "etnews.com", "zdnet.co.kr", "bloter.net", "bbc.com", "bbc.co.uk", "cnn.com", "nytimes.com", "reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "ft.com", "economist.com", "theguardian.com", "washingtonpost.com", "npr.org", "axios.com", "politico.com"
    ]
    MAX_ARTICLES_TO_SCRAPE: int = int(os.getenv("MAX_ARTICLES_TO_SCRAPE", "5")) # 스크랩할 최대 뉴스 기사 수
    MAX_TOTAL_OPINIONS_TARGET: int = int(os.getenv("MAX_TOTAL_OPINIONS_TARGET", "60")) # 스크랩/처리할 최대 의견 수
    MAX_OPINIONS_PER_PLATFORM_SAMPLING: Dict[str, int] = _parse_json_helper(
        os.getenv("MAX_OPINIONS_PER_PLATFORM_SAMPLING"),
        default={"Twitter": 15, "Reddit": 15, "YouTube": 10, "Blog": 10, "Community": 10}
    )
    MIN_OPINIONS_PER_PLATFORM_SAMPLING: Dict[str, int] = _parse_json_helper(
        os.getenv("MIN_OPINIONS_PER_PLATFORM_SAMPLING"),
        default={"Twitter": 2, "Reddit": 2, "YouTube": 1, "Blog": 2, "Community": 2}
    )
    SCRAPER_USER_AGENT: str = os.getenv("SCRAPER_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    SCRAPER_HTTP_TIMEOUT: int = int(os.getenv("SCRAPER_HTTP_TIMEOUT", "20"))
    SCRAPER_CONCURRENCY: int = int(os.getenv("SCRAPER_CONCURRENCY", "5")) # 뉴스 스크래핑 동시성
    OPINION_SCRAPER_CONCURRENCY: int = int(os.getenv("OPINION_SCRAPER_CONCURRENCY", "3")) # 의견 스크래핑 동시성
    MIN_EXTRACTED_TEXT_LENGTH: int = int(os.getenv("MIN_EXTRACTED_TEXT_LENGTH", "100"))
    MIN_LANGDETECT_TEXT_LENGTH: int = int(os.getenv("MIN_LANGDETECT_TEXT_LENGTH", "50"))
    SELENIUM_GRID_URL: Optional[str] = os.getenv("SELENIUM_GRID_URL")
    SELENIUM_HEADLESS: bool = os.getenv("SELENIUM_HEADLESS", "True").lower() in ["true", "1", "yes"]
    SELENIUM_RETRY_ATTEMPTS: int = int(os.getenv("SELENIUM_RETRY_ATTEMPTS", "2"))
    SELENIUM_RETRY_WAIT_SECONDS: int = int(os.getenv("SELENIUM_RETRY_WAIT_SECONDS", "1"))
    SCRAPER_MIN_DELAY_MS: int = int(os.getenv("SCRAPER_MIN_DELAY_MS", "1500"))
    SCRAPER_MAX_DELAY_MS: int = int(os.getenv("SCRAPER_MAX_DELAY_MS", "4000"))
    SCRAPER_USE_PROXY: bool = os.getenv("SCRAPER_USE_PROXY", "false").lower() == "true"
    SCRAPER_PROXY_URL: Optional[str] = os.getenv("SCRAPER_PROXY_URL")
    SCRAPER_ROTATE_UA: bool = os.getenv("SCRAPER_ROTATE_UA", "false").lower() == "true"
    LANGUAGE_FILTER: List[str] = _parse_list_helper(os.getenv("LANGUAGE_FILTER", "en,ko")) # Node 07
    SPAM_KEYWORDS: List[str] = _parse_list_helper(os.getenv("SPAM_KEYWORDS", "buy now,click here,order now,special offer,limited time,discount,free,cheap,amazing,winner,prize,cash,casino,lottery,viagra,pharmacy,subscribe,loan,credit,mortgage,debt,investment,earn money,make money,weight loss,diet,guaranteed,100%,$$$")) # Node 07
    SPAM_MAX_URL_COUNT: int = int(os.getenv("SPAM_MAX_URL_COUNT", "2")) # Node 07
    SPAM_MAX_UPPERCASE_RATIO: float = float(os.getenv("SPAM_MAX_UPPERCASE_RATIO", "0.5")) # Node 07
    SIMHASH_THRESHOLD: int = int(os.getenv("SIMHASH_THRESHOLD", "3")) # Node 07
    SIMHASH_TOKEN_WIDTH: int = int(os.getenv("SIMHASH_TOKEN_WIDTH", "64")) # Node 07 (참고: Python simhash는 보통 64 또는 128)
    KMEANS_DEFAULT_CLUSTERS: int = int(os.getenv("KMEANS_DEFAULT_CLUSTERS", "5")) # Node 07
    KMEANS_MIN_SAMPLES: int = int(os.getenv("KMEANS_MIN_SAMPLES", "10")) # Node 07
    TFIDF_MAX_FEATURES: int = int(os.getenv("TFIDF_MAX_FEATURES", "5000")) # Node 07
    TFIDF_STOP_WORDS: Optional[str] = os.getenv("TFIDF_STOP_WORDS", "english") # Node 07
    TFIDF_MIN_DF: int = int(os.getenv("TFIDF_MIN_DF", "2")) # Node 07
    TFIDF_MAX_DF: float = float(os.getenv("TFIDF_MAX_DF", "0.90")) # Node 07
    TFIDF_NGRAM_RANGE_MIN: int = int(os.getenv("TFIDF_NGRAM_RANGE_MIN", "1")) # Node 07
    TFIDF_NGRAM_RANGE_MAX: int = int(os.getenv("TFIDF_NGRAM_RANGE_MAX", "2")) # Node 07
    KMEANS_N_INIT: int = int(os.getenv("KMEANS_N_INIT", "10")) # Node 07
    KMEANS_MAX_NO_IMPROVEMENT: int = int(os.getenv("KMEANS_MAX_NO_IMPROVEMENT", "20")) # Node 07

    # --- Evaluation & Decision Settings (Nodes 08, 10, 11) ---
    FEQA_THRESHOLD: float = float(os.getenv("FEQA_THRESHOLD", "0.5"))
    EVALUATION_THRESHOLDS: Dict[str, float] = _parse_json_helper(
        os.getenv("EVALUATION_THRESHOLDS"), default={"rouge_l": 0.3, "bert_score": 0.7, "topic_coverage": 0.6}
    )
    DECISION_LOGIC_THRESHOLDS: Dict[str, float] = _parse_json_helper(
        os.getenv("DECISION_LOGIC_THRESHOLDS"), default={"very_low_rouge": 0.1, "very_low_bertscore": 0.5, "very_low_coverage": 0.3, "low_coverage_high_metrics": 0.7}
    )
    BERTSCORE_LANG: str = os.getenv("BERTSCORE_LANG", "en")

    # --- Summarization & Creative Settings (Nodes 08-10, 14-16) ---
    SUMMARIZER_CONCURRENCY: int = int(os.getenv("SUMMARIZER_CONCURRENCY", "3"))

    # --- Reporting Settings (Nodes 13, 16) ---
    TEMPLATE_DIR: str = os.getenv("TEMPLATE_DIR", os.path.join(PROJECT_ROOT_DIR, "app", "templates"))
    PROGRESS_REPORT_TEMPLATE_A_FILENAME: str = os.getenv("PROGRESS_REPORT_TEMPLATE_A_FILENAME", "progress_report_template_a.md.j2")
    PROGRESS_REPORT_TEMPLATE_B_FILENAME: str = os.getenv("PROGRESS_REPORT_TEMPLATE_B_FILENAME", "scenario_report_template_b.md.j2")
    TRENDS_REPORT_TOP_N: int = int(os.getenv("TRENDS_REPORT_TOP_N", "3"))

    # --- Postprocessing Settings (Node 19) ---
    IMAGE_STORAGE_PATH: str = os.getenv("IMAGE_STORAGE_PATH", os.path.join(PROJECT_ROOT_DIR, 'storage', 'images')) # 이미지 클라이언트 저장 경로
    DEFAULT_FONT_PATH: Optional[str] = os.getenv("DEFAULT_FONT_PATH")
    FINAL_COMIC_SAVE_DIR: str = os.getenv("FINAL_COMIC_SAVE_DIR", os.path.join(PROJECT_ROOT_DIR, "final_comics"))
    FINAL_IMAGE_FORMAT: str = os.getenv("FINAL_IMAGE_FORMAT", "WEBP").upper()
    FINAL_IMAGE_WIDTH: int = int(os.getenv("FINAL_IMAGE_WIDTH", "1024"))
    FINAL_IMAGE_QUALITY: int = int(os.getenv("FINAL_IMAGE_QUALITY", "85"))
    TEXT_OVERLAY_FONT_SIZE_RATIO: float = float(os.getenv("TEXT_OVERLAY_FONT_SIZE_RATIO", "20"))
    TEXT_OVERLAY_COLOR: str = os.getenv("TEXT_OVERLAY_COLOR", "black")
    IMAGE_DOWNLOAD_RETRIES: int = int(os.getenv("IMAGE_DOWNLOAD_RETRIES", "3"))
    MAX_ALT_TEXT_LEN: int = int(os.getenv("MAX_ALT_TEXT_LEN", "300"))

    # --- Global Result Saving Settings ---
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", os.path.join(PROJECT_ROOT_DIR, 'results'))
    SAVE_AGENT_RESULTS: bool = os.getenv("SAVE_AGENT_RESULTS", "True").lower() in ["true", "1", "yes"]
    SAVE_AGENT_INPUTS: bool = os.getenv("SAVE_AGENT_INPUTS", "False").lower() in ["true", "1", "yes"]
    SAVE_DEBUG_INFO: bool = os.getenv("SAVE_DEBUG_INFO", "True").lower() in ["true", "1", "yes"]


    def __init__(self):
        """설정 객체 초기화 시 필요한 작업 수행 (예: 디렉토리 생성)."""
        # 필요한 디렉토리 생성 (결과, 이미지 저장 등)
        dirs_to_create = [self.RESULTS_DIR, self.IMAGE_STORAGE_PATH, self.FINAL_COMIC_SAVE_DIR]
        for dir_path in dirs_to_create:
             if isinstance(dir_path, str):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except OSError as e:
                     # 앱 초기 단계이므로 print 사용
                     print(f"Warning: Failed to create directory '{dir_path}'. Error: {e}")

# --- 애플리케이션 전체에서 사용할 단일 설정 인스턴스 생성 ---
settings = Settings()