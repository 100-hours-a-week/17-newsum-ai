# app/nodes/01_initialize_node.py (Merged Version)

import uuid
import re
from datetime import datetime, timezone
from typing import Dict, Any, List

# --- 프로젝트 구성 요소 임포트 ---
from app.config.settings import settings        # 중앙 설정 객체
from app.utils.logger import get_logger         # 로거 유틸리티
from app.workflows.state import ComicState      # 워크플로우 상태 모델

# --- 로거 설정 ---
logger = get_logger("InitializeNode")

class InitializeNode:
    """
    (Refactored & Merged) 워크플로우 실행을 초기화하고 초기 상태를 설정합니다.
    중앙 'settings' 객체에서 비-민감 설정을 로드하여 상태의 'config' 필드를 채웁니다.
    """

    def __init__(self):
        """노드 초기화 (현재는 특별한 로직 없음)"""
        pass

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """
        (Refactored & Merged) 중앙 설정을 사용하여 워크플로우 초기 상태를 설정합니다.

        Args:
            state: 입력 ComicState ('initial_query', 'trace_id' 포함 가능).

        Returns:
            ComicState 필드를 업데이트하기 위한 딕셔너리.
        """
        start_time = datetime.now(timezone.utc)
        # 상태에 trace_id가 있으면 사용, 없으면 새로 생성
        trace_id = state.trace_id or str(uuid.uuid4())
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing InitializeNode...")

        comic_id = str(uuid.uuid4())
        timestamp = start_time.isoformat()

        initial_query = state.initial_query or ""
        normalized_query = self._normalize_query(initial_query)
        logger.info(f"{log_prefix} Original query: '{initial_query}', Normalized query: '{normalized_query}'")

        # --- 중앙 settings 객체에서 설정을 읽어 config 딕셔너리 생성 ---
        # 이 config 딕셔너리는 후속 노드에서 state.config 를 통해 접근
        # 중요: API 키, 비밀번호 등 민감 정보는 포함하지 않음!
        #       서비스/도구 클라이언트가 초기화 시 settings 에서 직접 읽도록 함.
        node_config = {
            # 모델명
            "llm_model": settings.DEFAULT_LLM_MODEL,
            "image_model": settings.DEFAULT_IMAGE_MODEL,

            # LLM 온도 및 토큰 (작업별)
            "llm_temperature_analysis": settings.LLM_TEMPERATURE_ANALYSIS,
            "llm_temperature_extract": settings.LLM_TEMPERATURE_EXTRACT,
            "llm_temperature_summarize": settings.LLM_TEMPERATURE_SUMMARIZE,
            "llm_temperature_qa_gen": settings.LLM_TEMPERATURE_QA_GEN,
            "llm_temperature_qa_verify": settings.LLM_TEMPERATURE_QA_VERIFY,
            "llm_temperature_stance": settings.LLM_TEMPERATURE_STANCE,
            "llm_temperature_sentiment": settings.LLM_TEMPERATURE_SENTIMENT,
            "llm_temperature_opinion_summarize": settings.LLM_TEMPERATURE_OPINION_SUMMARIZE,
            "llm_temperature_synthesis": settings.LLM_TEMPERATURE_SYNTHESIS,
            "llm_temperature_creative": settings.LLM_TEMPERATURE_CREATIVE, # Idea/Scenario 공통 사용 가능
            "llm_temperature_scenario": settings.LLM_TEMPERATURE_SCENARIO, # Scenario 별도 온도
            "llm_temperature_scenario_eval": settings.LLM_TEMPERATURE_SCENARIO_EVAL,
            "max_tokens_extract": settings.MAX_TOKENS_EXTRACT,
            "max_tokens_summarize": settings.MAX_TOKENS_SUMMARIZE,
            "max_tokens_qa_gen": settings.MAX_TOKENS_QA_GEN,
            "max_tokens_qa_verify": settings.MAX_TOKENS_QA_VERIFY,
            "max_tokens_stance": settings.MAX_TOKENS_STANCE,
            "max_tokens_sentiment": settings.MAX_TOKENS_SENTIMENT,
            "max_tokens_opinion_summarize": settings.MAX_TOKENS_OPINION_SUMMARIZE,
            "max_tokens_synthesis": settings.MAX_TOKENS_SYNTHESIS,
            "max_tokens_idea": settings.MAX_TOKENS_IDEA,
            "max_tokens_scenario": settings.MAX_TOKENS_SCENARIO,
            "max_tokens_scenario_eval": settings.MAX_TOKENS_SCENARIO_EVAL,

            # 텍스트 길이 제한
            "max_article_text_len": settings.MAX_ARTICLE_TEXT_LEN,
            "max_opinion_text_len": settings.MAX_OPINION_TEXT_LEN,
            "max_context_len_scenario": settings.MAX_CONTEXT_LEN_SCENARIO,
            "max_image_prompt_len": settings.MAX_IMAGE_PROMPT_LEN,
            "max_alt_text_len": settings.MAX_ALT_TEXT_LEN,

            # 기능 플래그 및 언어 설정
            "translation_enabled": settings.TRANSLATION_ENABLED,
            "target_language": settings.TARGET_TRANSLATION_LANGUAGE,
            "source_language": settings.SOURCE_LANGUAGE, # Optional
            "language_filter": settings.LANGUAGE_FILTER,
            "enable_scenario_evaluation": settings.ENABLE_SCENARIO_EVALUATION,
            "upload_to_s3": settings.UPLOAD_TO_S3,

            # 수집/처리 관련 파라미터
            "max_articles": settings.MAX_ARTICLES_TO_SCRAPE,
            "max_opinions_per_platform": settings.MAX_OPINIONS_PER_PLATFORM_SAMPLING, # 이름 변경 및 딕셔너리 사용
            "min_opinions_per_platform": settings.MIN_OPINIONS_PER_PLATFORM_SAMPLING, # 이름 변경 및 딕셔너리 사용
            "max_total_opinions_target": settings.MAX_TOTAL_OPINIONS_TARGET,
            "search_result_count": settings.SEARCH_RESULT_COUNT,
            "target_urls_per_keyword": settings.TARGET_URLS_PER_KEYWORD,
            "trusted_news_domains": settings.TRUSTED_NEWS_DOMAINS,
            "target_community_domains": settings.TARGET_COMMUNITY_DOMAINS,
            "reddit_target_subreddits": settings.REDDIT_TARGET_SUBREDDITS,
            "youtube_opinion_max_results": settings.YOUTUBE_OPINION_MAX_RESULTS,
            "twitter_opinion_max_results": settings.TWITTER_OPINION_MAX_RESULTS,
            "reddit_opinion_max_results": settings.REDDIT_OPINION_MAX_RESULTS,
            "blog_opinion_max_results": settings.BLOG_OPINION_MAX_RESULTS,
            "community_opinion_max_results": settings.COMMUNITY_OPINION_MAX_RESULTS,

            # 스크래핑 설정
            "scraper_user_agent": settings.SCRAPER_USER_AGENT,
            "scraper_http_timeout": settings.SCRAPER_HTTP_TIMEOUT,
            "scraper_concurrency": settings.SCRAPER_CONCURRENCY,
            "opinion_scraper_concurrency": settings.OPINION_SCRAPER_CONCURRENCY,
            "min_extracted_text_length": settings.MIN_EXTRACTED_TEXT_LENGTH,
            "min_langdetect_text_length": settings.MIN_LANGDETECT_TEXT_LENGTH,
            "selenium_grid_url": settings.SELENIUM_GRID_URL,
            "selenium_headless": settings.SELENIUM_HEADLESS,
            "selenium_retry_attempts": settings.SELENIUM_RETRY_ATTEMPTS,
            "selenium_retry_wait_seconds": settings.SELENIUM_RETRY_WAIT_SECONDS,
            "scraper_min_delay_ms": settings.SCRAPER_MIN_DELAY_MS,
            "scraper_max_delay_ms": settings.SCRAPER_MAX_DELAY_MS,
            "scraper_use_proxy": settings.SCRAPER_USE_PROXY,
            "scraper_proxy_url": settings.SCRAPER_PROXY_URL,
            "scraper_rotate_ua": settings.SCRAPER_ROTATE_UA,

             # 분석/필터링 설정
            "spam_keywords": settings.SPAM_KEYWORDS,
            "spam_max_url_count": settings.SPAM_MAX_URL_COUNT,
            "spam_max_uppercase_ratio": settings.SPAM_MAX_UPPERCASE_RATIO,
            "simhash_threshold": settings.SIMHASH_THRESHOLD,
            "simhash_token_width": settings.SIMHASH_TOKEN_WIDTH,
            "kmeans_default_clusters": settings.KMEANS_DEFAULT_CLUSTERS,
            "kmeans_min_samples": settings.KMEANS_MIN_SAMPLES,
            "tfidf_max_features": settings.TFIDF_MAX_FEATURES,
            "tfidf_stop_words": settings.TFIDF_STOP_WORDS,
            "tfidf_min_df": settings.TFIDF_MIN_DF,
            "tfidf_max_df": settings.TFIDF_MAX_DF,
            "tfidf_ngram_range": (settings.TFIDF_NGRAM_RANGE_MIN, settings.TFIDF_NGRAM_RANGE_MAX), # 튜플로 전달
            "kmeans_n_init": settings.KMEANS_N_INIT,
            "kmeans_max_no_improvement": settings.KMEANS_MAX_NO_IMPROVEMENT,

            # 타임아웃 및 동시성 (일반)
            "http_timeout": settings.HTTP_TIMEOUT, # 일반 HTTP 호출용 (aiohttp 등)
            "translator_concurrency": settings.TRANSLATOR_CONCURRENCY,
            "summarizer_concurrency": settings.SUMMARIZER_CONCURRENCY, # News/Opinion 공통 사용 가정

            # 임계값 (Evaluation & Decision)
            "feqa_threshold": settings.FEQA_THRESHOLD,
            "evaluation_thresholds": settings.EVALUATION_THRESHOLDS,
            "decision_logic_thresholds": settings.DECISION_LOGIC_THRESHOLDS,
            "bertscore_lang": settings.BERTSCORE_LANG,

            # 이미지/결과 관련 설정
            "image_default_style": settings.IMAGE_DEFAULT_STYLE,
            "image_height": settings.IMAGE_HEIGHT,
            "image_width": settings.IMAGE_WIDTH,
            "image_negative_prompt": settings.IMAGE_NEGATIVE_PROMPT,
            "image_style_preset": settings.IMAGE_STYLE_PRESET,
            "default_font_path": settings.DEFAULT_FONT_PATH,
            "final_comic_save_dir": settings.FINAL_COMIC_SAVE_DIR,
            "final_image_format": settings.FINAL_IMAGE_FORMAT,
            "final_image_width": settings.FINAL_IMAGE_WIDTH,
            "final_image_quality": settings.FINAL_IMAGE_QUALITY,
            "text_overlay_font_size_ratio": settings.TEXT_OVERLAY_FONT_SIZE_RATIO,
            "text_overlay_color": settings.TEXT_OVERLAY_COLOR,

            # 보고서 관련 설정
            "template_dir": settings.TEMPLATE_DIR,
            "progress_report_template_a_filename": settings.PROGRESS_REPORT_TEMPLATE_A_FILENAME,
            "progress_report_template_b_filename": settings.PROGRESS_REPORT_TEMPLATE_B_FILENAME,
            "trends_report_top_n": settings.TRENDS_REPORT_TOP_N,

            # 결과 저장 설정
            "save_agent_results": settings.SAVE_AGENT_RESULTS,
            "save_agent_inputs": settings.SAVE_AGENT_INPUTS,
            "save_debug_info": settings.SAVE_DEBUG_INFO,
            "results_dir": settings.RESULTS_DIR,

             # 재시도 횟수 (참고용, 실제 재시도는 클라이언트/도구가 settings 참조)
             "llm_api_retries": settings.LLM_API_RETRIES,
             "image_api_retries": settings.IMAGE_API_RETRIES,
             "image_download_retries": settings.IMAGE_DOWNLOAD_RETRIES,
             "selenium_retry_attempts": settings.SELENIUM_RETRY_ATTEMPTS,
             # "tool_retry_attempts": settings.TOOL_RETRY_ATTEMPTS, # TOOL_RETRY_ATTEMPTS 대신 구체적인 retry 사용
        }
        logger.info(f"{log_prefix} Generated node config dictionary with {len(node_config)} keys from settings.")
        # logger.debug(f"{log_prefix} Config details: {node_config}") # 너무 길 수 있으므로 주석 처리

        # --- 초기 상태 필드 설정 ---
        used_links: List[Dict[str, str]] = []
        processing_stats: Dict[str, float] = {}
        end_time = datetime.now(timezone.utc)
        processing_stats['initialize_node_time'] = (end_time - start_time).total_seconds()

        logger.info(f"{log_prefix} Initialization complete: comic_id={comic_id}")

        # --- ComicState 업데이트를 위한 결과 반환 ---
        # 상태 모델에 정의된 모든 필드를 포함하여 반환 (기본값 또는 초기값)
        update_data: Dict[str, Any] = {
            "comic_id": comic_id,
            "config": node_config,
            "trace_id": trace_id,
            "timestamp": timestamp,
            "used_links": used_links,
            "initial_query": normalized_query,
            "processing_stats": processing_stats,
            # 나머지 필드들은 ComicState 모델의 기본값(빈 리스트/딕셔너리, None)으로 시작
            # 명시적으로 설정하지 않아도 LangGraph가 처리하지만 가독성을 위해 추가 가능
            "topic_analysis": {},
            "search_keywords": [],
            "fact_urls": [],
            "opinion_urls": [],
            "articles": [],
            "opinions_raw": [],
            "opinions_clean": [],
            "news_summaries": [],
            "opinion_summaries": {},
            "final_summary": None,
            "evaluation_metrics": {},
            "decision": None,
            "trend_scores": [],
            "comic_ideas": [],
            "chosen_idea": None,
            "scenarios": [],
            "scenario_prompt": None,
            "image_urls": [],
            "translated_text": None,
            "final_comic": {},
            "error_message": None
        }
        # 반환 전, ComicState 모델에 정의된 필드만 포함하는지 확인 (안전장치)
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}


    def _normalize_query(self, query: str) -> str:
        """쿼리 문자열 정규화"""
        if not query: return ""
        return re.sub(r'\s+', ' ', query.strip())