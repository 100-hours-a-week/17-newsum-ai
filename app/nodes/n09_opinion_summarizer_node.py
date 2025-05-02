# app/nodes/09_opinion_summarizer_node.py (Improved Version)

import asyncio
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 기본값 참조용
from app.services.llm_server_client_v2 import LLMService # LLM 서비스 클라이언트
from datetime import datetime, timezone, timedelta
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# 로거 설정
logger = get_logger(__name__)

class OpinionSummarizerNode:
    """
    정리된 의견들을 요약합니다 (스탠스 분류, 감성 분석, 최종 요약).
    - LLMService를 사용하여 모든 LLM 호출 수행.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["opinions_clean", "topic_analysis", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["opinion_summaries", "processing_stats", "error_message"]

    # LLMService 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        llm_client: LLMService,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        if not llm_client: raise ValueError("LLMService is required for OpinionSummarizerNode")
        self.llm_client = llm_client
        # self.langsmith_service = langsmith_service
        # 분류 레이블 정의 (고정값)
        self.stance_labels = ["Pro", "Con", "Neutral"]
        self.sentiment_labels = ["Positive", "Negative", "Neutral"]
        logger.info("OpinionSummarizerNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        # LLM 호출 파라미터
        self.llm_temp_stance = float(config.get("llm_temperature_stance", settings.DEFAULT_LLM_TEMP_STANCE))
        self.max_tokens_stance = int(config.get("max_tokens_stance", settings.DEFAULT_MAX_TOKENS_STANCE))
        self.llm_temp_sentiment = float(config.get("llm_temperature_sentiment", settings.DEFAULT_LLM_TEMP_SENTIMENT))
        self.max_tokens_sentiment = int(config.get("max_tokens_sentiment", settings.DEFAULT_MAX_TOKENS_SENTIMENT))
        self.llm_temp_opinion_summarize = float(config.get("llm_temperature_opinion_summarize", settings.DEFAULT_LLM_TEMP_OP_SUMMARIZE))
        self.max_tokens_opinion_summarize = int(config.get("max_tokens_opinion_summarize", settings.DEFAULT_MAX_TOKENS_OP_SUMMARIZE))
        # 텍스트 길이 제한
        self.max_opinion_text_len = int(config.get("max_opinion_text_len", settings.MAX_OPINION_TEXT_LEN))
        # 처리 제한 및 동시성
        self.max_opinions_for_stance = int(config.get("max_opinions_for_stance", settings.DEFAULT_MAX_OPINIONS_STANCE))
        self.max_opinions_for_sentiment = int(config.get("max_opinions_for_sentiment", settings.DEFAULT_MAX_OPINIONS_SENTIMENT))
        self.concurrency_limit = int(config.get('summarizer_concurrency', settings.DEFAULT_SUMMARIZER_CONCURRENCY)) # News Summarizer와 공유 가능

        logger.debug(f"Runtime config loaded. Max Opinions (Stance/Sent): {self.max_opinions_for_stance}/{self.max_opinions_for_sentiment}, Concurrency: {self.concurrency_limit}")
        logger.debug(f"LLM Temps (St/Se/Sum): {self.llm_temp_stance}/{self.llm_temp_sentiment}/{self.llm_temp_opinion_summarize}")
        logger.debug(f"Max Tokens (St/Se/Sum): {self.max_tokens_stance}/{self.max_tokens_sentiment}/{self.max_tokens_opinion_summarize}")
        logger.debug(f"Max Opinion Text Length: {self.max_opinion_text_len}")


    # --- LLM 호출 래퍼 (재시도 적용) ---
    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float, max_tokens: int,
                                   trace_id: Optional[str] = None, **kwargs) -> str:
        """LLMService.generate_text를 재시도 로직과 함께 호출"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Calling LLMService (Temp: {temperature}, MaxTokens: {max_tokens})...")

        result = await self.llm_client.generate_text(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        if "error" in result:
            error_msg = result['error']
            logger.error(f"{log_prefix} LLMService call failed: {error_msg}")
            raise RuntimeError(f"LLMService error: {error_msg}")
        elif "generated_text" not in result or not isinstance(result.get("generated_text"), str) or not result["generated_text"].strip():
            logger.error(f"{log_prefix} LLMService returned invalid or empty text. Response: {result}")
            raise ValueError("LLMService returned invalid or empty text")
        else:
            logger.debug(f"{log_prefix} LLMService call successful.")
            return result["generated_text"].strip()

    # --- 스탠스 분류 ---
    def _create_stance_prompt_en(self, opinion_text: str, topic: str) -> str:
        """스탠스 분류 프롬프트 생성 (길이 제한 적용)"""
        truncated_text = opinion_text[:self.max_opinion_text_len] + ("..." if len(opinion_text) > self.max_opinion_text_len else "")
        # 프롬프트 내용은 이전과 동일
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a text classification assistant. Classify the stance of the provided opinion text towards the given topic. Respond with only ONE of the following labels: Pro, Con, or Neutral.<|eot_id|><|start_header_id|>user<|end_header_id|>
Classify the stance of the following opinion text regarding the topic: "{topic}".

[Opinion Text]
{truncated_text}

[Instructions]
- Read the text and determine the author's overall stance towards the topic.
- Stance options are:
    - Pro: Supports or agrees with the topic/subject.
    - Con: Opposes or disagrees with the topic/subject.
    - Neutral: Presents a balanced view, is objective, or expresses no clear stance.
- Respond with ONLY ONE word: Pro, Con, or Neutral.

[Stance Classification]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    async def _classify_stance(self, opinion_text: str, topic: str, trace_id: Optional[str]) -> str:
        """LLM을 사용하여 단일 의견의 스탠스 분류"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not opinion_text: return "Neutral"

        prompt = self._create_stance_prompt_en(opinion_text, topic)
        try:
            response = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_stance,
                max_tokens=self.max_tokens_stance,
                trace_id=trace_id
            )
            response_clean = response.strip().capitalize()
            if response_clean in self.stance_labels:
                return response_clean
            else:
                logger.warning(f"{log_prefix} Unexpected stance LLM response: '{response}'. Defaulting to Neutral.")
                return "Neutral"
        except RetryError as e:
             logger.error(f"{log_prefix} Failed to classify stance after retries for text '{opinion_text[:30]}...': {e}")
             return "Neutral" # 오류 시 기본값
        except Exception as e:
            logger.error(f"{log_prefix} Error classifying stance for text '{opinion_text[:30]}...': {e}", exc_info=True)
            return "Neutral"

    # --- 감성 분석 ---
    def _create_sentiment_prompt_en(self, opinion_text: str) -> str:
        """감성 분석 프롬프트 생성 (길이 제한 적용)"""
        truncated_text = opinion_text[:self.max_opinion_text_len] + ("..." if len(opinion_text) > self.max_opinion_text_len else "")
        # 프롬프트 내용은 이전과 동일
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a text classification assistant. Analyze the sentiment expressed in the provided text. Respond with only ONE of the following labels: Positive, Negative, or Neutral.<|eot_id|><|start_header_id|>user<|end_header_id|>
Analyze the overall sentiment of the following opinion text.

[Opinion Text]
{truncated_text}

[Instructions]
- Determine the dominant sentiment: Positive, Negative, or Neutral.
- Respond with ONLY ONE word: Positive, Negative, or Neutral.

[Sentiment Analysis]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    async def _analyze_sentiment(self, opinion_text: str, trace_id: Optional[str]) -> str:
        """LLM을 사용하여 단일 의견의 감성 분석"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not opinion_text: return "Neutral"

        prompt = self._create_sentiment_prompt_en(opinion_text)
        try:
            response = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_sentiment,
                max_tokens=self.max_tokens_sentiment,
                trace_id=trace_id
            )
            response_clean = response.strip().capitalize()
            if response_clean in self.sentiment_labels:
                return response_clean
            else:
                logger.warning(f"{log_prefix} Unexpected sentiment LLM response: '{response}'. Defaulting to Neutral.")
                return "Neutral"
        except RetryError as e:
             logger.error(f"{log_prefix} Failed to analyze sentiment after retries for text '{opinion_text[:30]}...': {e}")
             return "Neutral"
        except Exception as e:
            logger.error(f"{log_prefix} Error analyzing sentiment for text '{opinion_text[:30]}...': {e}", exc_info=True)
            return "Neutral"

    # --- 통합 요약 생성 ---
    def _format_stance_clusters_for_prompt(self, stance_clusters: Dict[str, List[str]]) -> str:
        """요약 프롬프트용 스탠스 클러스터 포맷팅"""
        formatted_string = ""
        for stance in self.stance_labels: # Pro, Con, Neutral 순서 보장
            texts = stance_clusters.get(stance, [])
            if texts:
                formatted_string += f"\n[{stance} Opinions]\n"
                # 최대 3개 예시 포함, 길이 제한
                for text in texts[:3]:
                    preview = text[:150] + "..." if len(text) > 150 else text
                    formatted_string += f"- {preview}\n"
            else:
                formatted_string += f"\n[{stance} Opinions]\n- No specific examples provided.\n"
        return formatted_string.strip()

    def _format_sentiment_for_prompt(self, sentiment_distribution: Dict[str, float]) -> str:
        """요약 프롬프트용 감성 분포 포맷팅"""
        if not sentiment_distribution: return "Sentiment data not available."
        return ", ".join([f"{label}: {distribution * 100:.1f}%" # 소수점 1자리
                          for label, distribution in sentiment_distribution.items()])

    def _create_summary_prompt_en(self, topic: str, formatted_stances: str, formatted_sentiment: str) -> str:
        """최종 의견 요약 생성 프롬프트"""
        # 프롬프트 내용은 이전과 동일
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an insightful analyst tasked with summarizing public opinion on a given topic. Synthesize the provided stance-based opinion examples and overall sentiment distribution into a concise, balanced overview.<|eot_id|><|start_header_id|>user<|end_header_id|>
Synthesize the following classified opinions and sentiment data regarding the topic "{topic}" into a comprehensive summary of approximately 150-200 words.

[Opinion Examples by Stance]
{formatted_stances}

[Overall Sentiment Distribution]
{formatted_sentiment}

[Instructions]
- Write a summary of about 150-200 words.
- Reflect the main arguments for each major stance (Pro, Con, Neutral) based on the examples, if available.
- Mention the overall sentiment distribution.
- Analyze and synthesize the information, don't just list the points.
- Maintain a neutral and objective tone in the summary itself.

[Opinion Summary]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    async def _generate_final_summary(self, topic: str, stance_clusters: Dict[str, List[str]],
                                      sentiment_distribution: Dict[str, float],
                                      trace_id: Optional[str]) -> str:
        """LLM을 사용하여 최종 의견 요약 생성"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        formatted_stances = self._format_stance_clusters_for_prompt(stance_clusters)
        formatted_sentiment = self._format_sentiment_for_prompt(sentiment_distribution)

        prompt = self._create_summary_prompt_en(topic, formatted_stances, formatted_sentiment)
        try:
            summary = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_opinion_summarize,
                max_tokens=self.max_tokens_opinion_summarize,
                trace_id=trace_id
            )
            logger.info(f"{log_prefix} Generated final opinion summary (Length: {len(summary)}).")
            return summary
        except RetryError as e:
            logger.error(f"{log_prefix} Failed to generate final opinion summary after retries: {e}")
            return f"Error: Could not generate opinion summary due to LLM failure."
        except Exception as e:
            logger.exception(f"{log_prefix} Failed to generate final opinion summary: {e}")
            return f"Error: Could not generate opinion summary due to an internal error."

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """의견 요약 워크플로우 실행"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing OpinionSummarizerNode...")

        opinions_clean = state.opinions_clean or []
        topic_analysis = state.topic_analysis or {}
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        if not opinions_clean:
            logger.warning(f"{log_prefix} No cleaned opinions provided. Skipping.")
            processing_stats['opinion_summarizer_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"opinion_summaries": {}, "processing_stats": processing_stats} # 빈 딕셔너리 반환

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        # --------------------------

        main_topic = topic_analysis.get('main_topic', 'the analyzed topic')
        logger.info(f"{log_prefix} Summarizing {len(opinions_clean)} cleaned opinions on topic: '{main_topic}'...")

        # 분석 대상 의견 선택 (is_representative 우선)
        opinions_to_analyze = [op for op in opinions_clean if op.get('is_representative')]
        if not opinions_to_analyze:
            logger.info(f"{log_prefix} No representative opinions found, using all {len(opinions_clean)} cleaned opinions for analysis.")
            opinions_to_analyze = opinions_clean
        else:
            logger.info(f"{log_prefix} Using {len(opinions_to_analyze)} representative opinions for analysis.")

        # 실제 LLM 호출 대상 제한
        opinions_for_stance_call = opinions_to_analyze[:self.max_opinions_for_stance]
        opinions_for_sentiment_call = opinions_to_analyze[:self.max_opinions_for_sentiment]
        logger.info(f"{log_prefix} Calling LLM for stance ({len(opinions_for_stance_call)}) and sentiment ({len(opinions_for_sentiment_call)}).")

        task_errors: List[str] = [] # 개별 작업 오류 기록

        # --- 1. 스탠스 분류 (동시 실행) ---
        stance_tasks = []
        async def classify_stance_task(opinion):
            async with semaphore:
                text = opinion.get('text', '')
                stance = await self._classify_stance(text, main_topic, trace_id)
                return stance, text # 원본 텍스트 반환

        for opinion in opinions_for_stance_call: stance_tasks.append(classify_stance_task(opinion))
        stance_results = await asyncio.gather(*stance_tasks, return_exceptions=True)

        stance_clusters: Dict[str, List[str]] = defaultdict(list)
        stance_counts = Counter()
        for result in stance_results:
            if isinstance(result, tuple):
                stance, text = result
                if stance in self.stance_labels and text:
                    stance_clusters[stance].append(text) # 요약 프롬프트용 텍스트 저장
                    stance_counts[stance] += 1
            elif isinstance(result, Exception):
                err_msg = f"Stance classification task failed: {result}"
                logger.error(f"{log_prefix} {err_msg}")
                task_errors.append(err_msg)
        logger.info(f"{log_prefix} Stance classification complete. Counts: {dict(stance_counts)}")

        # --- 2. 감성 분석 (동시 실행) ---
        sentiment_tasks = []
        async def analyze_sentiment_task(opinion):
            async with semaphore:
                text = opinion.get('text', '')
                return await self._analyze_sentiment(text, trace_id)

        for opinion in opinions_for_sentiment_call: sentiment_tasks.append(analyze_sentiment_task(opinion))
        sentiment_results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)

        sentiment_counter = Counter()
        valid_sentiment_results = 0
        for result in sentiment_results:
            if isinstance(result, str) and result in self.sentiment_labels:
                sentiment_counter[result] += 1
                valid_sentiment_results += 1
            elif isinstance(result, Exception):
                err_msg = f"Sentiment analysis task failed: {result}"
                logger.error(f"{log_prefix} {err_msg}")
                task_errors.append(err_msg)

        sentiment_distribution: Dict[str, float] = {}
        if valid_sentiment_results > 0:
            for label in self.sentiment_labels:
                sentiment_distribution[label] = round(sentiment_counter[label] / valid_sentiment_results, 3)
        else:
            logger.warning(f"{log_prefix} No valid sentiment results obtained. Setting default distribution.")
            sentiment_distribution = {"Positive": 0.0, "Negative": 0.0, "Neutral": 1.0} # 기본값
        logger.info(f"{log_prefix} Sentiment analysis complete. Distribution: {sentiment_distribution}")

        # --- 3. 최종 요약 생성 ---
        summary_text = await self._generate_final_summary(
            main_topic, stance_clusters, sentiment_distribution, trace_id
        )

        # --- 출력 포맷 구성 ---
        stance_clusters_output = [
            {"stance": stance, "representative_texts": texts[:5]} # 대표 텍스트 5개 포함
            for stance, texts in stance_clusters.items() if texts
        ]
        opinion_summaries_output = {
            "summary_text": summary_text,
            "stance_clusters": stance_clusters_output,
            "sentiment_distribution": sentiment_distribution
        }

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['opinion_summarizer_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} OpinionSummarizerNode finished in {processing_stats['opinion_summarizer_node_time']:.2f} seconds.")

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"{log_prefix} Some errors occurred during opinion analysis: {final_error_message}")

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "opinion_summaries": opinion_summaries_output,
            "processing_stats": processing_stats,
            "error_message": final_error_message # 부분 오류 요약
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}