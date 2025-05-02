# app/nodes/09_opinion_summarizer_node.py

import asyncio
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 프로젝트 구성 요소 임포트 ---
from app.config.settings import settings            # 설정 객체 (재시도 횟수 등 참조)
from app.services.llm_server_client_v2 import LLMService # 실제 LLM 서비스 클라이언트
from datetime import datetime, timezone, timedelta
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# --- 로거 설정 ---
logger = get_logger("OpinionSummarizerNode")

class OpinionSummarizerNode:
    """
    (Refactored) 정리된 의견들을 요약합니다.
    1. LLM을 사용하여 스탠스(찬성/반대/중립) 분류.
    2. LLM을 사용하여 감성 분포 분석.
    3. LLM을 사용하여 스탠스와 감성을 통합한 최종 요약 생성.
    - LLMService를 사용하여 모든 LLM 호출 수행.
    - 설정은 상태의 config 딕셔너리에서 로드.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["opinions_clean", "topic_analysis", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["opinion_summaries", "processing_stats", "error_message"]

    # LLMService 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        llm_client: LLMService,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        self.llm_client = llm_client
        # self.langsmith_service = langsmith_service
        # 분류 레이블 정의
        self.stance_labels = ["Pro", "Con", "Neutral"]
        self.sentiment_labels = ["Positive", "Negative", "Neutral"]
        logger.info("OpinionSummarizerNode initialized with LLMService.")

    # --- LLM 호출 래퍼 (재시도 적용) ---
    # 이전 08_news_summarizer_node.py와 동일한 래퍼 사용
    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float, max_tokens: int,
                                   trace_id: Optional[str] = None, **kwargs) -> str:
        """(Refactored) LLMService.generate_text를 재시도 로직과 함께 호출"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Calling LLMService (Temp: {temperature}, MaxTokens: {max_tokens})...")

        result = await self.llm_client.generate_text(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
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
    def _create_stance_prompt_en(self, opinion_text: str, topic: str, max_text_len: int) -> str:
        """스탠스 분류 프롬프트 생성 (텍스트 길이 제한 적용)"""
        truncated_text = opinion_text[:max_text_len] + ("..." if len(opinion_text) > max_text_len else "")
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

    async def _classify_stance(self, opinion_text: str, topic: str, config: Dict, trace_id: Optional[str]) -> str:
        """LLM을 사용하여 단일 의견의 스탠스 분류"""
        if not opinion_text: return "Neutral"

        # 설정 로드
        llm_model = config.get("llm_model", "default_model")
        temperature = config.get("llm_temperature_stance", 0.1)
        max_tokens = config.get("max_tokens_stance", 10)
        max_text_len = config.get("max_opinion_text_len", 500)

        prompt = self._create_stance_prompt_en(opinion_text, topic, max_text_len)
        try:
            response = await self._call_llm_with_retry(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, trace_id=trace_id
            )
            # 응답 파싱 (대소문자 및 공백 처리)
            response_clean = response.strip().capitalize()
            if response_clean in self.stance_labels:
                return response_clean
            else:
                logger.warning(f"[{trace_id}] Unexpected stance LLM response: '{response}'. Defaulting to Neutral.")
                return "Neutral"
        except Exception as e:
            logger.error(f"[{trace_id}] Failed to classify stance for text fragment '{opinion_text[:50]}...': {e}")
            return "Neutral" # 오류 시 기본값

    # --- 감성 분석 ---
    def _create_sentiment_prompt_en(self, opinion_text: str, max_text_len: int) -> str:
        """감성 분석 프롬프트 생성 (텍스트 길이 제한 적용)"""
        truncated_text = opinion_text[:max_text_len] + ("..." if len(opinion_text) > max_text_len else "")
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

    async def _analyze_sentiment(self, opinion_text: str, config: Dict, trace_id: Optional[str]) -> str:
        """LLM을 사용하여 단일 의견의 감성 분석"""
        if not opinion_text: return "Neutral"

        # 설정 로드
        llm_model = config.get("llm_model", "default_model")
        temperature = config.get("llm_temperature_sentiment", 0.1)
        max_tokens = config.get("max_tokens_sentiment", 10)
        max_text_len = config.get("max_opinion_text_len", 500)

        prompt = self._create_sentiment_prompt_en(opinion_text, max_text_len)
        try:
            response = await self._call_llm_with_retry(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, trace_id=trace_id
            )
            # 응답 파싱
            response_clean = response.strip().capitalize()
            if response_clean in self.sentiment_labels:
                return response_clean
            else:
                logger.warning(f"[{trace_id}] Unexpected sentiment LLM response: '{response}'. Defaulting to Neutral.")
                return "Neutral"
        except Exception as e:
            logger.error(f"[{trace_id}] Failed to analyze sentiment for text fragment '{opinion_text[:50]}...': {e}")
            return "Neutral"

    # --- 통합 요약 생성 ---
    def _format_stance_clusters_for_prompt(self, stance_clusters: Dict[str, List[str]]) -> str:
        """요약 프롬프트용 스탠스 클러스터 포맷팅 (이전과 동일)"""
        formatted_string = ""
        for stance in self.stance_labels:
            texts = stance_clusters.get(stance, [])
            if texts:
                formatted_string += f"\n[{stance} Opinions]\n"
                for text in texts[:3]: # 최대 3개 예시 포함
                    preview = text[:150] + "..." if len(text) > 150 else text
                    formatted_string += f"- {preview}\n"
            else:
                formatted_string += f"\n[{stance} Opinions]\n- No specific examples provided.\n"
        return formatted_string.strip()

    def _format_sentiment_for_prompt(self, sentiment_distribution: Dict[str, float]) -> str:
        """요약 프롬프트용 감성 분포 포맷팅 (이전과 동일)"""
        if not sentiment_distribution: return "Sentiment data not available."
        return ", ".join([f"{label}: {distribution * 100:.0f}%"
                          for label, distribution in sentiment_distribution.items()])

    def _create_summary_prompt_en(self, topic: str, formatted_stances: str, formatted_sentiment: str) -> str:
        """최종 의견 요약 생성 프롬프트 (이전과 동일)"""
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
                                      sentiment_distribution: Dict[str, float], config: Dict,
                                      trace_id: Optional[str]) -> str:
        """LLM을 사용하여 최종 의견 요약 생성"""
        formatted_stances = self._format_stance_clusters_for_prompt(stance_clusters)
        formatted_sentiment = self._format_sentiment_for_prompt(sentiment_distribution)

        # 설정 로드
        llm_model = config.get("llm_model", "default_model")
        temperature = config.get("llm_temperature_opinion_summarize", 0.4)
        max_tokens = config.get("max_tokens_opinion_summarize", 400)

        prompt = self._create_summary_prompt_en(topic, formatted_stances, formatted_sentiment)
        try:
            summary = await self._call_llm_with_retry(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, trace_id=trace_id
            )
            logger.info(f"[{trace_id}] Generated final opinion summary (Length: {len(summary)}).")
            return summary # _call_llm_with_retry 에서 strip() 처리됨
        except Exception as e:
            # 요약 생성 실패 시 오류 메시지 반환 또는 기본 텍스트 반환 결정
            logger.exception(f"[{trace_id}] Failed to generate final opinion summary: {e}")
            return f"Error: Could not generate opinion summary due to an internal error."

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """의견 요약 워크플로우 실행"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing OpinionSummarizerNode...")

        # 상태 및 설정 로드
        opinions_clean = state.opinions_clean or []
        topic_analysis = state.topic_analysis or {}
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        # 입력 유효성 검사
        if not opinions_clean:
            logger.warning(f"{log_prefix} No cleaned opinions provided. Skipping.")
            # ComicState.opinion_summaries 기본값은 {} 이므로 빈 딕셔너리 반환
            return {"opinion_summaries": {}, "processing_stats": processing_stats}

        # 설정값 가져오기
        main_topic = topic_analysis.get('main_topic', 'the analyzed topic')
        llm_model = config.get("llm_model", "default_model") # LLM 호출 시 사용될 모델명 (참고용)
        concurrency_limit = config.get('summarizer_concurrency', 5) # 스탠스/감성 분석 동시성
        max_opinions_for_stance = config.get("max_opinions_for_stance", 30)
        max_opinions_for_sentiment = config.get("max_opinions_for_sentiment", 50)

        logger.info(f"{log_prefix} Summarizing {len(opinions_clean)} cleaned opinions on topic: '{main_topic}'...")

        # 분석 대상 의견 선택 (is_representative 우선, 없으면 전체 사용)
        opinions_to_analyze = [op for op in opinions_clean if op.get('is_representative')]
        if not opinions_to_analyze:
            logger.info(f"{log_prefix} No representative opinions, using all {len(opinions_clean)} cleaned opinions.")
            opinions_to_analyze = opinions_clean
        else:
            logger.info(f"{log_prefix} Using {len(opinions_to_analyze)} representative opinions.")

        # 실제 LLM 호출 대상 제한
        opinions_for_stance_call = opinions_to_analyze[:max_opinions_for_stance]
        opinions_for_sentiment_call = opinions_to_analyze[:max_opinions_for_sentiment]

        semaphore = asyncio.Semaphore(concurrency_limit)
        task_errors: List[str] = [] # 개별 작업 오류 기록

        # --- 1. 스탠스 분류 (동시 실행) ---
        logger.info(f"{log_prefix} Classifying stance for {len(opinions_for_stance_call)} opinions...")
        stance_tasks = []
        async def classify_stance_task(opinion):
            async with semaphore:
                text = opinion.get('text', '')
                stance = await self._classify_stance(text, main_topic, config, state.trace_id)
                # 스탠스 결과와 함께 원본 텍스트 (또는 ID) 반환 필요
                return stance, text, opinion.get('url', None) # url 또는 고유 ID 추가

        for opinion in opinions_for_stance_call:
            stance_tasks.append(classify_stance_task(opinion))
        stance_results = await asyncio.gather(*stance_tasks, return_exceptions=True)

        # 결과 처리: 스탠스별 텍스트 그룹화 및 카운트
        stance_clusters: Dict[str, List[str]] = defaultdict(list)
        stance_counts = Counter()
        for result in stance_results:
            if isinstance(result, tuple):
                stance, text, _ = result # url은 여기선 사용 안함
                if stance in self.stance_labels and text:
                    stance_clusters[stance].append(text) # 요약 프롬프트용 텍스트 저장
                    stance_counts[stance] += 1
            elif isinstance(result, Exception):
                logger.error(f"{log_prefix} Stance classification task failed: {result}")
                task_errors.append(f"Stance classification error: {result}")
        logger.info(f"{log_prefix} Stance classification complete. Counts: {dict(stance_counts)}")

        # --- 2. 감성 분석 (동시 실행) ---
        logger.info(f"{log_prefix} Analyzing sentiment for {len(opinions_for_sentiment_call)} opinions...")
        sentiment_tasks = []
        async def analyze_sentiment_task(opinion):
            async with semaphore:
                text = opinion.get('text', '')
                return await self._analyze_sentiment(text, config, state.trace_id)

        for opinion in opinions_for_sentiment_call:
            sentiment_tasks.append(analyze_sentiment_task(opinion))
        sentiment_results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)

        # 결과 처리: 감성 분포 계산
        sentiment_counter = Counter()
        valid_sentiment_results = 0
        for result in sentiment_results:
            if isinstance(result, str) and result in self.sentiment_labels:
                sentiment_counter[result] += 1
                valid_sentiment_results += 1
            elif isinstance(result, Exception):
                logger.error(f"{log_prefix} Sentiment analysis task failed: {result}")
                task_errors.append(f"Sentiment analysis error: {result}")

        sentiment_distribution: Dict[str, float] = {}
        if valid_sentiment_results > 0:
            for label in self.sentiment_labels:
                sentiment_distribution[label] = round(sentiment_counter[label] / valid_sentiment_results, 3)
        else: # 유효 결과 없으면 중립 100%
            sentiment_distribution = {"Positive": 0.0, "Negative": 0.0, "Neutral": 1.0}
        logger.info(f"{log_prefix} Sentiment analysis complete. Distribution: {sentiment_distribution}")

        # --- 3. 최종 요약 생성 ---
        logger.info(f"{log_prefix} Generating final opinion summary...")
        summary_text = await self._generate_final_summary(
            main_topic, stance_clusters, sentiment_distribution, config, state.trace_id
        )

        # --- 출력 포맷 구성 (ComicState.opinion_summaries 형식) ---
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
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['opinion_summarizer_node_time'] = node_processing_time
        logger.info(f"{log_prefix} OpinionSummarizerNode finished in {node_processing_time:.2f} seconds.")

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"{log_prefix} Some errors occurred during opinion analysis: {final_error_message}")

        # TODO: LangSmith 로깅

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "opinion_summaries": opinion_summaries_output,
            "processing_stats": processing_stats,
            "error_message": final_error_message # 부분 오류 요약
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}