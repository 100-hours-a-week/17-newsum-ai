# app/nodes/10_synthesis_summarizer_node.py (Improved Version)

import asyncio
import re
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

class SynthesisSummarizerNode:
    """
    뉴스 요약과 의견 요약을 종합하여 최종 요약을 생성합니다.
    - LLMService를 사용하여 통합 요약 생성.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["news_summaries", "opinion_summaries", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["final_summary", "processing_stats", "error_message"]

    # LLMService 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        llm_client: LLMService,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        if not llm_client: raise ValueError("LLMService is required for SynthesisSummarizerNode")
        self.llm_client = llm_client
        # self.langsmith_service = langsmith_service
        logger.info("SynthesisSummarizerNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        # 뉴스 요약 선택 기준
        self.feqa_threshold = float(config.get("feqa_threshold", settings.DEFAULT_FEQA_THRESHOLD))
        self.max_news_summaries = int(config.get("max_news_summaries_for_synthesis", settings.DEFAULT_MAX_SUMMARIES_SYNTHESIS))
        # LLM 호출 파라미터
        self.llm_temp_synthesis = float(config.get("llm_temperature_synthesis", settings.DEFAULT_LLM_TEMP_SYNTHESIS))
        self.max_tokens_synthesis = int(config.get("max_tokens_synthesis", settings.DEFAULT_MAX_TOKENS_SYNTHESIS))
        # 목표 단어 수 (프롬프트용)
        self.target_word_count = int(config.get("synthesis_target_word_count", settings.DEFAULT_SYNTHESIS_WORD_COUNT))

        logger.debug(f"Runtime config loaded. FEQA Threshold: {self.feqa_threshold}, Max News Summaries: {self.max_news_summaries}")
        logger.debug(f"LLM Temp: {self.llm_temp_synthesis}, Max Tokens: {self.max_tokens_synthesis}, Target Words: {self.target_word_count}")


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

    # --- 입력 준비 ---
    def _prepare_news_input(self, news_summaries: List[Dict[str, Any]], trace_id: Optional[str]) -> str:
        """통합 요약 프롬프트를 위한 뉴스 요약 입력 준비"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not news_summaries: return "No news summaries available."

        # FEQA 점수 기준으로 고품질 요약 필터링 및 선택
        high_quality_summaries = sorted(
            [s for s in news_summaries if s.get("feqa_score", 0.0) >= self.feqa_threshold and s.get("summary_text")],
            key=lambda x: x.get("feqa_score", 0.0), reverse=True
        )[:self.max_news_summaries]

        selected_texts = [s['summary_text'] for s in high_quality_summaries]

        if not selected_texts:
            logger.warning(f"{log_prefix} No news summaries met FEQA threshold {self.feqa_threshold}. Using highest available (up to {self.max_news_summaries}).")
            all_valid_summaries = sorted(
                [s for s in news_summaries if s.get("summary_text")],
                key=lambda x: x.get("feqa_score", 0.0), reverse=True
            )[:self.max_news_summaries]
            selected_texts = [s['summary_text'] for s in all_valid_summaries]

        if not selected_texts: return "No usable news summaries available."

        logger.info(f"{log_prefix} Prepared {len(selected_texts)} news summaries for synthesis input.")
        return "\n\n".join(f"[News Summary {i+1}]\n{text}" for i, text in enumerate(selected_texts))

    def _prepare_opinion_input(self, opinion_summaries: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        """통합 요약 프롬프트를 위한 의견 요약 및 감성 정보 준비"""
        if not opinion_summaries:
            return "No opinion summary available.", "Sentiment data not available."

        opinion_summary_text = opinion_summaries.get("summary_text", "No opinion summary text provided.")
        sentiment_distribution = opinion_summaries.get("sentiment_distribution", {})

        if sentiment_distribution and isinstance(sentiment_distribution, dict):
            sentiment_parts = [f"{label}: {dist*100:.1f}%" # 소수점 1자리
                               for label, dist in sentiment_distribution.items() if isinstance(dist, (int, float))]
            sentiment_text = ", ".join(sentiment_parts) if sentiment_parts else "Sentiment data format error."
        else:
            sentiment_text = "Sentiment data not available or invalid format."

        return opinion_summary_text, sentiment_text

    # --- 통합 요약 프롬프트 및 생성 ---
    def _create_synthesis_prompt_en(self, news_content: str, opinion_content: str, sentiment_text: str) -> str:
        """뉴스 및 의견 통합 요약 프롬프트 생성"""
        # 프롬프트 내용은 이전과 동일
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a skilled editor synthesizing factual news reports and public opinion into a concise, balanced overview.<|eot_id|><|start_header_id|>user<|end_header_id|>
Synthesize the following news summaries and public opinion summary into a single, coherent paragraph of approximately {self.target_word_count} words.

[Factual News Summaries]
{news_content}

[Public Opinion Summary]
{opinion_content}

[Overall Sentiment Distribution]
{sentiment_text}

[Instructions]
1.  **Integrate** both the factual points from the news and the key perspectives from the opinion summary.
2.  **Distinguish** clearly between factual reporting (e.g., "Reports indicate...") and public opinion (e.g., "Public sentiment suggests...", "Some argue... while others worry...").
3.  **Balance** different viewpoints mentioned in the opinion summary (e.g., pro/con arguments).
4.  **Include** a mention of the overall sentiment distribution.
5.  **Synthesize** the information logically; do not just list points.
6.  **Adhere** strictly to the approximate word count of {self.target_word_count} words.
7.  **Maintain** a neutral and objective tone throughout the final summary.

[Synthesized Final Summary]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    async def _generate_synthesis(self, news_content: str, opinion_content: str, sentiment_text: str, trace_id: Optional[str]) -> str:
        """LLM을 사용하여 통합 요약 생성"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if news_content.startswith("No ") and opinion_content.startswith("No "):
            logger.warning(f"{log_prefix} Both news and opinion content are unavailable. Cannot generate synthesis.")
            return "Error: Insufficient information to generate a synthesized summary."

        prompt = self._create_synthesis_prompt_en(news_content, opinion_content, sentiment_text)
        try:
            summary = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_synthesis,
                max_tokens=self.max_tokens_synthesis,
                trace_id=trace_id
            )
            logger.info(f"{log_prefix} Generated synthesized summary (Length: {len(summary)}).")
            return summary
        except RetryError as e:
             logger.error(f"{log_prefix} Failed to generate synthesis summary after retries: {e}")
             return f"Error: Could not generate the final synthesis summary due to LLM failure."
        except Exception as e:
            logger.exception(f"{log_prefix} Failed to generate synthesis summary: {e}")
            return f"Error: Could not generate the final synthesis summary due to an internal error."

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """통합 요약 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing SynthesisSummarizerNode...")

        news_summaries = state.news_summaries or []
        opinion_summaries = state.opinion_summaries # Optional[Dict] 이므로 None일 수 있음
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        # --------------------------

        # 입력 데이터 확인 및 로깅
        has_news = bool(news_summaries)
        has_opinion = bool(opinion_summaries and opinion_summaries.get("summary_text"))

        if not has_news and not has_opinion:
            logger.error(f"{log_prefix} Insufficient input: No news summaries and no opinion summary text found.")
            processing_stats['synthesis_summarizer_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"error_message": "Insufficient data for synthesis summarization.", "processing_stats": processing_stats}
        elif not has_news:
             logger.warning(f"{log_prefix} News summaries missing. Synthesizing based only on opinion.")
        elif not has_opinion:
             logger.warning(f"{log_prefix} Opinion summary missing. Synthesizing based only on news.")

        # --- LLM 입력 준비 ---
        news_input = self._prepare_news_input(news_summaries, trace_id)
        opinion_input, sentiment_str = self._prepare_opinion_input(opinion_summaries)

        # --- 통합 요약 생성 ---
        final_summary = await self._generate_synthesis(
            news_input, opinion_input, sentiment_str, trace_id
        )

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['synthesis_summarizer_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} SynthesisSummarizerNode finished in {processing_stats['synthesis_summarizer_node_time']:.2f} seconds.")

        error_message = final_summary if final_summary.startswith("Error:") else None

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "final_summary": final_summary if not error_message else None, # 오류 시 None 저장
            "processing_stats": processing_stats,
            "error_message": error_message # 오류 메시지 전달
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}