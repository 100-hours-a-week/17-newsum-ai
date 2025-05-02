# app/nodes/10_synthesis_summarizer_node.py

import asyncio
import re
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
logger = get_logger("SynthesisSummarizerNode")

class SynthesisSummarizerNode:
    """
    (Refactored) 뉴스 요약과 의견 요약을 종합하여 최종 요약을 생성합니다.
    - LLMService를 사용하여 통합 요약 생성.
    - 설정은 상태의 config 딕셔너리에서 로드.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["news_summaries", "opinion_summaries", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["final_summary", "processing_stats", "error_message"]

    # LLMService 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        llm_client: LLMService,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        self.llm_client = llm_client
        # self.langsmith_service = langsmith_service
        logger.info("SynthesisSummarizerNode initialized with LLMService.")

    # --- LLM 호출 래퍼 (재시도 적용) ---
    # 이전 노드들(08, 09)과 동일한 래퍼 사용
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

    # --- 입력 준비 ---
    def _prepare_news_input(self, news_summaries: List[Dict[str, Any]], config: Dict, trace_id: Optional[str]) -> str:
        """통합 요약 프롬프트를 위한 뉴스 요약 입력 준비"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not news_summaries: return "No news summaries available."

        feqa_threshold = float(config.get("feqa_threshold", 0.5))
        max_summaries = config.get("max_news_summaries_for_synthesis", 3)

        # FEQA 점수 기준으로 고품질 요약 필터링 및 선택
        high_quality_summaries = sorted(
            [s for s in news_summaries if s.get("feqa_score", 0.0) >= feqa_threshold and s.get("summary_text")],
            key=lambda x: x.get("feqa_score", 0.0),
            reverse=True # 높은 점수 우선
        )[:max_summaries] # 최대 개수 제한

        selected_texts = [s['summary_text'] for s in high_quality_summaries]

        # 고품질 요약이 없는 경우 처리
        if not selected_texts:
            logger.warning(f"{log_prefix} No news summaries met FEQA threshold {feqa_threshold}. Using highest available (up to {max_summaries}).")
            # 점수 무관하게 상위 N개 사용 (단, 요약 텍스트가 있어야 함)
            all_valid_summaries = sorted(
                [s for s in news_summaries if s.get("summary_text")],
                key=lambda x: x.get("feqa_score", 0.0), # 여전히 점수 높은 순 시도
                reverse=True
            )[:max_summaries]
            selected_texts = [s['summary_text'] for s in all_valid_summaries]

        if not selected_texts: return "No usable news summaries available."

        logger.info(f"{log_prefix} Prepared {len(selected_texts)} news summaries for synthesis input.")
        # 각 요약을 명확히 구분하여 LLM에 전달
        return "\n\n".join(f"[News Summary {i+1}]\n{text}" for i, text in enumerate(selected_texts))

    def _prepare_opinion_input(self, opinion_summaries: Dict[str, Any]) -> Tuple[str, str]:
        """통합 요약 프롬프트를 위한 의견 요약 및 감성 정보 준비"""
        if not opinion_summaries:
            return "No opinion summary available.", "Sentiment data not available."

        # OpinionSummarizerNode의 출력 형식에 맞춰 데이터 추출
        opinion_summary_text = opinion_summaries.get("summary_text", "No opinion summary text provided.")
        sentiment_distribution = opinion_summaries.get("sentiment_distribution", {})

        # 감성 분포 문자열 포맷팅
        if sentiment_distribution and isinstance(sentiment_distribution, dict):
            sentiment_parts = [f"{label}: {dist*100:.0f}%"
                               for label, dist in sentiment_distribution.items() if isinstance(dist, (int, float))]
            sentiment_text = ", ".join(sentiment_parts) if sentiment_parts else "Sentiment data format error."
        else:
            sentiment_text = "Sentiment data not available or invalid format."

        return opinion_summary_text, sentiment_text

    # --- 통합 요약 프롬프트 및 생성 ---
    def _create_synthesis_prompt_en(self, news_content: str, opinion_content: str, sentiment_text: str, target_word_count: int) -> str:
        """뉴스 및 의견 통합 요약 프롬프트 생성"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a skilled editor synthesizing factual news reports and public opinion into a concise, balanced overview.<|eot_id|><|start_header_id|>user<|end_header_id|>
Synthesize the following news summaries and public opinion summary into a single, coherent paragraph of approximately {target_word_count} words.

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
6.  **Adhere** strictly to the approximate word count of {target_word_count} words.
7.  **Maintain** a neutral and objective tone throughout the final summary.

[Synthesized Final Summary]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    async def _generate_synthesis(self, news_content: str, opinion_content: str, sentiment_text: str, config: Dict, trace_id: Optional[str]) -> str:
        """LLM을 사용하여 통합 요약 생성"""
        # 입력 데이터가 모두 없는 경우 처리
        if news_content.startswith("No ") and opinion_content.startswith("No "):
            logger.warning(f"[{trace_id}] Both news and opinion content are unavailable. Cannot generate synthesis.")
            return "Insufficient information to generate a synthesized summary."

        # 설정 로드
        llm_model = config.get("llm_model", "default_model")
        temperature = config.get("llm_temperature_synthesis", 0.5)
        max_tokens = config.get("max_tokens_synthesis", 300)
        # 목표 단어 수 (프롬프트 생성 시 사용) - 설정에서 가져오거나 기본값 설정 가능
        target_words = 110 # 예시 값 (약 300자)

        prompt = self._create_synthesis_prompt_en(news_content, opinion_content, sentiment_text, target_words)
        try:
            summary = await self._call_llm_with_retry(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, trace_id=trace_id
            )
            # 추가적인 후처리 (예: 길이 검사 및 조정) 필요시 여기에 구현
            logger.info(f"[{trace_id}] Generated synthesized summary (Length: {len(summary)}).")
            return summary
        except Exception as e:
            logger.exception(f"[{trace_id}] Failed to generate synthesis summary: {e}")
            return f"Error: Could not generate the final synthesis summary due to an internal error."

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """통합 요약 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing SynthesisSummarizerNode...")

        # 상태 및 설정 로드
        news_summaries = state.news_summaries or []
        opinion_summaries = state.opinion_summaries or {} # 기본값 {}
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        # 입력 데이터 확인 및 로깅
        if not news_summaries and not opinion_summaries.get("summary_text"):
            logger.error(f"{log_prefix} Insufficient input: No news summaries and no opinion summary text found.")
            return {"error_message": "Insufficient data for synthesis summarization."}
        elif not news_summaries:
             logger.warning(f"{log_prefix} News summaries missing. Synthesizing based only on opinion.")
        elif not opinion_summaries.get("summary_text"):
             logger.warning(f"{log_prefix} Opinion summary text missing. Synthesizing based only on news.")

        # --- LLM 입력 준비 ---
        news_input = self._prepare_news_input(news_summaries, config, state.trace_id)
        opinion_input, sentiment_str = self._prepare_opinion_input(opinion_summaries)

        # --- 통합 요약 생성 ---
        final_summary = await self._generate_synthesis(
            news_input, opinion_input, sentiment_str, config, state.trace_id
        )

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['synthesis_summarizer_node_time'] = node_processing_time
        logger.info(f"{log_prefix} SynthesisSummarizerNode finished in {node_processing_time:.2f} seconds.")

        # TODO: LangSmith 로깅

        # 오류 발생 시 final_summary에 오류 메시지가 포함될 수 있음
        error_message = final_summary if final_summary.startswith("Error:") else None

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "final_summary": final_summary if not error_message else None, # 오류 시 None 저장
            "processing_stats": processing_stats,
            "error_message": error_message # 오류 메시지 전달
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}