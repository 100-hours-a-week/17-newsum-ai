# app/nodes/10_synthesis_summarizer_node.py (Refactored)

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.services.llm_server_client_v2 import LLMService
from datetime import datetime, timezone, timedelta
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

class SynthesisSummarizerNode:
    """
    뉴스 요약과 의견 요약을 종합하여 최종 요약을 생성합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["news_summaries", "opinion_summaries", "trace_id", "comic_id", "config"] # MODIFIED: Added comic_id
    outputs: List[str] = ["final_summary", "node10_processing_stats", "error_message"]

    def __init__(self, llm_client: LLMService):
        if not llm_client: raise ValueError("LLMService is required for SynthesisSummarizerNode")
        self.llm_client = llm_client
        logger.info("SynthesisSummarizerNode initialized.")

    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict): # MODIFIED: Added extra_log_data
        self.feqa_threshold = float(config.get("feqa_threshold", settings.DEFAULT_FEQA_THRESHOLD))
        self.max_news_summaries = int(config.get("max_news_summaries_for_synthesis", settings.DEFAULT_MAX_SUMMARIES_SYNTHESIS))
        self.llm_temp_synthesis = float(config.get("llm_temperature_synthesis", settings.DEFAULT_LLM_TEMP_SYNTHESIS))
        self.max_tokens_synthesis = int(config.get("max_tokens_synthesis", settings.DEFAULT_MAX_TOKENS_SYNTHESIS))
        self.target_word_count = int(config.get("synthesis_target_word_count", settings.DEFAULT_SYNTHESIS_WORD_COUNT))

        logger.debug(f"Runtime config loaded. FEQA Threshold: {self.feqa_threshold}, Max News Summaries: {self.max_news_summaries}", extra=extra_log_data) # MODIFIED
        logger.debug(f"LLM Temp: {self.llm_temp_synthesis}, Max Tokens: {self.max_tokens_synthesis}, Target Words: {self.target_word_count}", extra=extra_log_data) # MODIFIED

    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float, max_tokens: int,
                                   trace_id: Optional[str] = None, comic_id: Optional[str] = None, **kwargs) -> str: # MODIFIED: Added comic_id
        llm_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        logger.debug(f"Calling LLMService (Temp: {temperature}, MaxTokens: {max_tokens})...", extra=llm_log_data) # MODIFIED

        result = await self.llm_client.generate_text(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens,
            # trace_id=trace_id, comic_id=comic_id, # Pass if supported
            **kwargs
        )

        if "error" in result:
            error_msg = result['error']
            logger.error(f"LLMService call failed: {error_msg}", extra=llm_log_data) # MODIFIED
            raise RuntimeError(f"LLMService error: {error_msg}")
        elif "generated_text" not in result or not isinstance(result.get("generated_text"), str) or not result["generated_text"].strip():
            logger.error(f"LLMService returned invalid or empty text. Response: {result}", extra=llm_log_data) # MODIFIED
            raise ValueError("LLMService returned invalid or empty text")
        else:
            logger.debug(f"LLMService call successful.", extra=llm_log_data) # MODIFIED
            return result["generated_text"].strip()

    def _prepare_news_input(self, news_summaries: List[Dict[str, Any]], trace_id: Optional[str], comic_id: Optional[str]) -> str: # MODIFIED: Added comic_id
        """통합 요약 프롬프트를 위한 뉴스 요약 입력 준비"""
        prep_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not news_summaries: return "No news summaries available."

        # Ensure summaries have text and score is float
        valid_summaries = [
            s for s in news_summaries
            if isinstance(s.get("summary_text"), str) and s["summary_text"].strip() and isinstance(s.get("feqa_score"), (int, float))
        ]

        high_quality_summaries = sorted(
            [s for s in valid_summaries if s.get("feqa_score", 0.0) >= self.feqa_threshold],
            key=lambda x: x.get("feqa_score", 0.0), reverse=True
        )[:self.max_news_summaries]

        selected_texts = [s['summary_text'] for s in high_quality_summaries]

        if not selected_texts and valid_summaries: # Fallback if no summaries meet threshold
            logger.warning(f"No news summaries met FEQA threshold {self.feqa_threshold}. Using highest available (up to {self.max_news_summaries}).", extra=prep_log_data) # MODIFIED
            all_sorted = sorted(valid_summaries, key=lambda x: x.get("feqa_score", 0.0), reverse=True)
            selected_texts = [s['summary_text'] for s in all_sorted[:self.max_news_summaries]]

        if not selected_texts:
             logger.warning("No usable news summaries found after filtering.", extra=prep_log_data) # MODIFIED
             return "No usable news summaries available."

        logger.info(f"Prepared {len(selected_texts)} news summaries for synthesis input.", extra=prep_log_data) # MODIFIED
        # Add clear separation and numbering
        return "\n\n".join(f"[News Summary {i+1} (FEQA: {high_quality_summaries[i].get('feqa_score', 'N/A'):.2f})]\n{text}"
                           for i, text in enumerate(selected_texts)).strip()

    def _prepare_opinion_input(self, opinion_summaries: Optional[Dict[str, Any]]) -> Tuple[str, str]:
        """통합 요약 프롬프트를 위한 의견 요약 및 감성 정보 준비"""
        if not opinion_summaries or not isinstance(opinion_summaries, dict): # Added type check
            return "No opinion summary available.", "Sentiment data not available."

        # Ensure summary text is a non-empty string
        opinion_summary_text = opinion_summaries.get("summary_text", "")
        if not isinstance(opinion_summary_text, str) or not opinion_summary_text.strip():
            opinion_summary_text = "No valid opinion summary text provided."

        sentiment_distribution = opinion_summaries.get("sentiment_distribution", {})
        if sentiment_distribution and isinstance(sentiment_distribution, dict):
            sentiment_parts = [f"{label}: {dist*100:.1f}%"
                               for label, dist in sentiment_distribution.items() if isinstance(dist, (int, float))]
            sentiment_text = ", ".join(sentiment_parts) if sentiment_parts else "Sentiment data format error."
        else:
            sentiment_text = "Sentiment data not available or invalid format."

        return opinion_summary_text, sentiment_text

    def _create_synthesis_prompt_en(self, news_content: str, opinion_content: str, sentiment_text: str) -> str:
        # [... existing prompt ...]
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


    async def _generate_synthesis(self, news_content: str, opinion_content: str, sentiment_text: str, trace_id: Optional[str], comic_id: Optional[str]) -> str: # MODIFIED: Added comic_id
        """LLM을 사용하여 통합 요약 생성"""
        synth_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        # Check if content is actually unavailable vs. just the placeholder string
        news_unavailable = news_content == "No news summaries available." or news_content == "No usable news summaries available."
        opinion_unavailable = opinion_content == "No opinion summary available." or opinion_content == "No valid opinion summary text provided."

        if news_unavailable and opinion_unavailable:
            logger.error("Both news and opinion content are unavailable. Cannot generate synthesis.", extra=synth_log_data) # MODIFIED (Error level)
            return "Error: Insufficient information to generate a synthesized summary."

        prompt = self._create_synthesis_prompt_en(news_content, opinion_content, sentiment_text)
        try:
            # Pass IDs
            summary = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_synthesis,
                max_tokens=self.max_tokens_synthesis,
                trace_id=trace_id,
                comic_id=comic_id
            )
            logger.info(f"Generated synthesized summary (Length: {len(summary)}).", extra=synth_log_data) # MODIFIED
            return summary
        except RetryError as e:
             logger.error(f"Failed to generate synthesis summary after retries: {e}", extra=synth_log_data) # MODIFIED
             return f"Error: Could not generate the final synthesis summary due to LLM failure."
        except Exception as e:
            logger.exception("Failed to generate synthesis summary.", extra=synth_log_data) # MODIFIED use exception
            return f"Error: Could not generate the final synthesis summary due to an internal error."

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """통합 요약 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        # --- MODIFIED: Get trace_id and comic_id safely ---
        comic_id = getattr(state, 'comic_id', 'unknown_comic')
        trace_id = getattr(state, 'trace_id', comic_id)
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        # -------------------------------------------------

        node_class_name = self.__class__.__name__
        # --- ADDED: Start Logging ---
        logger.info(f"--- Executing {node_class_name} ---", extra=extra_log_data)
        logger.debug(f"Entering state:\n{summarize_for_logging(state, is_state_object=True)}", extra=extra_log_data)
        # --------------------------

        news_summaries = getattr(state, 'news_summaries', []) # Safe access
        opinion_summaries = getattr(state, 'opinion_summaries', None) # Safe access, allow None
        config = getattr(state, 'config', {}) or {}

        # --- ADDED: Input Validation (moved from _generate_synthesis) ---
        # Check if *any* usable data exists
        has_valid_news = any(isinstance(s.get("summary_text"), str) and s["summary_text"].strip() for s in news_summaries)
        has_valid_opinion = isinstance(opinion_summaries, dict) and isinstance(opinion_summaries.get("summary_text"), str) and opinion_summaries["summary_text"].strip()

        if not has_valid_news and not has_valid_opinion:
            error_message = "Insufficient input: No valid news summaries or opinion summary text found."
            logger.error(error_message, extra=extra_log_data)
            end_time = datetime.now(timezone.utc)
            node10_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "final_summary": None,
                "node10_processing_stats": node10_processing_stats,
                "error_message": error_message
            }
            # --- ADDED: End Logging (Error Case) ---
            logger.debug(f"Returning updates (insufficient data):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Error: Insufficient Data) --- (Elapsed: {node10_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        elif not has_valid_news:
             logger.warning("No valid news summaries found. Synthesizing based only on opinion.", extra=extra_log_data)
        elif not has_valid_opinion:
             logger.warning("No valid opinion summary found. Synthesizing based only on news.", extra=extra_log_data)
        # ------------------------------------------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        # --------------------------

        # --- LLM Input Preparation ---
        news_input = self._prepare_news_input(news_summaries, trace_id, comic_id) # Pass IDs
        opinion_input, sentiment_str = self._prepare_opinion_input(opinion_summaries)

        # --- Synthesis Generation ---
        # Pass IDs
        final_summary = await self._generate_synthesis(
            news_input, opinion_input, sentiment_str, trace_id, comic_id
        )

        end_time = datetime.now(timezone.utc)
        node10_processing_stats = (end_time - start_time).total_seconds()
        error_message = final_summary if final_summary.startswith("Error:") else None

        update_data: Dict[str, Any] = {
            "final_summary": final_summary if not error_message else None,
            "node10_processing_stats": node10_processing_stats,
            "error_message": error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if error_message else logger.info
        log_level(f"Synthesis result: Summary {'Failed' if error_message else 'Generated'}. Errors: {error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node10_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}