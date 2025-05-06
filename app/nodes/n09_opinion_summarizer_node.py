# app/nodes/09_opinion_summarizer_node.py (Refactored)

import asyncio
from collections import Counter, defaultdict
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

class OpinionSummarizerNode:
    """
    정리된 의견들을 요약합니다 (스탠스 분류, 감성 분석, 최종 요약).
    [... existing docstring ...]
    """
    inputs: List[str] = ["opinions_clean", "topic_analysis", "trace_id", "comic_id", "config"] # MODIFIED: Added comic_id
    outputs: List[str] = ["opinion_summaries", "node9_processing_stats", "error_message"]

    def __init__(self, llm_client: LLMService):
        if not llm_client: raise ValueError("LLMService is required for OpinionSummarizerNode")
        self.llm_client = llm_client
        self.stance_labels = ["Pro", "Con", "Neutral"]
        self.sentiment_labels = ["Positive", "Negative", "Neutral"]
        logger.info("OpinionSummarizerNode initialized.")

    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict): # MODIFIED: Added extra_log_data
        self.llm_temp_stance = float(config.get("llm_temperature_stance", settings.LLM_TEMPERATURE_STANCE))
        self.max_tokens_stance = int(config.get("max_tokens_stance", settings.MAX_TOKENS_STANCE))
        self.llm_temp_sentiment = float(config.get("llm_temperature_sentiment", settings.LLM_TEMPERATURE_SENTIMENT))
        self.max_tokens_sentiment = int(config.get("max_tokens_sentiment", settings.MAX_TOKENS_SENTIMENT))
        self.llm_temp_opinion_summarize = float(
            config.get("llm_temperature_opinion_summarize", settings.LLM_TEMPERATURE_OPINION_SUMMARIZE))
        self.max_tokens_opinion_summarize = int(
            config.get("max_tokens_opinion_summarize", settings.MAX_TOKENS_OPINION_SUMMARIZE))

        self.max_opinion_text_len = int(config.get("max_opinion_text_len", settings.MAX_ALT_TEXT_LEN))
        self.max_opinions_for_stance = int(config.get("max_opinions_for_stance", settings.MAX_OPINIONS_STANCE))
        self.max_opinions_for_sentiment = int(
            config.get("max_opinions_for_sentiment", settings.DEFAULT_MAX_OPINIONS_SENTIMENT))
        self.concurrency_limit = int(config.get('summarizer_concurrency', settings.DEFAULT_SUMMARIZER_CONCURRENCY))

        logger.debug(f"Runtime config loaded. Max Opinions (Stance/Sent): {self.max_opinions_for_stance}/{self.max_opinions_for_sentiment}, Concurrency: {self.concurrency_limit}", extra=extra_log_data) # MODIFIED
        logger.debug(f"LLM Temps (St/Se/Sum): {self.llm_temp_stance}/{self.llm_temp_sentiment}/{self.llm_temp_opinion_summarize}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Max Tokens (St/Se/Sum): {self.max_tokens_stance}/{self.max_tokens_sentiment}/{self.max_tokens_opinion_summarize}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Max Opinion Text Length: {self.max_opinion_text_len}", extra=extra_log_data) # MODIFIED

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

    def _create_stance_prompt_en(self, opinion_text: str, topic: str) -> str:
        truncated_text = opinion_text[:self.max_opinion_text_len] + ("..." if len(opinion_text) > self.max_opinion_text_len else "")
        # [... existing prompt ...]
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

    async def _classify_stance(self, opinion_text: str, topic: str, trace_id: Optional[str], comic_id: Optional[str]) -> str: # MODIFIED: Added comic_id
        """LLM을 사용하여 단일 의견의 스탠스 분류. Returns label or 'Neutral' on error."""
        stance_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not opinion_text: return "Neutral"

        prompt = self._create_stance_prompt_en(opinion_text, topic)
        try:
            # Pass IDs
            response = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_stance,
                max_tokens=self.max_tokens_stance,
                trace_id=trace_id,
                comic_id=comic_id
            )
            response_clean = response.strip().capitalize()
            if response_clean in self.stance_labels:
                return response_clean
            else:
                logger.warning(f"Unexpected stance LLM response: '{response}'. Defaulting to Neutral.", extra=stance_log_data) # MODIFIED
                return "Neutral"
        except RetryError as e:
             logger.error(f"Failed to classify stance after retries for text '{opinion_text[:30]}...': {e}", extra=stance_log_data) # MODIFIED
             return "Neutral"
        except Exception as e:
            logger.error(f"Error classifying stance for text '{opinion_text[:30]}...': {e}", exc_info=True, extra=stance_log_data) # MODIFIED
            return "Neutral"

    def _create_sentiment_prompt_en(self, opinion_text: str) -> str:
        truncated_text = opinion_text[:self.max_opinion_text_len] + ("..." if len(opinion_text) > self.max_opinion_text_len else "")
        # [... existing prompt ...]
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

    async def _analyze_sentiment(self, opinion_text: str, trace_id: Optional[str], comic_id: Optional[str]) -> str: # MODIFIED: Added comic_id
        """LLM을 사용하여 단일 의견의 감성 분석. Returns label or 'Neutral' on error."""
        sent_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not opinion_text: return "Neutral"

        prompt = self._create_sentiment_prompt_en(opinion_text)
        try:
            # Pass IDs
            response = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_sentiment,
                max_tokens=self.max_tokens_sentiment,
                trace_id=trace_id,
                comic_id=comic_id
            )
            response_clean = response.strip().capitalize()
            if response_clean in self.sentiment_labels:
                return response_clean
            else:
                logger.warning(f"Unexpected sentiment LLM response: '{response}'. Defaulting to Neutral.", extra=sent_log_data) # MODIFIED
                return "Neutral"
        except RetryError as e:
             logger.error(f"Failed to analyze sentiment after retries for text '{opinion_text[:30]}...': {e}", extra=sent_log_data) # MODIFIED
             return "Neutral"
        except Exception as e:
            logger.error(f"Error analyzing sentiment for text '{opinion_text[:30]}...': {e}", exc_info=True, extra=sent_log_data) # MODIFIED
            return "Neutral"

    def _format_stance_clusters_for_prompt(self, stance_clusters: Dict[str, List[str]]) -> str:
        formatted_string = ""
        for stance in self.stance_labels:
            texts = stance_clusters.get(stance, [])
            if texts:
                formatted_string += f"\n[{stance} Opinions]\n"
                for text in texts[:3]: # Limit examples
                    preview = text[:150] + "..." if len(text) > 150 else text
                    formatted_string += f"- {preview}\n"
            else:
                formatted_string += f"\n[{stance} Opinions]\n- No specific examples provided for this stance.\n" # More informative
        return formatted_string.strip()

    def _format_sentiment_for_prompt(self, sentiment_distribution: Dict[str, float]) -> str:
        if not sentiment_distribution: return "Sentiment data not available."
        return ", ".join([f"{label}: {distribution * 100:.1f}%"
                          for label, distribution in sentiment_distribution.items()])

    def _create_summary_prompt_en(self, topic: str, formatted_stances: str, formatted_sentiment: str) -> str:
        # [... existing prompt ...]
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
                                      trace_id: Optional[str], comic_id: Optional[str]) -> str: # MODIFIED: Added comic_id
        """LLM을 사용하여 최종 의견 요약 생성"""
        final_sum_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        formatted_stances = self._format_stance_clusters_for_prompt(stance_clusters)
        formatted_sentiment = self._format_sentiment_for_prompt(sentiment_distribution)

        prompt = self._create_summary_prompt_en(topic, formatted_stances, formatted_sentiment)
        try:
            # Pass IDs
            summary = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_opinion_summarize,
                max_tokens=self.max_tokens_opinion_summarize,
                trace_id=trace_id,
                comic_id=comic_id
            )
            logger.info(f"Generated final opinion summary (Length: {len(summary)}).", extra=final_sum_log_data) # MODIFIED
            return summary
        except RetryError as e:
            logger.error(f"Failed to generate final opinion summary after retries: {e}", extra=final_sum_log_data) # MODIFIED
            return f"Error: Could not generate opinion summary due to LLM failure."
        except Exception as e:
            logger.exception("Failed to generate final opinion summary.", extra=final_sum_log_data) # MODIFIED use exception
            return f"Error: Could not generate opinion summary due to an internal error."

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """의견 요약 워크플로우 실행"""
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

        opinions_clean = getattr(state, 'opinions_clean', []) # Safe access
        topic_analysis = getattr(state, 'topic_analysis', {}) or {} # Safe access, ensure dict
        config = getattr(state, 'config', {}) or {}

        # --- ADDED: Input Validation ---
        if not opinions_clean:
            error_message = "No cleaned opinions provided for summarization."
            logger.warning(error_message, extra=extra_log_data) # Warning is ok
            end_time = datetime.now(timezone.utc)
            node9_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "opinion_summaries": {}, # Return empty dict
                "node9_processing_stats": node9_processing_stats,
                "error_message": error_message
            }
             # --- ADDED: End Logging (Early Exit) ---
            logger.debug(f"Returning updates (no opinions):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (No Opinions) --- (Elapsed: {node9_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}

        if not topic_analysis or not topic_analysis.get('main_topic'):
             # Allow proceeding, but use a default topic and log a warning
             main_topic = "the analyzed topic"
             logger.warning("main_topic missing from topic_analysis. Using default topic for summarization.", extra=extra_log_data)
        else:
             main_topic = topic_analysis.get('main_topic')
        # -------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        # --------------------------

        logger.info(f"Summarizing {len(opinions_clean)} cleaned opinions on topic: '{main_topic}'...", extra=extra_log_data)

        opinions_to_analyze = [op for op in opinions_clean if op.get('is_representative')]
        if not opinions_to_analyze:
            logger.info(f"No representative opinions found, using all {len(opinions_clean)} cleaned opinions for analysis.", extra=extra_log_data)
            opinions_to_analyze = opinions_clean
        else:
            logger.info(f"Using {len(opinions_to_analyze)} representative opinions for analysis.", extra=extra_log_data)

        opinions_for_stance_call = opinions_to_analyze[:self.max_opinions_for_stance]
        opinions_for_sentiment_call = opinions_to_analyze[:self.max_opinions_for_sentiment]
        logger.info(f"Calling LLM for stance ({len(opinions_for_stance_call)}) and sentiment ({len(opinions_for_sentiment_call)}).", extra=extra_log_data)

        task_errors: List[str] = []

        # --- Stance Classification ---
        stance_tasks = []
        async def classify_stance_task(opinion):
            async with semaphore:
                text = opinion.get('text', '')
                # Pass comic_id
                stance = await self._classify_stance(text, main_topic, trace_id, comic_id)
                return stance, text

        for opinion in opinions_for_stance_call: stance_tasks.append(classify_stance_task(opinion))
        # MODIFIED: Use return_exceptions=True
        stance_results = await asyncio.gather(*stance_tasks, return_exceptions=True)

        stance_clusters: Dict[str, List[str]] = defaultdict(list)
        stance_counts = Counter()
        # MODIFIED: Handle exceptions from gather
        for result in stance_results:
            if isinstance(result, tuple):
                stance, text = result
                if stance in self.stance_labels and text:
                    stance_clusters[stance].append(text)
                    stance_counts[stance] += 1
            elif isinstance(result, Exception):
                err_msg = f"Stance classification task failed: {result}"
                logger.error(err_msg, exc_info=result, extra=extra_log_data) # Log exception details
                task_errors.append(f"Stance Task Failed: {result}") # Summary error
            else:
                logger.warning(f"Unexpected stance result type: {type(result)}", extra=extra_log_data)
        logger.info(f"Stance classification complete. Counts: {dict(stance_counts)}", extra=extra_log_data)

        # --- Sentiment Analysis ---
        sentiment_tasks = []
        async def analyze_sentiment_task(opinion):
            async with semaphore:
                text = opinion.get('text', '')
                # Pass comic_id
                return await self._analyze_sentiment(text, trace_id, comic_id)

        for opinion in opinions_for_sentiment_call: sentiment_tasks.append(analyze_sentiment_task(opinion))
        # MODIFIED: Use return_exceptions=True
        sentiment_results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)

        sentiment_counter = Counter()
        valid_sentiment_results = 0
        # MODIFIED: Handle exceptions from gather
        for result in sentiment_results:
            if isinstance(result, str) and result in self.sentiment_labels:
                sentiment_counter[result] += 1
                valid_sentiment_results += 1
            elif isinstance(result, Exception):
                err_msg = f"Sentiment analysis task failed: {result}"
                logger.error(err_msg, exc_info=result, extra=extra_log_data) # Log exception details
                task_errors.append(f"Sentiment Task Failed: {result}") # Summary error
            else:
                # Could be 'Neutral' returned on error, or unexpected type
                if isinstance(result, str) and result == "Neutral":
                     sentiment_counter["Neutral"] += 1 # Count default Neutral on error
                     valid_sentiment_results += 1 # Count it as valid for distribution
                else:
                     logger.warning(f"Unexpected sentiment result type: {type(result)}", extra=extra_log_data)


        sentiment_distribution: Dict[str, float] = {}
        if valid_sentiment_results > 0:
            for label in self.sentiment_labels:
                sentiment_distribution[label] = round(sentiment_counter[label] / valid_sentiment_results, 3)
        else:
            logger.warning("No valid sentiment results obtained. Setting default distribution.", extra=extra_log_data)
            sentiment_distribution = {"Positive": 0.0, "Negative": 0.0, "Neutral": 1.0}
        logger.info(f"Sentiment analysis complete. Distribution: {sentiment_distribution}", extra=extra_log_data)

        # --- Final Summary Generation ---
        # Pass comic_id
        summary_text = await self._generate_final_summary(
            main_topic, stance_clusters, sentiment_distribution, trace_id, comic_id
        )

        # --- Output Formatting ---
        stance_clusters_output = [
            {"stance": stance, "representative_texts": texts[:5]}
            for stance, texts in stance_clusters.items() if texts
        ]
        opinion_summaries_output = {
            "summary_text": summary_text,
            "stance_clusters": stance_clusters_output,
            "sentiment_distribution": sentiment_distribution
        }

        end_time = datetime.now(timezone.utc)
        node9_processing_stats = (end_time - start_time).total_seconds()
        final_error_message = "; ".join(task_errors) if task_errors else None
        # Check if summary generation itself failed
        if summary_text.startswith("Error:"):
             final_error_message = f"{final_error_message or ''}; {summary_text}".strip('; ')


        update_data: Dict[str, Any] = {
            # Include summary even if it's an error message for context? Or set to None? Setting to None.
            "opinion_summaries": opinion_summaries_output if not summary_text.startswith("Error:") else {},
            "node9_processing_stats": node9_processing_stats,
            "error_message": final_error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if final_error_message else logger.info
        num_clusters = len(opinion_summaries_output.get("stance_clusters", []))
        summary_ok = not summary_text.startswith("Error:")
        log_level(f"Opinion summarization result: Summary {'OK' if summary_ok else 'Failed'}, Clusters: {num_clusters}. Errors: {final_error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node9_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}