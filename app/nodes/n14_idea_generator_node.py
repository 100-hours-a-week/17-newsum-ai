# app/nodes/14_idea_generator_node.py (Refactored)

import asyncio
import re
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.services.llm_server_client_v2 import LLMService
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

class IdeaGeneratorNode:
    """
    최종 요약과 트렌드 점수를 기반으로 4컷 만화 아이디어를 생성합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["final_summary", "trend_scores", "trace_id", "comic_id", "config"] # MODIFIED: Added comic_id
    outputs: List[str] = ["comic_ideas", "node14_processing_stats", "error_message"]

    def __init__(self, llm_client: LLMService):
        if not llm_client: raise ValueError("LLMService is required for IdeaGeneratorNode")
        self.llm_client = llm_client
        logger.info("IdeaGeneratorNode initialized.")

    # --- MODIFIED: Added extra_log_data argument ---
    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.llm_temp_creative = float(config.get("llm_temperature_creative", settings.DEFAULT_LLM_TEMP_CREATIVE))
        self.max_tokens_idea = int(config.get("llm_max_tokens_idea", settings.DEFAULT_MAX_TOKENS_IDEA))
        self.top_n_trends_for_prompt = int(config.get("trends_report_top_n", settings.DEFAULT_TRENDS_REPORT_TOP_N))

        logger.debug(f"Runtime config loaded. LLM Temp: {self.llm_temp_creative}, Max Tokens: {self.max_tokens_idea}, Top Trends: {self.top_n_trends_for_prompt}", extra=extra_log_data) # MODIFIED

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
            logger.debug("LLMService call successful.", extra=llm_log_data) # MODIFIED
            return result["generated_text"].strip()

    def _prepare_trend_info_for_prompt(self, trend_scores: List[Dict[str, Any]]) -> str:
        """LLM 프롬프트용 상위 트렌드 정보 포맷팅"""
        if not trend_scores or not isinstance(trend_scores, list): # Added type check
             return "No specific trending topics identified recently."

        # Ensure items are valid dicts with scores before filtering/sorting
        valid_trends = [t for t in trend_scores if isinstance(t, dict) and isinstance(t.get('score'), (int, float))]
        # Node 12 already sorted, just take top N positive scores
        top_trends = [t for t in valid_trends if t.get('score', 0) > 0][:self.top_n_trends_for_prompt]

        if not top_trends: return "No significant trending topics identified recently."

        trend_lines = [f"- \"{trend.get('keyword', 'N/A')}\" (Score: {trend.get('score', 0):.1f})" for trend in top_trends]
        return "Recent potentially relevant trending keywords:\n" + "\n".join(trend_lines)

    def _create_idea_prompt_en(self, summary: str, trend_info: str) -> str:
        # [... existing prompt ...]
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a creative assistant tasked with generating compelling 4-panel comic ideas based on news summaries and trending topics. Provide exactly 5 diverse ideas in the specified JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>
Generate 5 distinct 4-panel comic ideas inspired by the following news summary and trending keywords. For each idea, provide a catchy title, a brief concept description, and a self-assessed creativity score (0.0 to 1.0).

[News Summary]
{summary}

[Trending Keywords Context]
{trend_info}

[Instructions]
1.  Create exactly 5 different comic ideas.
2.  Each idea should be suitable for a 4-panel comic format.
3.  Ideas should creatively connect the news summary and/or trending keywords.
4.  For each idea, include:
    * `idea_title`: A short, engaging title (string).
    * `concept`: A 1-2 sentence description of the comic's premise or storyline (string).
    * `creative_score`: Your assessment of the idea's creativity and potential interest (float between 0.0 and 1.0).
5.  Respond ONLY with a valid JSON list containing the 5 idea objects. Ensure keys and string values use double quotes.

[Required JSON Output Format]
```json
[
  {{
    "idea_title": "Example Title 1",
    "concept": "Example concept description 1.",
    "creative_score": 0.85
  }},
  {{
    "idea_title": "Example Title 2",
    "concept": "Example concept description 2.",
    "creative_score": 0.70
  }},
  {{
    "idea_title": "Example Title 3",
    "concept": "Example concept description 3.",
    "creative_score": 0.90
  }},
  {{
    "idea_title": "Example Title 4",
    "concept": "Example concept description 4.",
    "creative_score": 0.65
  }},
  {{
    "idea_title": "Example Title 5",
    "concept": "Example concept description 5.",
    "creative_score": 0.80
  }}
]
```<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
        return prompt


    # --- MODIFIED: Added extra_log_data argument ---
    def _parse_llm_response(self, response_json: str, trace_id: Optional[str], comic_id: Optional[str]) -> List[Dict[str, Any]]:
        """LLM의 아이디어 JSON 응답 파싱 및 검증"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        try:
            logger.debug(f"Raw LLM response for ideas: {response_json[:500]}...", extra=extra_log_data) # MODIFIED
            # Improved JSON extraction
            match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_json, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1)
            else:
                # Attempt to find the list directly if no backticks
                start = response_json.find('[')
                end = response_json.rfind(']')
                if start != -1 and end != -1 and end > start:
                    json_str = response_json[start:end+1]
                else: # Give up if no clear list found
                    raise ValueError("Could not extract JSON list from LLM response.")

            logger.debug(f"Cleaned JSON for parsing: {json_str[:500]}...", extra=extra_log_data) # MODIFIED
            ideas = json.loads(json_str)

            if not isinstance(ideas, list): raise ValueError("LLM response is not a JSON list.")

            validated_ideas = []
            required_keys = {"idea_title", "concept", "creative_score"}
            for idx, idea in enumerate(ideas):
                if isinstance(idea, dict) and required_keys.issubset(idea.keys()):
                    title = idea.get('idea_title')
                    concept = idea.get('concept')
                    score_val = idea.get('creative_score')
                    # Check types rigorously
                    if isinstance(title, str) and title.strip() and \
                       isinstance(concept, str) and concept.strip() and \
                       isinstance(score_val, (float, int)):
                        score = max(0.0, min(1.0, float(score_val))) # Clamp score
                        validated_ideas.append({
                            "idea_title": title.strip(),
                            "concept": concept.strip(),
                            "creative_score": round(score, 3) # Standard precision
                        })
                    else: logger.warning(f"Idea #{idx+1} has invalid data types: {idea}", extra=extra_log_data) # MODIFIED
                else: logger.warning(f"Idea #{idx+1} format invalid or missing keys: {idea}", extra=extra_log_data) # MODIFIED

            if len(validated_ideas) < 5: logger.warning(f"LLM returned fewer than 5 valid ideas ({len(validated_ideas)}).", extra=extra_log_data) # MODIFIED
            elif len(validated_ideas) > 5: logger.warning(f"LLM returned more than 5 ideas. Using first 5.", extra=extra_log_data) # MODIFIED

            return validated_ideas[:5]

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed parsing/validating LLM idea response: {e}. Response: '{response_json[:200]}...'", extra=extra_log_data) # MODIFIED
            return []
        except Exception as e:
            logger.exception("Unexpected error parsing LLM idea response.", extra=extra_log_data) # MODIFIED use exception
            return []

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """아이디어 생성 프로세스 실행"""
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

        final_summary = getattr(state, 'final_summary', None) # Safe access
        trend_scores = getattr(state, 'trend_scores', []) or []
        config = getattr(state, 'config', {}) or {}

        # --- ADDED: Input Validation ---
        if not final_summary or not isinstance(final_summary, str) or not final_summary.strip():
            error_message = "Final summary is missing or empty. Cannot generate ideas."
            logger.error(error_message, extra=extra_log_data)
            end_time = datetime.now(timezone.utc)
            node14_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "comic_ideas": [],
                "node14_processing_stats": node14_processing_stats,
                "error_message": error_message
            }
            # --- ADDED: End Logging (Error Case) ---
            logger.debug(f"Returning updates (no summary):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Error: Missing Summary) --- (Elapsed: {node14_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        # --------------------------

        logger.info("Starting comic idea generation...", extra=extra_log_data)
        error_message: Optional[str] = None
        comic_ideas: List[Dict[str, Any]] = []

        try:
            trend_info = self._prepare_trend_info_for_prompt(trend_scores)
            prompt = self._create_idea_prompt_en(final_summary, trend_info)
            llm_kwargs = {"response_format": {"type": "json_object"}}

            # Pass IDs
            response_str = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_creative,
                max_tokens=self.max_tokens_idea,
                trace_id=trace_id,
                comic_id=comic_id, # Pass ID
                **llm_kwargs
            )
            # Pass IDs
            comic_ideas = self._parse_llm_response(response_str, trace_id, comic_id)

            if not comic_ideas:
                 # Parser already logged details
                 error_message = "Failed to generate or parse any valid comic ideas from LLM."
            else:
                 logger.info(f"Successfully generated {len(comic_ideas)} comic ideas.", extra=extra_log_data)

        except RetryError as e:
            error_message = f"Idea generation LLM call failed after multiple retries: {e}"
            logger.error(error_message, extra=extra_log_data)
            comic_ideas = []
        except Exception as e:
            error_message = f"Idea generation failed: {str(e)}"
            logger.exception("Unexpected error during idea generation.", extra=extra_log_data) # Use exception
            comic_ideas = []

        end_time = datetime.now(timezone.utc)
        node14_processing_stats = (end_time - start_time).total_seconds()

        update_data: Dict[str, Any] = {
            "comic_ideas": comic_ideas,
            "node14_processing_stats": node14_processing_stats,
            "error_message": error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if error_message or not comic_ideas else logger.info
        log_level(f"Idea generation result: {len(comic_ideas)} ideas generated. Errors: {error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node14_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}