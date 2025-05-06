# app/nodes/15_scenario_writer_node.py (Refactored)

import asyncio
import re
import json
import hashlib
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

class ScenarioWriterNode:
    """
    선택된 아이디어와 요약 정보를 바탕으로 4컷 만화 시나리오를 생성합니다 (JSON 형식).
    [... existing docstring ...]
    """
    inputs: List[str] = [ # MODIFIED: Added comic_id
        "chosen_idea", "final_summary", "opinion_summaries", "articles", "opinions_raw",
        "trace_id", "comic_id", "config", "used_links"
    ]
    outputs: List[str] = ["scenarios", "scenario_prompt", "used_links", "node15_processing_stats", "error_message"]

    def __init__(self, llm_client: LLMService):
        if not llm_client: raise ValueError("LLMService is required for ScenarioWriterNode")
        self.llm_client = llm_client
        logger.info("ScenarioWriterNode initialized.")

    # --- MODIFIED: Added extra_log_data argument ---
    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.llm_temp_creative = float(config.get("llm_temperature_creative", settings.DEFAULT_LLM_TEMP_CREATIVE))
        self.max_tokens_scenario = int(config.get("llm_max_tokens_scenario", settings.DEFAULT_MAX_TOKENS_SCENARIO))
        self.max_context_len = int(config.get("max_context_len_scenario", settings.DEFAULT_MAX_CONTEXT_LEN_SCENARIO))

        logger.debug(f"Runtime config loaded. LLM Temp: {self.llm_temp_creative}, Max Tokens: {self.max_tokens_scenario}, Max Context Len: {self.max_context_len}", extra=extra_log_data) # MODIFIED

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

    def _prepare_context_for_prompt(self, final_summary: Optional[str], opinion_summary: Optional[Dict[str, Any]]) -> str:
        """시나리오 생성 프롬프트를 위한 컨텍스트 포맷팅 (길이 제한 적용)"""
        context_parts = []
        if final_summary and isinstance(final_summary, str): # Check type
            context_parts.append("[Overall Synthesis]\n" + final_summary)
        # Optionally add opinion summary text if valid
        # if opinion_summary and isinstance(opinion_summary, dict) and isinstance(opinion_summary.get('summary_text'), str) and opinion_summary['summary_text'].strip():
        #     context_parts.append("\n[Opinion Summary Detail]\n" + opinion_summary['summary_text'])

        full_context = "\n\n".join(context_parts).strip()
        if not full_context: return "No summary context available."
        # Truncate context based on max length
        truncated_context = full_context[:self.max_context_len]
        if len(full_context) > self.max_context_len:
             truncated_context += "..."
             # Log truncation
             # logger.debug(f"Context truncated from {len(full_context)} to {len(truncated_context)} chars.")
        return truncated_context

    def _create_scenario_prompt_en(self, idea_title: str, idea_concept: str, context: str) -> str:
        # [... existing prompt ...]
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a creative writer specializing in writing concise 4-panel comic scenarios. Generate a complete scenario based on the provided idea and context. Ensure the output is ONLY a valid JSON list containing 4 panel objects.<|eot_id|><|start_header_id|>user<|end_header_id|>
Generate a 4-panel comic scenario based on the following idea and background context.

[Chosen Comic Idea]
Title: {idea_title}
Concept: {idea_concept}

[Background Context from News/Opinions]
{context}

[Instructions]
1.  Create a scenario consisting of exactly 4 panels (scenes).
2.  For each panel, provide the following details:
    * `scene` (integer): The panel number (1, 2, 3, or 4).
    * `panel_description` (string): A brief visual description of the scene, including setting, characters, and key actions. Focus on what should be drawn.
    * `dialogue` (string): The dialogue or caption for the panel. Use an empty string "" if there is no text. Keep dialogue concise.
    * `seed_tags` (list of strings): 3-5 descriptive keywords or tags relevant to the panel's visual content (e.g., "office", "talking heads", "computer screen", "surprised expression", "city background"). These tags will guide image generation.
3.  The scenario should logically progress the chosen idea concept over the 4 panels.
4.  Ensure the output is ONLY a single, valid JSON list containing exactly 4 panel objects in the specified format. Do not include any introductory text or explanations outside the JSON structure.

[Required JSON Output Format]
```json
[
  {{
    "scene": 1,
    "panel_description": "string",
    "dialogue": "string",
    "seed_tags": ["string", "string", ...]
  }},
  {{
    "scene": 2,
    "panel_description": "string",
    "dialogue": "string",
    "seed_tags": ["string", "string", ...]
  }},
  {{
    "scene": 3,
    "panel_description": "string",
    "dialogue": "string",
    "seed_tags": ["string", "string", ...]
  }},
  {{
    "scene": 4,
    "panel_description": "string",
    "dialogue": "string",
    "seed_tags": ["string", "string", ...]
  }}
]
```<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
        return prompt

    # --- MODIFIED: Added extra_log_data argument ---
    def _parse_llm_response(self, response_json: str, trace_id: Optional[str], comic_id: Optional[str]) -> List[Dict[str, Any]]:
        """LLM의 시나리오 JSON 응답 파싱 및 검증"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        try:
            logger.debug(f"Raw LLM response for scenario: {response_json[:500]}...", extra=extra_log_data) # MODIFIED
            # Improved JSON extraction
            match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_json, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1)
            else:
                start = response_json.find('[')
                end = response_json.rfind(']')
                if start != -1 and end != -1 and end > start:
                    json_str = response_json[start:end+1]
                else:
                    raise ValueError("Could not extract JSON list from LLM response.")

            logger.debug(f"Cleaned JSON for parsing: {json_str[:500]}...", extra=extra_log_data) # MODIFIED
            scenarios = json.loads(json_str)

            if not isinstance(scenarios, list): raise ValueError("LLM response is not a JSON list.")

            validated_scenarios = []
            required_keys = {"scene", "panel_description", "dialogue", "seed_tags"}
            expected_scene = 1
            for panel in scenarios:
                if len(validated_scenarios) >= 4: break
                if isinstance(panel, dict) and required_keys.issubset(panel.keys()):
                    scene_num = panel.get('scene')
                    desc = panel.get('panel_description')
                    dialogue = panel.get('dialogue', "") # Default to empty string
                    tags = panel.get('seed_tags')

                    # Rigorous type checking
                    if isinstance(scene_num, int) and \
                       isinstance(desc, str) and desc.strip() and \
                       isinstance(dialogue, str) and \
                       isinstance(tags, list) and \
                       all(isinstance(tag, str) and tag.strip() for tag in tags):
                        panel['scene'] = expected_scene # Ensure scene numbers are correct
                        panel['seed_tags'] = [tag.strip() for tag in tags if tag.strip()] # Clean tags
                        validated_scenarios.append(panel)
                        expected_scene += 1
                    else: logger.warning(f"Panel #{expected_scene} invalid data types: {panel}", extra=extra_log_data) # MODIFIED
                else: logger.warning(f"Panel #{expected_scene} invalid format or missing keys: {panel}", extra=extra_log_data) # MODIFIED

            if len(validated_scenarios) != 4:
                 logger.error(f"Failed to validate exactly 4 panels (found {len(validated_scenarios)}).", extra=extra_log_data) # MODIFIED
                 return []
            return validated_scenarios

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed parsing/validating scenario response: {e}. Response: '{response_json[:200]}...'", extra=extra_log_data) # MODIFIED
            return []
        except Exception as e:
            logger.exception("Unexpected error parsing scenario response.", extra=extra_log_data) # MODIFIED use exception
            return []

    # --- MODIFIED: Added extra_log_data argument ---
    def _update_context_links(self, state: ComicState, trace_id: Optional[str], comic_id: Optional[str]) -> List[Dict[str, Any]]:
        """컨텍스트 생성에 사용된 링크 추적 (Placeholder)"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        logger.warning("Updating used_links for scenario context using PLACEHOLDER logic. Needs refinement!", extra=extra_log_data) # MODIFIED
        current_used_links = getattr(state, 'used_links', []) or []
        updated_used_links = list(current_used_links) # Create a copy
        urls_assumed_used = set()

        # Assume articles that were successfully summarized were used
        news_summaries = getattr(state, 'news_summaries', []) or []
        for summary in news_summaries:
             if isinstance(summary, dict) and summary.get("original_url"): # Check if original URL is stored in summary
                  urls_assumed_used.add(summary["original_url"])

        # Assume opinions that were successfully summarized were used (needs raw opinions mapping)
        opinion_summaries = getattr(state, 'opinion_summaries', {}) or {}
        # This is difficult without knowing *which* raw opinions contributed to the summary.
        # Simplistic approach: Assume all *successfully scraped* raw opinions were potentially used.
        opinions_raw = getattr(state, 'opinions_raw', []) or []
        for opinion in opinions_raw:
             if isinstance(opinion, dict) and opinion.get('text'): # Check if scraped
                  urls_assumed_used.add(opinion.get('url'))

        urls_assumed_used.discard(None)
        links_updated_count = 0
        temp_updated_links = [] # Build a new list to avoid index issues while iterating/modifying
        for link_info in updated_used_links:
             if isinstance(link_info, dict) and link_info.get('url') in urls_assumed_used:
                  updated_link = link_info.copy() # Modify a copy
                  purpose = updated_link.get('purpose', '')
                  if "Scenario Context" not in purpose: # Avoid multiple appends
                      updated_link['purpose'] = f"{purpose} | Used for Scenario Context" if purpose else "Used for Scenario Context"
                      links_updated_count += 1
                  updated_link['status'] = 'context_used' # Update status
                  temp_updated_links.append(updated_link)
             else:
                  temp_updated_links.append(link_info) # Keep unchanged link

        logger.info(f"Marked {links_updated_count} links as (assumed) used for scenario context.", extra=extra_log_data) # MODIFIED
        return temp_updated_links


    async def run(self, state: ComicState) -> Dict[str, Any]:
        """시나리오 생성 프로세스 실행"""
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

        chosen_idea = getattr(state, 'chosen_idea', None) # Safe access
        final_summary = getattr(state, 'final_summary', None)
        opinion_summaries = getattr(state, 'opinion_summaries', None)
        config = getattr(state, 'config', {}) or {}
        current_used_links = getattr(state, 'used_links', []) or []

        # --- Input Validation ---
        if not chosen_idea or not isinstance(chosen_idea, dict) or \
           not chosen_idea.get('idea_title') or not chosen_idea.get('concept'):
            error_message = "Valid 'chosen_idea' with title and concept is required for scenario writing."
            logger.error(error_message, extra=extra_log_data)
            end_time = datetime.now(timezone.utc)
            node15_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "scenarios": [], "scenario_prompt": None, "used_links": current_used_links,
                "node15_processing_stats": node15_processing_stats, "error_message": error_message
            }
            # --- ADDED: End Logging (Error Case) ---
            logger.debug(f"Returning updates (invalid idea):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Error: Invalid Idea) --- (Elapsed: {node15_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}

        if not final_summary:
            logger.warning("Final summary is empty, scenario context might be weak.", extra=extra_log_data)

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        # --------------------------

        logger.info(f"Starting scenario generation for idea: '{chosen_idea.get('idea_title', 'N/A')}'", extra=extra_log_data)
        error_message: Optional[str] = None
        scenarios: List[Dict[str, Any]] = []
        scenario_prompt: Optional[str] = None

        try:
            context = self._prepare_context_for_prompt(final_summary, opinion_summaries)
            scenario_prompt = self._create_scenario_prompt_en(
                chosen_idea['idea_title'], chosen_idea['concept'], context
            )
            llm_kwargs = {"response_format": {"type": "json_object"}}

            # Pass IDs
            response_str = await self._call_llm_with_retry(
                prompt=scenario_prompt,
                temperature=self.llm_temp_creative,
                max_tokens=self.max_tokens_scenario,
                trace_id=trace_id,
                comic_id=comic_id, # Pass ID
                **llm_kwargs
            )
            # Pass IDs
            scenarios = self._parse_llm_response(response_str, trace_id, comic_id)

            if not scenarios:
                 # Parser already logged details
                 error_message = "Failed to generate or parse a valid 4-panel scenario from LLM."
                 scenarios = []
            else:
                 logger.info("Successfully generated 4-panel scenario.", extra=extra_log_data)

        except RetryError as e:
            error_message = f"Scenario generation LLM call failed after multiple retries: {e}"
            logger.error(error_message, extra=extra_log_data)
            scenarios = []
            scenario_prompt = scenario_prompt # Keep prompt even on failure if generated
        except Exception as e:
            error_message = f"Scenario generation failed: {str(e)}"
            logger.exception("Unexpected error during scenario generation.", extra=extra_log_data) # Use exception
            scenarios = []
            scenario_prompt = scenario_prompt

        # --- Update used links (Placeholder logic) ---
        # Pass IDs
        updated_used_links = self._update_context_links(state, trace_id, comic_id)

        end_time = datetime.now(timezone.utc)
        node15_processing_stats = (end_time - start_time).total_seconds()

        update_data: Dict[str, Any] = {
            "scenarios": scenarios,
            "scenario_prompt": scenario_prompt,
            "used_links": updated_used_links,
            "node15_processing_stats": node15_processing_stats,
            "error_message": error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if error_message or not scenarios else logger.info
        log_level(f"Scenario generation result: {'Failed' if error_message or not scenarios else 'Success'}. Errors: {error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node15_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}