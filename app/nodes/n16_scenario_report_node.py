# app/nodes/16_scenario_report_node.py (Refactored)

import os
import re
import json
import hashlib
import asyncio
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

JINJA2_AVAILABLE = False
template_env_cache: Optional[Dict[str, 'jinja2.Environment']] = {}
try:
    import jinja2
    JINJA2_AVAILABLE = True
    logger.info("jinja2 library found.")
except ImportError:
    jinja2 = None
    logger.error("jinja2 library not installed. Report generation disabled.")
except Exception as e:
    jinja2 = None
    logger.exception(f"Error importing jinja2: {e}")

class ScenarioReportNode:
    """
    시나리오 중간 보고서(Markdown, Template B)를 생성합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = [
        "scenarios", "chosen_idea", "fact_urls", "opinion_urls", "used_links",
        "timestamp", "trace_id", "comic_id", "config", "scenario_prompt"
    ]
    outputs: List[str] = ["scenario_report", "node16_processing_stats", "error_message"]

    def __init__(self, llm_client: Optional[LLMService] = None):
        self.llm_client = llm_client
        logger.info(f"ScenarioReportNode initialized {'with' if llm_client else 'without'} LLMService.")
        if not JINJA2_AVAILABLE:
            logger.error("Report generation disabled due to missing Jinja2 library.")

    # --- MODIFIED: Added extra_log_data argument ---
    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.template_dir = config.get("template_dir", settings.DEFAULT_TEMPLATE_DIR)
        self.template_name = config.get("progress_report_template_b_filename", settings.DEFAULT_TEMPLATE_B_FILENAME)
        self.enable_scenario_evaluation = config.get("enable_scenario_evaluation", settings.ENABLE_SCENARIO_EVALUATION)
        self.llm_temp_scenario_eval = float(config.get("llm_temperature_scenario_eval", settings.DEFAULT_LLM_TEMP_SCENARIO_EVAL))
        self.max_tokens_scenario_eval = int(config.get("llm_max_tokens_scenario_eval", settings.DEFAULT_MAX_TOKENS_SCENARIO_EVAL))
        self.max_title_len = config.get("report_max_title_len", 40)
        self.max_panel_text_len = config.get("report_max_panel_text_len", 25)
        self.max_scenario_eval_len = config.get("max_scenario_eval_len", 2000)

        logger.debug(f"Runtime config loaded. TemplateDir: {self.template_dir}, TemplateName: {self.template_name}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Scenario Eval Enabled: {self.enable_scenario_evaluation}", extra=extra_log_data) # MODIFIED
        if self.enable_scenario_evaluation:
             logger.debug(f"Eval LLM Temp: {self.llm_temp_scenario_eval}, Max Tokens: {self.max_tokens_scenario_eval}", extra=extra_log_data) # MODIFIED

    # --- MODIFIED: Added extra_log_data argument ---
    def _setup_jinja_env(self, trace_id: Optional[str], comic_id: Optional[str]) -> Optional['jinja2.Environment']:
        """설정된 경로로 Jinja2 환경 로드 또는 캐시된 환경 반환"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not JINJA2_AVAILABLE or not self.template_dir: return None
        if self.template_dir in template_env_cache:
            logger.debug("Using cached Jinja2 environment.", extra=extra_log_data) # MODIFIED
            return template_env_cache[self.template_dir]
        try:
            if not os.path.isdir(self.template_dir):
                logger.error(f"Jinja2 template directory not found: {self.template_dir}", extra=extra_log_data) # MODIFIED
                return None
            loader = jinja2.FileSystemLoader(searchpath=self.template_dir)
            env = jinja2.Environment(
                loader=loader,
                autoescape=jinja2.select_autoescape(['html', 'xml', 'md']),
                undefined=jinja2.StrictUndefined
            )
            logger.info(f"Jinja2 environment loaded from: {self.template_dir}", extra=extra_log_data) # MODIFIED
            template_env_cache[self.template_dir] = env
            return env
        except Exception as e:
            logger.exception(f"Error initializing Jinja2 environment from {self.template_dir}", extra=extra_log_data) # MODIFIED
            return None

    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES or 2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float, max_tokens: int,
                                   trace_id: Optional[str] = None, comic_id: Optional[str] = None, **kwargs) -> str: # MODIFIED: Added comic_id
        """LLMService.generate_text를 재시도 로직과 함께 호출 (평가/제안용)"""
        if not self.llm_client: raise RuntimeError("LLM client is not available for scenario evaluation.")
        llm_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        logger.debug(f"Calling LLMService for Report Eval/Suggest (Temp: {temperature}, MaxTokens: {max_tokens})...", extra=llm_log_data) # MODIFIED
        result = await self.llm_client.generate_text(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens,
            # trace_id=trace_id, comic_id=comic_id, # Pass if supported
            **kwargs
        )
        if "error" in result:
            logger.error(f"LLM Eval/Suggest call failed: {result['error']}", extra=llm_log_data) # MODIFIED
            raise RuntimeError(f"LLMService error: {result['error']}")
        if "generated_text" not in result or not isinstance(result.get("generated_text"), str) or not result["generated_text"].strip():
            logger.error(f"LLM Eval/Suggest returned invalid text. Response: {result}", extra=llm_log_data) # MODIFIED
            raise ValueError("LLMService returned invalid or empty text")
        return result["generated_text"].strip()

    # --- MODIFIED: Added extra_log_data argument ---
    async def _evaluate_scenario_quality(self, scenario_text: str, trace_id: Optional[str], comic_id: Optional[str]) -> Dict[str, int]:
        """LLM으로 시나리오 품질 평가 (점수: 1-5)"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        default_scores = {"consistency": 0, "flow": 0, "dialogue": 0}
        if not scenario_text or not self.llm_client: return default_scores

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a script evaluator. Analyze the provided 4-panel comic scenario based on consistency, flow, and dialogue quality. Respond ONLY with a JSON object containing integer scores from 1 (Poor) to 5 (Excellent) for each category.<|eot_id|><|start_header_id|>user<|end_header_id|>
Evaluate the quality of the following 4-panel comic scenario based on the criteria below.

[Scenario Text]
{scenario_text[:self.max_scenario_eval_len]}

[Evaluation Criteria]
1.  **Consistency**: Are the characters, setting, and tone consistent across the 4 panels?
2.  **Flow**: Does the story progress logically and smoothly from panel to panel? Is there a clear beginning, middle, and end (even if simple)?
3.  **Dialogue**: Is the dialogue (or caption) concise, engaging, and appropriate for the characters and scene?

[Instructions]
- Assign an integer score from 1 (Poor) to 5 (Excellent) for each criterion.
- Respond ONLY with a single, valid JSON object in the format: `{{"consistency": score, "flow": score, "dialogue": score}}`

[Evaluation Scores]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
        response_str = "" # Initialize for error logging
        try:
            response_str = await self._call_llm_with_retry(
                prompt, self.llm_temp_scenario_eval, self.max_tokens_scenario_eval, trace_id, comic_id, # Pass IDs
                response_format={"type": "json_object"}
            )
            # Improved JSON extraction
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_str, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            else:
                start = response_str.find('{')
                end = response_str.rfind('}')
                if start != -1 and end != -1 and end > start: json_str = response_str[start:end+1]
                else: raise ValueError("Could not extract JSON object from response.")

            eval_data = json.loads(json_str)
            scores = default_scores.copy()
            for key in scores.keys():
                 score_val = eval_data.get(key)
                 if isinstance(score_val, int): scores[key] = max(1, min(5, score_val)) # Clamp between 1 and 5
                 else: logger.warning(f"Invalid score type for '{key}': {score_val}. Using 0.", extra=extra_log_data) # MODIFIED
            logger.info(f"Scenario quality evaluated by LLM: {scores}", extra=extra_log_data) # MODIFIED
            return scores
        except RetryError as e:
            logger.error(f"LLM evaluation failed after retries: {e}", extra=extra_log_data) # MODIFIED
            return default_scores
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed parsing LLM evaluation response: {e}. Response: '{response_str[:100]}...'", extra=extra_log_data) # MODIFIED
            return default_scores
        except Exception as e:
            logger.exception("Unexpected error during scenario evaluation.", extra=extra_log_data) # MODIFIED use exception
            return default_scores

    # --- MODIFIED: Added extra_log_data argument ---
    async def _generate_suggestions(self, scenario_text: str, trace_id: Optional[str], comic_id: Optional[str]) -> List[str]:
         """LLM으로 시나리오 개선 제안 생성"""
         extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
         default_suggestion = ["Failed to generate suggestions."]
         if not scenario_text or not self.llm_client: return default_suggestion

         prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful script doctor. Provide constructive suggestions to improve the following 4-panel comic scenario.<|eot_id|><|start_header_id|>user<|end_header_id|>
Review the following 4-panel comic scenario and provide 2-3 specific, actionable suggestions for improvement. Focus on clarity, engagement, visual interest, or humor.

[Scenario Text]
{scenario_text[:self.max_scenario_eval_len]}

[Instructions]
- Provide 2 to 3 concrete suggestions.
- Format suggestions as a bulleted list (using '- '). Start each suggestion on a new line.

[Improvement Suggestions]
- <|eot_id|><|start_header_id|>assistant<|end_header_id|>
- """
         response_str = "" # Initialize for error logging
         try:
              response_str = await self._call_llm_with_retry(
                  prompt, self.llm_temp_scenario_eval, self.max_tokens_scenario_eval, trace_id, comic_id # Pass IDs
              )
              # Improved parsing for bullet points
              suggestions = [line.strip('-* ').strip() for line in response_str.split('\n')
                             if line.strip().startswith(('-', '*')) and line.strip('-* ').strip()]
              if suggestions:
                  logger.info(f"Generated {len(suggestions)} improvement suggestions.", extra=extra_log_data) # MODIFIED
                  return suggestions
              else:
                  logger.warning(f"LLM did not return suggestions in expected format. Response: '{response_str[:100]}...'", extra=extra_log_data) # MODIFIED
                  return ["No specific suggestions generated."] # More informative default
         except RetryError as e:
              logger.error(f"LLM suggestion generation failed after retries: {e}", extra=extra_log_data) # MODIFIED
              return default_suggestion
         except Exception as e:
              logger.exception("Failed to generate suggestions using LLM.", extra=extra_log_data) # MODIFIED use exception
              return default_suggestion

    def _truncate_text(self, text: Optional[str], max_length: int) -> str:
        if not text: return ""
        return text[:max_length - 3] + "..." if len(text) > max_length else text

    # --- MODIFIED: Added extra_log_data argument ---
    def _prepare_mapping_rows(self, chosen_idea: Optional[Dict], scenarios: List[Dict], trace_id: Optional[str], comic_id: Optional[str]) -> List[Dict]:
        """보고서용 매핑 테이블 데이터 준비"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        rows = []
        default_row = {'num': 1, 'title': 'N/A', 'c1': '-', 'c2': '-', 'c3': '-', 'c4': '-'}
        # Check chosen_idea is a dict
        if not chosen_idea or not isinstance(chosen_idea, dict) or \
           not scenarios or not isinstance(scenarios, list) or len(scenarios) != 4: # Added list type check
             logger.warning("Cannot prepare mapping rows: invalid chosen_idea or scenarios.", extra=extra_log_data) # MODIFIED
             return [default_row]

        title = self._truncate_text(chosen_idea.get('idea_title', 'N/A'), self.max_title_len)
        try:
             # Ensure panels are dicts before accessing keys
             get_panel_text = lambda p: p.get('panel_description', '') or p.get('dialogue', '-') if isinstance(p, dict) else '-'
             c1 = self._truncate_text(get_panel_text(scenarios[0]), self.max_panel_text_len)
             c2 = self._truncate_text(get_panel_text(scenarios[1]), self.max_panel_text_len)
             c3 = self._truncate_text(get_panel_text(scenarios[2]), self.max_panel_text_len)
             c4 = self._truncate_text(get_panel_text(scenarios[3]), self.max_panel_text_len)
             rows.append({'num': 1, 'title': title, 'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4})
        except IndexError:
             logger.error("Error accessing scenario panels while preparing mapping rows.", extra=extra_log_data) # MODIFIED
             rows.append(default_row) # Return default on error
        return rows

    # --- MODIFIED: Added extra_log_data argument ---
    def _calculate_link_usage(self, fact_urls: List[Dict], opinion_urls: List[Dict], used_links: List[Dict], trace_id: Optional[str], comic_id: Optional[str]) -> Dict[str, int]:
        """시나리오 컨텍스트 링크 사용량 계산"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        # Ensure inputs are lists
        fact_urls = fact_urls if isinstance(fact_urls, list) else []
        opinion_urls = opinion_urls if isinstance(opinion_urls, list) else []
        used_links = used_links if isinstance(used_links, list) else []

        usage = {'used_news': 0, 'total_news': len(fact_urls), 'used_op': 0, 'total_op': len(opinion_urls)}
        if not used_links: return usage

        # Filter for valid links marked as context_used
        context_urls = set(link.get('url') for link in used_links
                           if isinstance(link, dict) and link.get('status') == 'context_used' and isinstance(link.get('url'), str))

        if not context_urls: return usage

        # Ensure source URL lists contain valid dicts with URLs
        original_fact_urls = set(f.get('url') for f in fact_urls if isinstance(f, dict) and isinstance(f.get('url'), str))
        original_opinion_urls = set(o.get('url') for o in opinion_urls if isinstance(o, dict) and isinstance(o.get('url'), str))

        usage['used_news'] = len(context_urls.intersection(original_fact_urls))
        usage['used_op'] = len(context_urls.intersection(original_opinion_urls))
        logger.debug(f"Link usage calculated: News {usage['used_news']}/{usage['total_news']}, Opinion {usage['used_op']}/{usage['total_op']}", extra=extra_log_data) # MODIFIED
        return usage

    def _calculate_prompt_hash(self, prompt: Optional[str]) -> str:
        if not prompt or not isinstance(prompt, str): return "N/A" # Added type check
        try: return hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
        except Exception as e:
             logger.error(f"Error calculating prompt hash: {e}")
             return "Error Calculating Hash"
    # --- Node 13과 동일한 파일 저장 함수 추가 ---

    def _save_report_to_file(self, report_content: str, comic_id: str, filename_prefix: str, extra_log_data: Dict):
        """보고서 내용을 로컬 파일로 저장"""
        if not report_content or not comic_id:
            logger.warning("Report content or comic_id missing, skipping file save.", extra=extra_log_data)
            return False
        try:
            output_dir = settings.REPORT_OUTPUT_DIR
            if not output_dir:
                logger.warning("REPORT_OUTPUT_DIR not configured in settings, skipping file save.",
                               extra=extra_log_data)
                return False
            os.makedirs(output_dir, exist_ok=True)
            file_name = f"{filename_prefix}_{comic_id}.md"
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Report successfully saved to: {file_path}", extra=extra_log_data)
            return True
        except Exception as e:
            logger.error(f"Failed to save report to file '{file_path}': {e}", exc_info=True, extra=extra_log_data)
            return False

    # -------------------------------------------

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """시나리오 보고서 생성 실행"""
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

        config = getattr(state, 'config', {}) or {}
        report_content = "# Scenario Report Generation Failed\n\nInitial error."
        error_message: Optional[str] = None

        # --- Input Validation ---
        scenarios = getattr(state, 'scenarios', None)
        chosen_idea = getattr(state, 'chosen_idea', None)
        if not scenarios or not isinstance(scenarios, list) or len(scenarios) != 4:
             error_message = "Valid 4-panel 'scenarios' list is required for report generation."
             logger.error(error_message, extra=extra_log_data)
        elif not chosen_idea or not isinstance(chosen_idea, dict):
             error_message = "Valid 'chosen_idea' dictionary is required for report generation."
             logger.error(error_message, extra=extra_log_data)
        # ------------------------

        if error_message: # Exit early if required inputs are missing/invalid
            report_content = f"# Scenario Report Generation Failed\n\n{error_message}"
            end_time = datetime.now(timezone.utc)
            node16_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "scenario_report": report_content,
                "node16_processing_stats": node16_processing_stats,
                "error_message": error_message
            }
            # --- ADDED: End Logging (Error Case) ---
            logger.debug(f"Returning updates (input error):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Input Error) --- (Elapsed: {node16_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        # --------------------------

        # Pass IDs
        jinja_env = self._setup_jinja_env(trace_id, comic_id)
        if not JINJA2_AVAILABLE or not jinja_env:
            error_message = "Jinja2 library or environment not available."
            report_content = f"# Scenario Report Generation Failed\n\n{error_message}"
        elif not self.template_name:
            error_message = "Template B filename not configured."
            report_content = f"# Scenario Report Generation Failed\n\n{error_message}"
        else:
            logger.info(f"Generating scenario report using template: {self.template_name}", extra=extra_log_data) # MODIFIED
            try:
                template = jinja_env.get_template(self.template_name)
                scenario_full_text = ""
                # Create full text only if scenarios are valid list of dicts
                if isinstance(scenarios, list) and all(isinstance(p, dict) for p in scenarios):
                    panel_texts = [f"Panel {p.get('scene', i+1)}: Desc: {p.get('panel_description', '')} Dialogue: {p.get('dialogue', '')}"
                                   for i, p in enumerate(scenarios)]
                    scenario_full_text = "\n\n".join(panel_texts)

                quality_scores = {"consistency": 0, "flow": 0, "dialogue": 0}
                suggestions = ["Evaluation disabled or failed."]
                if self.enable_scenario_evaluation and self.llm_client and scenario_full_text:
                     logger.info("Performing LLM scenario evaluation/suggestion...", extra=extra_log_data) # MODIFIED
                     # Pass IDs
                     eval_task = self._evaluate_scenario_quality(scenario_full_text, trace_id, comic_id)
                     sugg_task = self._generate_suggestions(scenario_full_text, trace_id, comic_id)
                     eval_results = await asyncio.gather(eval_task, sugg_task, return_exceptions=True)
                     if isinstance(eval_results[0], dict): quality_scores = eval_results[0]
                     else: logger.error(f"Evaluation task failed: {eval_results[0]}", exc_info=isinstance(eval_results[0], Exception) and eval_results[0], extra=extra_log_data) # MODIFIED
                     if isinstance(eval_results[1], list): suggestions = eval_results[1]
                     else: logger.error(f"Suggestion task failed: {eval_results[1]}", exc_info=isinstance(eval_results[1], Exception) and eval_results[1], extra=extra_log_data) # MODIFIED
                elif not scenario_full_text:
                     logger.warning("Skipping LLM scenario evaluation/suggestion because scenario text could not be generated.", extra=extra_log_data) # MODIFIED
                else: logger.info("Skipping LLM scenario evaluation/suggestion (disabled or LLM client missing).", extra=extra_log_data) # MODIFIED

                timestamp_str = getattr(state, 'timestamp', None) or datetime.now(timezone.utc).isoformat()
                formatted_timestamp = timestamp_str
                try: formatted_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S %Z")
                except ValueError: pass

                # Pass IDs
                context = {
                    "timestamp": formatted_timestamp,
                    "trace_id": trace_id,
                    "comic_id": comic_id, # Add comic_id
                    "mapping_rows": self._prepare_mapping_rows(chosen_idea, scenarios, trace_id, comic_id),
                    "chosen_title": self._truncate_text(chosen_idea.get('idea_title', 'N/A'), self.max_title_len),
                    "link_usage": self._calculate_link_usage(state.fact_urls or [], state.opinion_urls or [], state.used_links or [], trace_id, comic_id),
                    "quality_scores": quality_scores,
                    "suggestions": suggestions,
                    "prompt_hash": self._calculate_prompt_hash(getattr(state, 'scenario_prompt', None)) # Safe access
                }
                logger.debug("Scenario report template data prepared.", extra=extra_log_data) # MODIFIED
                report_content = template.render(**context)
                logger.info("Scenario report generated successfully.", extra=extra_log_data) # MODIFIED

                # --- *** 보고서 파일 저장 시도 *** ---
                report_saved = self._save_report_to_file(
                    report_content,
                    comic_id,
                    "16 node report", # 파일명 접두사
                    extra_log_data
                )
                # ---------------------------------

            except jinja2.TemplateNotFound:
                error_message = f"Template '{self.template_name}' not found in '{self.template_dir}'."
                logger.error(error_message, extra=extra_log_data) # MODIFIED
                report_content = f"# Report Generation Failed\n\n{error_message}"
            except Exception as e:
                error_message = f"Failed to prepare/render template '{self.template_name}': {str(e)}"
                logger.exception("Template rendering error.", extra=extra_log_data) # MODIFIED
                report_content = f"# Scenario Report Generation Error\n\n{error_message}"

        end_time = datetime.now(timezone.utc)
        node16_processing_stats = (end_time - start_time).total_seconds()

        update_data: Dict[str, Any] = {
            "scenario_report": report_content,
            "node16_processing_stats": node16_processing_stats,
            "error_message": error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if error_message else logger.info
        log_level(f"Scenario report result: {'Failed' if error_message else 'Success'}. Errors: {error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node16_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}