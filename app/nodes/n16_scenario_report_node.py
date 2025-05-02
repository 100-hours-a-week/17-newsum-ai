# app/nodes/16_scenario_report_node.py (Improved Version)

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
from app.config.settings import settings # 기본값 참조용
from app.services.llm_server_client_v2 import LLMService # LLM 서비스 클라이언트
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# 로거 설정
logger = get_logger(__name__)

# Jinja2 의존성 처리
JINJA2_AVAILABLE = False
template_env_cache: Optional[Dict[str, 'jinja2.Environment']] = {} # 경로별 캐시 (Node 13과 공유 가능)
try:
    import jinja2
    JINJA2_AVAILABLE = True
    logger.info("jinja2 library found.")
except ImportError:
    jinja2 = None # type: ignore
    logger.error("jinja2 library not installed. Report generation disabled.")
except Exception as e:
    jinja2 = None # type: ignore
    logger.exception(f"Error importing jinja2: {e}")


class ScenarioReportNode:
    """
    시나리오 중간 보고서(Markdown, Template B)를 생성합니다.
    - 시나리오 상세, 링크 사용 정보, LLM 기반 품질 평가 및 제안(선택적) 포함.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = [
        "scenarios", "chosen_idea", "fact_urls", "opinion_urls", "used_links",
        "timestamp", "trace_id", "comic_id", "config", "processing_stats", "scenario_prompt"
    ]
    outputs: List[str] = ["scenario_report", "processing_stats", "error_message"]

    # LLMService는 평가/제안 기능 활성화 시에만 필요
    def __init__(
        self,
        llm_client: Optional[LLMService] = None,
        # langsmith_service: Optional[LangSmithService] = None
    ):
        self.llm_client = llm_client
        # self.langsmith = langsmith_service
        logger.info(f"ScenarioReportNode initialized {'with' if llm_client else 'without'} LLMService.")
        if not JINJA2_AVAILABLE:
            logger.error("Report generation disabled due to missing Jinja2 library.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.template_dir = config.get("template_dir", settings.DEFAULT_TEMPLATE_DIR)
        # 템플릿 파일명 B 사용
        self.template_name = config.get("progress_report_template_b_filename", settings.DEFAULT_TEMPLATE_B_FILENAME)
        # LLM 기반 평가/제안 관련 설정
        self.enable_scenario_evaluation = config.get("enable_scenario_evaluation", settings.ENABLE_SCENARIO_EVALUATION)
        self.llm_temp_scenario_eval = float(config.get("llm_temperature_scenario_eval", settings.DEFAULT_LLM_TEMP_SCENARIO_EVAL))
        self.max_tokens_scenario_eval = int(config.get("llm_max_tokens_scenario_eval", settings.DEFAULT_MAX_TOKENS_SCENARIO_EVAL))
        # 텍스트 길이 제한 (템플릿 및 평가용)
        self.max_title_len = config.get("report_max_title_len", 40)
        self.max_panel_text_len = config.get("report_max_panel_text_len", 25)
        self.max_scenario_eval_len = config.get("max_scenario_eval_len", 2000)

        logger.debug(f"Runtime config loaded. TemplateDir: {self.template_dir}, TemplateName: {self.template_name}")
        logger.debug(f"Scenario Eval Enabled: {self.enable_scenario_evaluation}")
        if self.enable_scenario_evaluation:
             logger.debug(f"Eval LLM Temp: {self.llm_temp_scenario_eval}, Max Tokens: {self.max_tokens_scenario_eval}")


    def _setup_jinja_env(self, trace_id: Optional[str]) -> Optional['jinja2.Environment']:
        """설정된 경로로 Jinja2 환경 로드 또는 캐시된 환경 반환"""
        # Node 13과 동일한 로직 사용
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not JINJA2_AVAILABLE or not self.template_dir: return None
        if self.template_dir in template_env_cache:
            logger.debug(f"{log_prefix} Using cached Jinja2 environment for {self.template_dir}")
            return template_env_cache[self.template_dir]
        try:
            if not os.path.isdir(self.template_dir):
                logger.error(f"{log_prefix} Jinja2 template directory not found: {self.template_dir}")
                return None
            loader = jinja2.FileSystemLoader(searchpath=self.template_dir)
            env = jinja2.Environment(
                loader=loader,
                autoescape=jinja2.select_autoescape(['html', 'xml', 'md']),
                undefined=jinja2.StrictUndefined
            )
            logger.info(f"{log_prefix} Jinja2 environment loaded from: {self.template_dir}")
            template_env_cache[self.template_dir] = env
            return env
        except Exception as e:
            logger.exception(f"{log_prefix} Error initializing Jinja2 environment from {self.template_dir}: {e}")
            return None

    # --- LLM 호출 래퍼 (평가/제안용) ---
    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES or 2), # 평가용은 재시도 줄일 수 있음
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float, max_tokens: int,
                                   trace_id: Optional[str] = None, **kwargs) -> str:
        """LLMService.generate_text를 재시도 로직과 함께 호출 (평가/제안용)"""
        if not self.llm_client: raise RuntimeError("LLM client is not available for scenario evaluation.")
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Calling LLMService for Report Eval/Suggest (Temp: {temperature}, MaxTokens: {max_tokens})...")
        result = await self.llm_client.generate_text(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        if "error" in result: raise RuntimeError(f"LLMService error: {result['error']}")
        if "generated_text" not in result or not isinstance(result.get("generated_text"), str) or not result["generated_text"].strip():
            raise ValueError("LLMService returned invalid or empty text")
        return result["generated_text"].strip()

    # --- LLM 기반 평가 및 제안 ---
    async def _evaluate_scenario_quality(self, scenario_text: str, trace_id: Optional[str]) -> Dict[str, int]:
        """LLM으로 시나리오 품질 평가 (점수: 1-5)"""
        default_scores = {"consistency": 0, "flow": 0, "dialogue": 0}
        if not scenario_text or not self.llm_client: return default_scores
        log_prefix = f"[{trace_id}]" if trace_id else ""

        # 프롬프트 정의 (이전과 유사)
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
        try:
            response_str = await self._call_llm_with_retry(
                prompt, self.llm_temp_scenario_eval, self.max_tokens_scenario_eval, trace_id,
                response_format={"type": "json_object"} # JSON 모드 요청
            )
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_str, re.DOTALL | re.IGNORECASE)
            json_str = match.group(1) if match else response_str.strip()
            eval_data = json.loads(json_str)

            scores = default_scores.copy()
            for key in scores.keys():
                 score_val = eval_data.get(key)
                 if isinstance(score_val, int): scores[key] = max(1, min(5, score_val))
                 else: logger.warning(f"{log_prefix} Invalid score for '{key}': {score_val}")
            logger.info(f"{log_prefix} Scenario quality evaluated by LLM: {scores}")
            return scores
        except RetryError as e:
            logger.error(f"{log_prefix} LLM evaluation failed after retries: {e}")
            return default_scores
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"{log_prefix} Failed parsing LLM evaluation response: {e}. Response: '{response_str[:100]}...'")
            return default_scores
        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error during scenario evaluation: {e}")
            return default_scores

    async def _generate_suggestions(self, scenario_text: str, trace_id: Optional[str]) -> List[str]:
         """LLM으로 시나리오 개선 제안 생성"""
         default_suggestion = ["Failed to generate suggestions."]
         if not scenario_text or not self.llm_client: return default_suggestion
         log_prefix = f"[{trace_id}]" if trace_id else ""

         # 프롬프트 정의 (이전과 유사)
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
         try:
              response_str = await self._call_llm_with_retry(
                  prompt, self.llm_temp_scenario_eval, self.max_tokens_scenario_eval, trace_id
              )
              suggestions = [line.strip('- ').strip() for line in response_str.strip().split('\n')
                             if line.strip().startswith('-') and line.strip('- ').strip()]
              if suggestions:
                  logger.info(f"{log_prefix} Generated {len(suggestions)} improvement suggestions.")
                  return suggestions
              else:
                  logger.warning(f"{log_prefix} LLM did not return suggestions in expected format. Response: '{response_str[:100]}...'")
                  return ["No specific suggestions generated."]
         except RetryError as e:
              logger.error(f"{log_prefix} LLM suggestion generation failed after retries: {e}")
              return default_suggestion
         except Exception as e:
              logger.exception(f"{log_prefix} Failed to generate suggestions using LLM: {e}")
              return default_suggestion

    # --- 데이터 준비 헬퍼 (Node 13과 유사/재사용) ---
    def _truncate_text(self, text: Optional[str], max_length: int) -> str:
        if not text: return ""
        return text[:max_length - 3] + "..." if len(text) > max_length else text

    def _prepare_mapping_rows(self, chosen_idea: Optional[Dict], scenarios: List[Dict], trace_id: Optional[str]) -> List[Dict]:
        """보고서용 매핑 테이블 데이터 준비"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        rows = []
        default_row = {'num': 1, 'title': 'N/A', 'c1': '-', 'c2': '-', 'c3': '-', 'c4': '-'}
        if not chosen_idea or not scenarios or len(scenarios) != 4:
             logger.warning(f"{log_prefix} Cannot prepare mapping rows: chosen idea or 4 panels missing.")
             return [default_row]

        title = self._truncate_text(chosen_idea.get('idea_title', 'N/A'), self.max_title_len)
        try:
            get_panel_text = lambda p: p.get('panel_description') or p.get('dialogue', '-')
            c1 = self._truncate_text(get_panel_text(scenarios[0]), self.max_panel_text_len)
            c2 = self._truncate_text(get_panel_text(scenarios[1]), self.max_panel_text_len)
            c3 = self._truncate_text(get_panel_text(scenarios[2]), self.max_panel_text_len)
            c4 = self._truncate_text(get_panel_text(scenarios[3]), self.max_panel_text_len)
            rows.append({'num': 1, 'title': title, 'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4})
        except IndexError:
             logger.error(f"{log_prefix} Error accessing scenario panels preparing mapping rows.")
             rows.append(default_row)
        return rows

    def _calculate_link_usage(self, fact_urls: List[Dict], opinion_urls: List[Dict], used_links: List[Dict], trace_id: Optional[str]) -> Dict[str, int]:
        """시나리오 컨텍스트 링크 사용량 계산"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        usage = {'used_news': 0, 'total_news': len(fact_urls), 'used_op': 0, 'total_op': len(opinion_urls)}
        if not used_links: return usage
        # Node 15에서 'context_used' 상태로 업데이트된 링크 필터링
        context_urls = set(link.get('url') for link in used_links if link.get('status') == 'context_used' and link.get('url'))
        if not context_urls: return usage

        original_fact_urls = set(f.get('url') for f in fact_urls if f.get('url'))
        original_opinion_urls = set(o.get('url') for o in opinion_urls if o.get('url'))
        usage['used_news'] = len(context_urls.intersection(original_fact_urls))
        usage['used_op'] = len(context_urls.intersection(original_opinion_urls))
        logger.debug(f"{log_prefix} Link usage calculated: News {usage['used_news']}/{usage['total_news']}, Opinion {usage['used_op']}/{usage['total_op']}")
        return usage

    def _calculate_prompt_hash(self, prompt: Optional[str]) -> str:
        """시나리오 생성 프롬프트 해시 계산"""
        if not prompt: return "N/A"
        try: return hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
        except Exception: return "Error Calculating Hash"

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """시나리오 보고서 생성 실행"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing ScenarioReportNode...")

        config = state.config or {}
        processing_stats = state.processing_stats or {}
        report_content = "# Scenario Report Generation Failed\n\nInitial error."
        error_message: Optional[str] = None

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        # --------------------------

        jinja_env = self._setup_jinja_env(trace_id)
        if not JINJA2_AVAILABLE or not jinja_env:
            error_message = "Jinja2 library or environment not available."
            report_content = f"# Scenario Report Generation Failed\n\n{error_message}"
        elif not self.template_name:
            error_message = "Template B filename not configured."
            report_content = f"# Scenario Report Generation Failed\n\n{error_message}"
        else:
            logger.info(f"{log_prefix} Generating scenario report using template: {self.template_name}")
            try:
                template = jinja_env.get_template(self.template_name)
                # --- 템플릿 데이터 준비 ---
                logger.debug(f"{log_prefix} Preparing data for scenario report template...")
                scenarios = state.scenarios or []
                chosen_idea = state.chosen_idea

                scenario_full_text = ""
                if scenarios and len(scenarios) == 4:
                    # 텍스트 조합 (길이 제한 적용)
                    panel_texts = [f"Panel {p.get('scene', i+1)}: Desc: {p.get('panel_description', '')} Dialogue: {p.get('dialogue', '')}"
                                   for i, p in enumerate(scenarios)]
                    scenario_full_text = "\n\n".join(panel_texts)

                quality_scores = {"consistency": 0, "flow": 0, "dialogue": 0}
                suggestions = ["Evaluation disabled or failed."]
                if self.enable_scenario_evaluation and self.llm_client and scenario_full_text:
                     logger.info(f"{log_prefix} Performing LLM scenario evaluation/suggestion...")
                     eval_task = self._evaluate_scenario_quality(scenario_full_text, trace_id)
                     sugg_task = self._generate_suggestions(scenario_full_text, trace_id)
                     eval_results = await asyncio.gather(eval_task, sugg_task, return_exceptions=True)
                     if isinstance(eval_results[0], dict): quality_scores = eval_results[0]
                     else: logger.error(f"{log_prefix} Evaluation task failed: {eval_results[0]}", exc_info=isinstance(eval_results[0], Exception) and eval_results[0])
                     if isinstance(eval_results[1], list): suggestions = eval_results[1]
                     else: logger.error(f"{log_prefix} Suggestion task failed: {eval_results[1]}", exc_info=isinstance(eval_results[1], Exception) and eval_results[1])
                else: logger.info(f"{log_prefix} Skipping LLM scenario evaluation/suggestion.")

                timestamp_str = state.timestamp or datetime.now(timezone.utc).isoformat()
                formatted_timestamp = timestamp_str
                try: formatted_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S %Z")
                except ValueError: pass

                context = {
                    "timestamp": formatted_timestamp,
                    "trace_id": trace_id,
                    "mapping_rows": self._prepare_mapping_rows(chosen_idea, scenarios, trace_id),
                    "chosen_title": self._truncate_text(chosen_idea.get('idea_title', 'N/A') if chosen_idea else 'N/A', self.max_title_len),
                    "link_usage": self._calculate_link_usage(state.fact_urls or [], state.opinion_urls or [], state.used_links or [], trace_id),
                    "quality_scores": quality_scores,
                    "suggestions": suggestions,
                    "prompt_hash": self._calculate_prompt_hash(state.scenario_prompt)
                }
                logger.debug(f"{log_prefix} Scenario report template data prepared.")
                report_content = template.render(**context)
                logger.info(f"{log_prefix} Scenario report generated successfully.")

            except jinja2.TemplateNotFound:
                error_message = f"Template '{self.template_name}' not found in '{self.template_dir}'."
                logger.error(f"{log_prefix} {error_message}")
                report_content = f"# Report Generation Failed\n\n{error_message}"
            except Exception as e:
                error_message = f"Failed to prepare/render template '{self.template_name}': {str(e)}"
                logger.exception(f"{log_prefix} Template rendering error:", exc_info=e)
                report_content = f"# Scenario Report Generation Error\n\n{error_message}"

        if error_message: logger.error(f"{log_prefix} Scenario report generation failed: {error_message}")

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['scenario_report_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} ScenarioReportNode finished in {processing_stats['scenario_report_node_time']:.2f} seconds.")

        # ComicState 업데이트
        update_data: Dict[str, Any] = {
            "scenario_report": report_content,
            "processing_stats": processing_stats,
            "error_message": error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}