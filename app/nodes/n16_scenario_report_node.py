# app/nodes/16_scenario_report_node.py

import os
import re
import json
import hashlib
# --- datetime, timezone 임포트 추가 ---
from datetime import datetime, timezone
# ------------------------------------
from typing import List, Dict, Any, Optional

# --- 프로젝트 구성 요소 임포트 ---
from app.config.settings import settings            # 설정 객체 (재시도 횟수 등 참조)
from app.services.llm_service_v2 import LLMService # 실제 LLM 서비스 클라이언트
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# --- 로거 설정 ---
logger = get_logger("ScenarioReportNode")

# --- Jinja2 의존성 처리 ---
JINJA2_AVAILABLE = False
template_env_cache: Optional['jinja2.Environment'] = None # 캐시된 환경 객체
template_dir_cache: Optional[str] = None # 캐시된 경로
try:
    import jinja2
    JINJA2_AVAILABLE = True
    logger.info("jinja2 library found.")
except ImportError:
    logger.error("jinja2 library not installed. Report generation disabled.")
except Exception as e:
    logger.exception(f"Error importing jinja2: {e}")

class ScenarioReportNode:
    """
    (Refactored) 시나리오 중간 보고서(Markdown 템플릿 B)를 생성합니다.
    - 시나리오 상세, 링크 사용 정보, LLM 기반 품질 평가 및 제안(선택적) 포함.
    - LLMService (평가/제안용), Jinja2 사용.
    - 설정은 상태의 config 딕셔너리에서 로드.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = [
        "scenarios", "chosen_idea", "fact_urls", "opinion_urls", "used_links",
        "timestamp", "trace_id", "config", "processing_stats", "scenario_prompt"
    ]
    outputs: List[str] = ["scenario_report", "processing_stats", "error_message"]

    # LLMService 인스턴스를 외부에서 주입받음 (평가/제안이 활성화된 경우 필요)
    def __init__(
        self,
        llm_client: Optional[LLMService] = None, # LLM 클라이언트는 선택적일 수 있음
        # langsmith_service: Optional[LangSmithService] = None
    ):
        self.llm_client = llm_client
        # self.langsmith_service = langsmith_service
        # Jinja2 환경은 run 시점에 경로 기반으로 로드/캐시 확인
        self.template_env: Optional[jinja2.Environment] = None
        logger.info(f"ScenarioReportNode initialized {'with' if llm_client else 'without'} LLMService.")
        if not JINJA2_AVAILABLE:
            logger.error("Report generation will be disabled due to missing Jinja2 library.")

    def _setup_jinja_env(self, template_dir: str) -> bool:
        """설정된 경로로 Jinja2 환경 로드 또는 캐시된 환경 반환"""
        global template_env_cache, template_dir_cache # 전역 캐시 사용
        if not JINJA2_AVAILABLE: return False

        # 경로가 같고 캐시된 환경이 있으면 재사용
        if template_env_cache and template_dir_cache == template_dir:
            self.template_env = template_env_cache
            return True

        # 새로 로드
        try:
            if not os.path.isdir(template_dir):
                logger.error(f"Jinja2 template directory not found: {template_dir}")
                self.template_env = None; template_env_cache = None; template_dir_cache = None
                return False

            loader = jinja2.FileSystemLoader(searchpath=template_dir)
            env = jinja2.Environment(
                loader=loader,
                autoescape=jinja2.select_autoescape(['html', 'xml', 'md']),
                undefined=jinja2.StrictUndefined
            )
            logger.info(f"Jinja2 environment loaded from: {template_dir}")
            self.template_env = env
            template_env_cache = env # 캐시 업데이트
            template_dir_cache = template_dir
            return True
        except Exception as e:
            logger.exception(f"Error initializing Jinja2 environment from {template_dir}: {e}")
            self.template_env = None; template_env_cache = None; template_dir_cache = None
            return False

    # --- LLM 호출 래퍼 (평가/제안용) ---
    # 이전 노드들의 _call_llm_with_retry 와 동일하게 사용 가능
    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES or 2), # 평가용은 재시도 줄일 수 있음
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float, max_tokens: int,
                                   trace_id: Optional[str] = None, **kwargs) -> str:
        """LLMService.generate_text를 재시도 로직과 함께 호출 (평가/제안용)"""
        if not self.llm_client: raise RuntimeError("LLM client is not available for this node.") # LLM 없으면 오류

        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Calling LLMService for Report Eval/Suggest (Temp: {temperature}, MaxTokens: {max_tokens})...")

        result = await self.llm_client.generate_text(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        if "error" in result:
            error_msg = result['error']
            logger.error(f"{log_prefix} LLMService call failed: {error_msg}")
            raise RuntimeError(f"LLMService error: {error_msg}")
        elif "generated_text" not in result or not isinstance(result.get("generated_text"), str) or not result["generated_text"].strip():
            logger.error(f"{log_prefix} LLMService returned invalid/empty text. Response: {result}")
            raise ValueError("LLMService returned invalid or empty text")
        else:
            logger.debug(f"{log_prefix} LLMService call successful.")
            return result["generated_text"].strip()

    # --- LLM 기반 평가 및 제안 (선택적) ---
    async def _evaluate_scenario_quality(self, scenario_text: str, config: Dict, trace_id: Optional[str]) -> Dict[str, int]:
        """LLM으로 시나리오 품질 평가 (점수: 1-5)"""
        default_scores = {"consistency": 0, "flow": 0, "dialogue": 0}
        if not scenario_text or not self.llm_client: return default_scores

        # 설정 로드
        llm_model = config.get("llm_model", "default_model")
        temperature = config.get("llm_temperature_scenario_eval", 0.3)
        max_tokens = config.get("llm_max_tokens_scenario_eval", 512)

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a script evaluator. Analyze the provided 4-panel comic scenario based on consistency, flow, and dialogue quality. Respond ONLY with a JSON object containing integer scores from 1 (Poor) to 5 (Excellent) for each category.<|eot_id|><|start_header_id|>user<|end_header_id|>
Evaluate the quality of the following 4-panel comic scenario based on the criteria below.

[Scenario Text]
{scenario_text}

[Evaluation Criteria]
1.  **Consistency**: Are the characters, setting, and tone consistent across the 4 panels?
2.  **Flow**: Does the story progress logically and smoothly from panel to panel? Is there a clear beginning, middle, and end (even if simple)?
3.  **Dialogue**: Is the dialogue (or caption) concise, engaging, and appropriate for the characters and scene?

[Instructions]
- Assign an integer score from 1 (Poor) to 5 (Excellent) for each criterion.
- Respond ONLY with a single, valid JSON object in the format:
  `{{"consistency": score, "flow": score, "dialogue": score}}`

[Evaluation Scores]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
        try:
            # JSON 모드 요청
            response_str = await self._call_llm_with_retry(
                prompt, temperature, max_tokens, trace_id, response_format={"type": "json_object"}
            )
            # JSON 파싱
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_str, re.DOTALL | re.IGNORECASE)
            json_str = match.group(1) if match else response_str.strip()
            eval_data = json.loads(json_str)

            scores = default_scores.copy() # 기본값으로 시작
            # 점수 검증 및 범위 제한 (1-5)
            for key in scores.keys():
                 score_val = eval_data.get(key)
                 if isinstance(score_val, int):
                      scores[key] = max(1, min(5, score_val))
                 else:
                      logger.warning(f"[{trace_id}] Invalid or missing score for '{key}' in LLM evaluation: {score_val}")
                      scores[key] = 0 # 유효하지 않으면 0점 처리

            logger.info(f"[{trace_id}] Scenario quality evaluated by LLM: {scores}")
            return scores
        except Exception as e:
            logger.error(f"[{trace_id}] Failed to evaluate scenario quality using LLM: {e}")
            return default_scores # 오류 시 기본 점수 반환

    async def _generate_suggestions(self, scenario_text: str, config: Dict, trace_id: Optional[str]) -> List[str]:
         """LLM으로 시나리오 개선 제안 생성"""
         default_suggestion = ["Failed to generate suggestions."]
         if not scenario_text or not self.llm_client: return default_suggestion

         # 설정 로드
         llm_model = config.get("llm_model", "default_model")
         temperature = config.get("llm_temperature_scenario_eval", 0.3)
         max_tokens = config.get("llm_max_tokens_scenario_eval", 512)

         prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful script doctor. Provide constructive suggestions to improve the following 4-panel comic scenario.<|eot_id|><|start_header_id|>user<|end_header_id|>
Review the following 4-panel comic scenario and provide 2-3 specific, actionable suggestions for improvement. Focus on clarity, engagement, visual interest, or humor.

[Scenario Text]
{scenario_text}

[Instructions]
- Provide 2 to 3 concrete suggestions.
- Format suggestions as a bulleted list (using '- '). Start each suggestion on a new line.

[Improvement Suggestions]
- <|eot_id|><|start_header_id|>assistant<|end_header_id|>
- """
         try:
              response_str = await self._call_llm_with_retry(prompt, temperature, max_tokens, trace_id)
              # 응답 파싱 (줄바꿈 기준, '-' 시작 필터링)
              suggestions = [line.strip('- ').strip() for line in response_str.strip().split('\n')
                             if line.strip().startswith('-')]
              if suggestions:
                  logger.info(f"[{trace_id}] Generated {len(suggestions)} improvement suggestions.")
                  return suggestions
              else:
                  logger.warning(f"[{trace_id}] LLM did not return suggestions in the expected format.")
                  return ["No specific suggestions generated."]
         except Exception as e:
              logger.error(f"[{trace_id}] Failed to generate suggestions using LLM: {e}")
              return default_suggestion

    # --- 데이터 준비 헬퍼 ---
    def _prepare_mapping_rows(self, chosen_idea: Optional[Dict], scenarios: List[Dict]) -> List[Dict]:
        """매핑 테이블 데이터 준비 (Node 13과 유사)"""
        rows = []
        default_row = {'num': 1, 'title': 'N/A', 'c1': '-', 'c2': '-', 'c3': '-', 'c4': '-'}
        if not chosen_idea or not scenarios or len(scenarios) != 4:
             logger.warning("Cannot prepare mapping rows: chosen idea or valid 4-panel scenario missing.")
             return [default_row]

        title = self._truncate_text(chosen_idea.get('idea_title', 'N/A'), 30)
        try:
            # 패널 텍스트 추출 (설명 또는 대화)
            c1 = self._truncate_text(scenarios[0].get('panel_description') or scenarios[0].get('dialogue', '-'), 25)
            c2 = self._truncate_text(scenarios[1].get('panel_description') or scenarios[1].get('dialogue', '-'), 25)
            c3 = self._truncate_text(scenarios[2].get('panel_description') or scenarios[2].get('dialogue', '-'), 25)
            c4 = self._truncate_text(scenarios[3].get('panel_description') or scenarios[3].get('dialogue', '-'), 25)
            rows.append({'num': 1, 'title': title, 'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4})
        except IndexError:
             logger.error("Error accessing scenario panels while preparing mapping rows.")
             rows.append(default_row) # 오류 시 기본값
        return rows

    def _calculate_link_usage(self, fact_urls: List[Dict], opinion_urls: List[Dict], used_links: List[Dict]) -> Dict[str, int]:
        """시나리오 컨텍스트 링크 사용량 계산 (Node 13과 동일)"""
        usage = {'used_news': 0, 'total_news': len(fact_urls), 'used_op': 0, 'total_op': len(opinion_urls)}
        if not used_links: return usage
        context_urls = set(link.get('url') for link in used_links if "Scenario Context" in link.get('purpose', '') and link.get('url'))
        if not context_urls: return usage
        original_fact_urls = set(f.get('url') for f in fact_urls if f.get('url'))
        original_opinion_urls = set(o.get('url') for o in opinion_urls if o.get('url'))
        usage['used_news'] = len(context_urls.intersection(original_fact_urls))
        usage['used_op'] = len(context_urls.intersection(original_opinion_urls))
        return usage

    def _calculate_prompt_hash(self, prompt: Optional[str]) -> str:
        """시나리오 생성 프롬프트 해시 계산 (Node 13과 동일)"""
        if not prompt: return "N/A"
        try: return hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
        except Exception: return "Error"

    def _truncate_text(self, text: Optional[str], max_length: int) -> str:
        """텍스트 축약 (Node 13과 동일)"""
        if not text: return ""
        return text[:max_length - 3] + "..." if len(text) > max_length else text

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """시나리오 보고서 생성 실행"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing ScenarioReportNode...")

        config = state.config or {}
        processing_stats = state.processing_stats or {}
        error_message: Optional[str] = None
        report_content = f"# Scenario Report Generation Failed\n\nJinja2 library not available." # 기본 오류

        # Jinja2 및 템플릿 확인
        if not JINJA2_AVAILABLE:
            error_message = "Jinja2 library not available."
        else:
            template_dir = config.get("template_dir")
            template_name = config.get("progress_report_template_b_filename") # 설정에서 템플릿 B 파일명 로드
            if not template_dir or not template_name:
                error_message = "Template directory or filename for Scenario Report (Template B) not found in config."
                report_content = f"# Report Generation Failed\n\n{error_message}"
            elif not self._setup_jinja_env(template_dir):
                error_message = f"Failed to initialize Jinja2 environment from directory: {template_dir}"
                report_content = f"# Report Generation Failed\n\n{error_message}"

        # 템플릿 준비 완료 시 보고서 생성 시도
        if not error_message and self.template_env:
            logger.info(f"{log_prefix} Generating scenario report using template: {template_name}")
            try:
                template = self.template_env.get_template(template_name)

                # --- 템플릿 데이터 준비 ---
                logger.debug(f"{log_prefix} Preparing data for scenario report template...")
                scenarios = state.scenarios or []
                chosen_idea = state.chosen_idea

                # 시나리오 텍스트 조합 (평가/제안용)
                scenario_full_text = ""
                if scenarios and len(scenarios) == 4:
                    scenario_full_text = "\n\n".join([
                        f"Panel {p.get('scene', i+1)}:\nDesc: {p.get('panel_description', '')}\nDialogue: {p.get('dialogue', '')}"
                        for i, p in enumerate(scenarios)
                    ])
                    scenario_full_text = self._truncate_text(scenario_full_text, 2000) # 평가용 길이 제한

                # LLM 평가 및 제안 (선택적 실행)
                quality_scores = {"consistency": 0, "flow": 0, "dialogue": 0}
                suggestions = ["Evaluation disabled or failed."]
                if config.get("enable_scenario_evaluation", False) and self.llm_client and scenario_full_text:
                     logger.info(f"{log_prefix} Performing LLM-based scenario evaluation/suggestion...")
                     # 동시 실행 또는 순차 실행 선택 가능
                     eval_task = self._evaluate_scenario_quality(scenario_full_text, config, state.trace_id)
                     sugg_task = self._generate_suggestions(scenario_full_text, config, state.trace_id)
                     eval_results = await asyncio.gather(eval_task, sugg_task, return_exceptions=True)

                     if isinstance(eval_results[0], dict): quality_scores = eval_results[0]
                     else: logger.error(f"{log_prefix} Evaluation task failed: {eval_results[0]}")

                     if isinstance(eval_results[1], list): suggestions = eval_results[1]
                     else: logger.error(f"{log_prefix} Suggestion task failed: {eval_results[1]}")
                else:
                     logger.info(f"{log_prefix} Skipping LLM-based scenario evaluation/suggestion.")

                # 타임스탬프 포맷팅
                timestamp_str = state.timestamp or datetime.now(timezone.utc).isoformat()
                formatted_timestamp = timestamp_str
                try: formatted_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S %Z")
                except ValueError: pass

                # 컨텍스트 준비
                context = {
                    "timestamp": formatted_timestamp,
                    "trace_id": state.trace_id,
                    "mapping_rows": self._prepare_mapping_rows(chosen_idea, scenarios),
                    "chosen_title": self._truncate_text(chosen_idea.get('idea_title', 'N/A') if chosen_idea else 'N/A', 40),
                    "link_usage": self._calculate_link_usage(state.fact_urls or [], state.opinion_urls or [], state.used_links or []),
                    "quality_scores": quality_scores,
                    "suggestions": suggestions,
                    "prompt_hash": self._calculate_prompt_hash(state.scenario_prompt)
                }
                logger.debug(f"{log_prefix} Scenario report template data prepared.")

                # 템플릿 렌더링
                report_content = template.render(**context)
                logger.info(f"{log_prefix} Scenario report generated successfully.")

            except jinja2.TemplateNotFound:
                error_message = f"Template file '{template_name}' not found in directory '{template_dir}'."
                logger.error(f"{log_prefix} {error_message}")
                report_content = f"# Report Generation Failed\n\n{error_message}"
            except Exception as e:
                error_message = f"Failed to prepare data or render scenario template: {str(e)}"
                logger.exception(f"{log_prefix} {error_message}")
                report_content = f"# Scenario Report Generation Error\n\nAn unexpected error occurred: {str(e)}"

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['scenario_report_node_time'] = node_processing_time
        logger.info(f"{log_prefix} ScenarioReportNode finished in {node_processing_time:.2f} seconds.")

        # TODO: LangSmith 로깅

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "scenario_report": report_content, # 성공/실패 시 모두 생성된 내용을 담음
            "processing_stats": processing_stats,
            "error_message": error_message # 생성 중 발생한 오류 메시지
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}