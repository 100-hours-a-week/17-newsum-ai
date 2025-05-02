# app/nodes/15_scenario_writer_node.py (Improved Version)

import asyncio
import re
import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 기본값 참조용
from app.services.llm_server_client_v2 import LLMService # LLM 서비스 클라이언트
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# 로거 설정
logger = get_logger(__name__)

class ScenarioWriterNode:
    """
    선택된 아이디어와 요약 정보를 바탕으로 4컷 만화 시나리오를 생성합니다 (JSON 형식).
    - LLMService를 사용하며, 결과는 장면 설명, 대화, 이미지 생성용 시드 태그를 포함합니다.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    - 생성에 사용된 프롬프트를 상태에 저장합니다.
    - 컨텍스트에 사용된 링크 추적은 현재 Placeholder입니다. (개선 필요)
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["chosen_idea", "final_summary", "opinion_summaries", "articles", "opinions_raw", "trace_id", "config", "processing_stats", "used_links"]
    outputs: List[str] = ["scenarios", "scenario_prompt", "used_links", "processing_stats", "error_message"]

    # LLMService 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        llm_client: LLMService,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        if not llm_client: raise ValueError("LLMService is required for ScenarioWriterNode")
        self.llm_client = llm_client
        # self.langsmith_service = langsmith_service
        logger.info("ScenarioWriterNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.llm_temp_creative = float(config.get("llm_temperature_creative", settings.DEFAULT_LLM_TEMP_CREATIVE))
        self.max_tokens_scenario = int(config.get("llm_max_tokens_scenario", settings.DEFAULT_MAX_TOKENS_SCENARIO))
        # 컨텍스트 길이 제한 설정
        self.max_context_len = int(config.get("max_context_len_scenario", settings.DEFAULT_MAX_CONTEXT_LEN_SCENARIO))

        logger.debug(f"Runtime config loaded. LLM Temp: {self.llm_temp_creative}, Max Tokens: {self.max_tokens_scenario}, Max Context Len: {self.max_context_len}")


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

    # --- 컨텍스트 준비 ---
    def _prepare_context_for_prompt(self, final_summary: Optional[str], opinion_summary: Optional[Dict[str, Any]]) -> str:
        """시나리오 생성 프롬프트를 위한 컨텍스트 포맷팅 (길이 제한 적용)"""
        context_parts = []
        if final_summary: context_parts.append("[Overall Synthesis]\n" + final_summary)
        # 의견 요약 텍스트 추가 (선택적)
        # if opinion_summary and isinstance(opinion_summary, dict) and opinion_summary.get('summary_text'):
        #     context_parts.append("\n[Opinion Summary Detail]\n" + opinion_summary['summary_text'])

        full_context = "\n\n".join(context_parts).strip()
        if not full_context: return "No summary context available."
        # 설정된 최대 길이로 자르기
        return full_context[:self.max_context_len] + ("..." if len(full_context) > self.max_context_len else "")

    # --- 프롬프트 생성 ---
    def _create_scenario_prompt_en(self, idea_title: str, idea_concept: str, context: str) -> str:
        """4컷 시나리오 생성 프롬프트 (JSON 출력 요구)"""
        # 프롬프트 내용은 이전과 동일
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

    # --- 응답 파싱 ---
    def _parse_llm_response(self, response_json: str, trace_id: Optional[str]) -> List[Dict[str, Any]]:
        """LLM의 시나리오 JSON 응답 파싱 및 검증"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        try:
            logger.debug(f"{log_prefix} Raw LLM response for scenario: {response_json[:500]}...")
            match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_json, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            else:
                json_str = response_json.strip()
                if not json_str.startswith('['): json_str = '[' + json_str
                if not json_str.endswith(']'): json_str = json_str + ']'

            logger.debug(f"{log_prefix} Cleaned JSON for parsing: {json_str[:500]}...")
            scenarios = json.loads(json_str)

            if not isinstance(scenarios, list): raise ValueError("LLM response is not a JSON list.")

            validated_scenarios = []
            required_keys = {"scene", "panel_description", "dialogue", "seed_tags"}
            expected_scene = 1
            for panel in scenarios:
                if len(validated_scenarios) >= 4: break # 최대 4개까지만 검증
                if isinstance(panel, dict) and required_keys.issubset(panel.keys()):
                    scene_num = panel.get('scene')
                    desc = panel.get('panel_description')
                    dialogue = panel.get('dialogue', "") # 기본값 ""
                    tags = panel.get('seed_tags')

                    if isinstance(scene_num, int) and \
                       isinstance(desc, str) and desc.strip() and \
                       isinstance(dialogue, str) and \
                       isinstance(tags, list) and \
                       all(isinstance(tag, str) and tag.strip() for tag in tags):
                        # Scene 번호 자동 조정 및 태그 정리
                        panel['scene'] = expected_scene
                        panel['seed_tags'] = [tag.strip() for tag in tags if tag.strip()]
                        validated_scenarios.append(panel)
                        expected_scene += 1
                    else: logger.warning(f"{log_prefix} Panel #{expected_scene} invalid data: {panel}")
                else: logger.warning(f"{log_prefix} Panel #{expected_scene} invalid format: {panel}")

            if len(validated_scenarios) != 4:
                 logger.error(f"{log_prefix} Failed to validate exactly 4 panels (found {len(validated_scenarios)}).")
                 return [] # 4개가 아니면 실패 처리
            return validated_scenarios

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"{log_prefix} Failed parsing/validating scenario response: {e}. Response: '{response_json[:200]}...'")
            return []
        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error parsing scenario response: {e}")
            return []

    # --- 사용된 링크 추적 업데이트 (Placeholder) ---
    # TODO/Warning: 이 로직은 실제 컨텍스트 생성에 사용된 링크를 정확히 추적하지 못합니다.
    #              요약 노드(08, 09, 10)에서 사용된 원본 URL 정보를 상태에 포함시켜 전달하고,
    #              여기서 해당 정보를 바탕으로 업데이트하는 방식이 필요합니다.
    def _update_context_links(self, state: ComicState, trace_id: Optional[str]) -> List[Dict[str, Any]]:
        """컨텍스트 생성에 사용된 링크 추적 (현재는 Placeholder 로직)"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.warning(f"{log_prefix} Updating used_links for scenario context using PLACEHOLDER logic. Needs refinement!")
        updated_used_links = list(state.used_links or [])
        urls_assumed_used = set() # 사용된 것으로 '가정'하는 URL 집합

        # 뉴스 요약 원본 가정 (스크랩 성공한 모든 기사)
        for article in state.articles or []:
             if article.get('text'): # 텍스트 있으면 스크랩 성공 간주
                  urls_assumed_used.add(article.get('url'))
                  urls_assumed_used.add(article.get('original_url_from_source'))

        # 의견 요약 원본 가정 (스크랩 성공한 모든 의견)
        for opinion in state.opinions_raw or []: # raw 데이터 사용
             if opinion.get('text'): # 텍스트 있으면 스크랩 성공 간주
                  urls_assumed_used.add(opinion.get('url'))

        urls_assumed_used.discard(None)
        links_updated_count = 0
        for link_info in updated_used_links:
            if link_info.get('url') in urls_assumed_used:
                purpose = link_info.get('purpose', '')
                if "Scenario Context" not in purpose:
                    link_info['purpose'] = f"{purpose} | Used for Scenario Context" if purpose else "Used for Scenario Context"
                    links_updated_count += 1
                link_info['status'] = 'context_used' # 상태 업데이트

        logger.info(f"{log_prefix} Marked {links_updated_count} links as (assumed) used for scenario context.")
        return updated_used_links

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """시나리오 생성 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing ScenarioWriterNode...")

        chosen_idea = state.chosen_idea
        final_summary = state.final_summary or ""
        # opinion_summaries는 컨텍스트 생성 시 선택적으로 사용될 수 있음
        opinion_summaries = state.opinion_summaries
        config = state.config or {}
        processing_stats = state.processing_stats or {}
        current_used_links = state.used_links or [] # used_links 업데이트 위해 필요

        # --- 입력 유효성 검사 ---
        if not chosen_idea or not isinstance(chosen_idea, dict) or \
           not chosen_idea.get('idea_title') or not chosen_idea.get('concept'):
            logger.error(f"{log_prefix} Valid 'chosen_idea' with title and concept is missing.")
            processing_stats['scenario_writer_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "scenarios": [], "scenario_prompt": None, "used_links": current_used_links,
                "processing_stats": processing_stats, "error_message": "Valid chosen_idea is required."
            }

        if not final_summary:
            logger.warning(f"{log_prefix} Final summary is empty, scenario context might be weak.")
            # 요약 없어도 아이디어만으로 생성 시도 가능하도록 진행

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        # --------------------------

        logger.info(f"{log_prefix} Starting scenario generation for idea: '{chosen_idea['idea_title']}'")
        error_message: Optional[str] = None
        scenarios: List[Dict[str, Any]] = []
        scenario_prompt: Optional[str] = None # 생성된 프롬프트 저장

        try:
            # --- LLM 입력 준비 및 프롬프트 생성 ---
            context = self._prepare_context_for_prompt(final_summary, opinion_summaries)
            scenario_prompt = self._create_scenario_prompt_en(
                chosen_idea['idea_title'], chosen_idea['concept'], context
            )

            # --- LLM 호출 및 파싱 ---
            llm_kwargs = {"response_format": {"type": "json_object"}}
            response_str = await self._call_llm_with_retry(
                prompt=scenario_prompt,
                temperature=self.llm_temp_creative,
                max_tokens=self.max_tokens_scenario,
                trace_id=trace_id,
                **llm_kwargs
            )
            scenarios = self._parse_llm_response(response_str, trace_id)

            if not scenarios:
                 error_message = "Failed to generate or parse a valid 4-panel scenario from LLM."
                 logger.error(f"{log_prefix} {error_message}")
                 scenarios = [] # 빈 리스트 반환 보장
            else:
                 logger.info(f"{log_prefix} Successfully generated 4-panel scenario.")

        except RetryError as e:
            error_message = f"Scenario generation LLM call failed after multiple retries: {e}"
            logger.error(f"{log_prefix} {error_message}")
            scenarios = []
        except Exception as e:
            error_message = f"Scenario generation failed: {str(e)}"
            logger.exception(f"{log_prefix} {error_message}")
            scenarios = []

        # --- 사용된 링크 추적 업데이트 (Placeholder) ---
        # TODO: 이 로직은 실제 사용된 링크 추적을 위해 개선되어야 합니다.
        updated_used_links = self._update_context_links(state, trace_id)

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['scenario_writer_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} ScenarioWriterNode finished in {processing_stats['scenario_writer_node_time']:.2f} seconds.")

        # ComicState 업데이트
        update_data: Dict[str, Any] = {
            "scenarios": scenarios,
            "scenario_prompt": scenario_prompt, # 생성된 프롬프트 저장
            "used_links": updated_used_links, # 업데이트된 링크 목록
            "processing_stats": processing_stats,
            "error_message": error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}