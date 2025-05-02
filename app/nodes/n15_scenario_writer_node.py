# app/nodes/15_scenario_writer_node.py

import asyncio
import re
import json
import hashlib
# --- datetime, timezone 임포트 추가 ---
from datetime import datetime, timezone
# ------------------------------------
from typing import List, Dict, Any, Optional
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 프로젝트 구성 요소 임포트 ---
from app.config.settings import settings            # 설정 객체 (재시도 횟수 등 참조)
from app.services.llm_server_client_v2 import LLMService # 실제 LLM 서비스 클라이언트
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# --- 로거 설정 ---
logger = get_logger("ScenarioWriterNode")

class ScenarioWriterNode:
    """
    (Refactored) 선택된 아이디어와 요약 정보를 바탕으로 4컷 만화 시나리오를 생성합니다.
    - LLMService를 사용하여 시나리오 생성.
    - 결과는 장면 설명, 대화, 이미지 생성용 시드 태그를 포함하는 JSON 구조.
    - 설정은 상태의 config 딕셔너리에서 로드.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["chosen_idea", "final_summary", "opinion_summaries", "articles", "opinions_raw", "trace_id", "config", "processing_stats", "used_links"]
    outputs: List[str] = ["scenarios", "scenario_prompt", "used_links", "processing_stats", "error_message"]

    # LLMService 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        llm_client: LLMService,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        self.llm_client = llm_client
        # self.langsmith_service = langsmith_service
        logger.info("ScenarioWriterNode initialized with LLMService.")

    # --- LLM 호출 래퍼 (재시도 적용) ---
    # 이전 노드들과 동일한 래퍼 사용
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
            **kwargs # response_format={"type": "json_object"} 등 전달
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
    def _prepare_context_for_prompt(self, final_summary: Optional[str], opinion_summary: Optional[Dict[str, Any]], max_len: int) -> str:
        """시나리오 생성 프롬프트를 위한 컨텍스트 포맷팅 (길이 제한 적용)"""
        context = ""
        if final_summary: context += "[Overall Synthesis]\n" + final_summary + "\n\n"
        # 의견 요약 텍스트 추가 (필요시)
        # if opinion_summary and opinion_summary.get('summary_text'):
        #     context += "[Opinion Summary Detail]\n" + opinion_summary['summary_text'] + "\n"

        if not context: return "No summary context available."
        # 설정된 최대 길이로 자르기
        return context[:max_len] + ("..." if len(context) > max_len else "")

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
            # 마크다운 코드 블록 제거
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
            for panel in scenarios: # 최대 4개 패널까지만 처리 시도
                if len(validated_scenarios) >= 4: break
                if isinstance(panel, dict) and required_keys.issubset(panel.keys()):
                    scene_num = panel.get('scene')
                    desc = panel.get('panel_description')
                    dialogue = panel.get('dialogue', "") # 기본값 ""
                    tags = panel.get('seed_tags')

                    # 타입 및 값 유효성 검사
                    if isinstance(scene_num, int) and \
                       isinstance(desc, str) and desc.strip() and \
                       isinstance(dialogue, str) and \
                       isinstance(tags, list) and \
                       all(isinstance(tag, str) and tag.strip() for tag in tags):

                        # scene 번호 자동 조정 (1부터 순차적으로)
                        panel['scene'] = expected_scene
                        # 태그 리스트 정리 (빈 태그 제거)
                        panel['seed_tags'] = [tag.strip() for tag in tags if tag.strip()]
                        validated_scenarios.append(panel)
                        expected_scene += 1
                    else:
                        logger.warning(f"{log_prefix} Scenario panel #{expected_scene} has invalid data types or empty values: {panel}")
                else:
                    logger.warning(f"{log_prefix} Scenario panel #{expected_scene} missing keys or not a dict: {panel}")

            # 최종적으로 4개 패널이 확보되었는지 확인
            if len(validated_scenarios) != 4:
                 logger.error(f"{log_prefix} Failed to validate exactly 4 panels (validated {len(validated_scenarios)}).")
                 # 4개가 아니면 빈 리스트 반환 또는 다른 오류 처리 방식 선택 가능
                 return []

            return validated_scenarios

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"{log_prefix} Failed parsing/validating LLM scenario response: {e}. Response fragment: '{response_json[:200]}...'")
            return []
        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error parsing LLM scenario response: {e}")
            return []

    # --- 사용된 링크 추적 업데이트 (Placeholder) ---
    def _update_context_links(self, state: ComicState, trace_id: Optional[str]) -> List[Dict[str, Any]]:
        """
        (Placeholder) 컨텍스트 생성에 사용된 링크들의 용도를 used_links에서 업데이트합니다.
        현재 로직은 요약 생성에 사용된 모든 원본 글/의견 URL을 사용했다고 가정합니다.
        더 정확한 추적을 위해서는 요약 노드에서 출처 URL 정보를 유지해야 합니다.
        """
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Updating used_links for scenario context (placeholder logic)...")
        updated_used_links = list(state.used_links or []) # 상태 복사
        urls_to_update = set()

        # 뉴스 요약의 원본 (성공적으로 스크랩된 기사)
        for article in state.articles or []:
             if article.get('text'): # 텍스트가 있는 = 성공적으로 스크랩된 것으로 간주
                  urls_to_update.add(article.get('url'))
                  urls_to_update.add(article.get('original_url_from_source')) # 혹시 다른 URL 필드가 있다면 추가

        # 의견 요약의 원본 (성공적으로 스크랩된 의견)
        for opinion in state.opinions_raw or []: # opinions_clean 대신 raw 사용 (URL 매핑 용이)
             if opinion.get('text'): # 텍스트 존재 여부로 성공 간주
                  urls_to_update.add(opinion.get('url'))

        urls_to_update.discard(None)

        links_updated_count = 0
        for link_info in updated_used_links:
            if link_info.get('url') in urls_to_update:
                purpose = link_info.get('purpose', '')
                if "Scenario Context" not in purpose: # 중복 추가 방지
                    link_info['purpose'] = f"{purpose} | Used for Scenario Context" if purpose else "Used for Scenario Context"
                    links_updated_count += 1
                link_info['status'] = 'context_used' # 상태 업데이트

        logger.info(f"{log_prefix} Marked {links_updated_count} links as used for scenario context.")
        return updated_used_links

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """시나리오 생성 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing ScenarioWriterNode...")

        # 상태 및 설정 로드
        chosen_idea = state.chosen_idea
        final_summary = state.final_summary or ""
        opinion_summaries = state.opinion_summaries or {}
        config = state.config or {}