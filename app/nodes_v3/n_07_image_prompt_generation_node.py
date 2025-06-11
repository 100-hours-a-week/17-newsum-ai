# app/nodes/n_07_image_prompt_generation_node.py

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

# --- 애플리케이션 구성 요소 임포트 ---
from app.services.llm_service import LLMService
from app.services.database_client import DatabaseClient
from app.utils.logger import get_logger
from app.workflows.state_v3 import (
    OverallWorkflowState,
    ImageConcept,
    ImagePromptItemPydantic,
    ImagePromptsPydanticState
)

# --- 로거 및 상수 설정 ---
logger = get_logger("n_07_ImagePromptGenerationNode")
LLM_TIMEOUT_SEC: int = 60  # 의도 분류 함수에서 사용할 타임아웃


class N07ImagePromptGenerationNode:
    """
    LangGraph 비동기 노드 – 이미지 콘셉트를 Flux Dev 프롬프트로 변환하고 사용자와 상호작용하여 확정합니다. (n_07)
    """

    def __init__(self, llm_service: LLMService, redis_client: DatabaseClient):
        """노드 초기화. 필요한 서비스 클라이언트들을 주입받습니다."""
        self.llm = llm_service
        self.redis = redis_client
        self.logger = logger

    async def __call__(self, current_state_dict: Dict[str, Any], user_response: Optional[str] = None) -> Dict[str, Any]:
        """LangGraph 노드의 메인 진입점 함수."""
        work_id = current_state_dict.get("work_id", "UNKNOWN_WORK_ID_N07")
        log_extra = {"work_id": work_id}
        self.logger.info(f"N07_ImagePromptGenerationNode 시작. 사용자 응답: '{user_response if user_response else '[없음]'}'.",
                         extra=log_extra)

        try:
            workflow_state = OverallWorkflowState(**current_state_dict)
            node_state = workflow_state.image_prompts
        except ValidationError as e:
            self.logger.error(f"N07 State 유효성 검사 실패: {e}", extra=log_extra)
            return current_state_dict

        # 1. 썸네일/4컷 프롬프트 생성 및 저장 (분리)
        if not node_state.prompt_candidates and not node_state.thumbnail_prompt_candidate and not node_state.is_ready:
            # 썸네일 프롬프트 생성
            final_thumbnail = workflow_state.image_concept.final_thumbnail
            if final_thumbnail:
                thumbnail_prompt = await self._convert_concept_to_prompt(final_thumbnail, work_id)
                node_state.thumbnail_prompt_candidate = thumbnail_prompt
            # 4컷 프롬프트 생성
            final_concepts = workflow_state.image_concept.final_concepts
            if final_concepts:
                prompt_tasks = [self._convert_concept_to_prompt(concept, work_id) for concept in final_concepts]
                prompt_candidates = await asyncio.gather(*prompt_tasks)
                node_state.prompt_candidates = [p for p in prompt_candidates if p]

        # 2. 피드백/수정 루프 (통합 피드백 처리)
        if node_state.question and user_response:
            await self._process_prompts_and_thumbnail_feedback(node_state, user_response, work_id)

        # 3. 질문 메시지 생성(썸네일+4컷 모두 포함)
        if not node_state.question:
            panel_text = self._formulate_choice_question(node_state.prompt_candidates)
            thumbnail = node_state.thumbnail_prompt_candidate
            if thumbnail:
                thumbnail_text = (
                    f"\n\n---\n\n"
                    f"### 썸네일 프롬프트\n"
                    f"- prompt: {thumbnail.prompt}\n"
                    f"- negative_prompt: {thumbnail.negative_prompt}"
                )
            else:
                thumbnail_text = ""
            node_state.question = f"아래는 4컷 프롬프트와 썸네일 프롬프트입니다.\n{panel_text}{thumbnail_text}\n\n각 항목에 대해 수정/확정/재생성 의견을 말씀해 주세요."

        workflow_state.image_prompts = node_state
        return await self._finalize_and_save_state(workflow_state, log_extra)

    async def _convert_concept_to_prompt(self, concept: ImageConcept, work_id: str) -> Optional[ImagePromptItemPydantic]:
        prompt_for_llm = self._build_prompt_conversion_prompt(concept)
        try:
            response = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt_for_llm}],
                request_id=f"prompt-generation-{concept.panel_id}-{work_id}",
                temperature=0.4,
                max_tokens=1000)
            json_string = response.get("generated_text", "{}").strip()
            if json_string.startswith("```json"): json_string = json_string[7:-3].strip()
            prompt_data = json.loads(json_string)
            return ImagePromptItemPydantic(
                panel_id=concept.panel_id,
                prompt=prompt_data.get("prompt", ""),
                negative_prompt=prompt_data.get("negative_prompt", "text, watermark, ugly, deformed, blurry")
            )
        except Exception as e:
            self.logger.error(f"{concept.panel_id}번 패널 프롬프트 변환 중 오류: {e}", extra={"work_id": work_id})
            return None

    def _build_prompt_conversion_prompt(self, concept: ImageConcept) -> str:
        return f"""
You are an expert Flux Dev–style image-prompt engineer.

Create a single fluent English sentence to visualize the scene for Flux Dev based on the details below.

**Requirements:**
1. Composition: Specify subject placement (foreground, midground, background).
2. Details: Describe color, lighting, texture, props, and mood.
3. Fluent Prose: Use one or two short English sentences with descriptive clauses (e.g., "with ...").
4. Few-Shot Examples: Provide prompt examples only; negative_prompt examples are not required here.
5. Negative Prompt: After the prompt, generate a **short list of 5 words or phrases** to avoid common flaws (e.g., "text, watermark, blurry, deformed, extra limbs").
6. Output only a JSON object with keys "prompt" and "negative_prompt".

**Examples:**
- "A single tree stands in the center, its left half lush green under a bright sunlit sky and its right half frosted bare under a stormy, thunderous backdrop."
- "In the foreground, a vintage car with a 'CLASSIC' license plate sits on cobblestone, behind it a bustling market of colorful awnings, and in the distance the silhouette of an ancient castle shrouded in mist."

[VISUAL CONCEPT TO CONVERT]
- Narrative Step (Korean): "{concept.narrative_step}"
- Concept Description (Korean): "{concept.concept_description}"
- Caption (Korean): "{concept.caption}"

Now generate only the JSON:
{{
  "prompt": "<Flux Dev–style English sentence>",
  "negative_prompt": "<Short English phrase describing what to avoid>"
}}
"""

    async def _process_prompts_and_thumbnail_feedback(self, node_state: ImagePromptsPydanticState, user_response: str, work_id: str):
        """
        사용자 피드백이 썸네일/패널/전체 중 어디에 대한 것인지 LLM이 판단하여 해당 부분만 수정/재생성/확정하는 통합 피드백 처리 함수
        """
        # 통합 intent 분류 프롬프트
        prompt = f"""
You are an expert image prompt engineer. The user has provided feedback after seeing both a 4-panel set of prompts and a thumbnail prompt. Your job is to:
1. Classify the feedback as targeting (a) the thumbnail, (b) one of the 4 panels, or (c) all/overall.
2. For each target, determine if the user wants to confirm (CO), revise (RE), regenerate (RG), or if the intent is unclear (UN).
3. Apply the user's feedback to the correct part(s) and return the updated prompts.

[Current Prompts]
- Thumbnail:
  - panel_id: 0
  - prompt: {node_state.thumbnail_prompt_candidate.prompt if node_state.thumbnail_prompt_candidate else ''}
  - negative_prompt: {node_state.thumbnail_prompt_candidate.negative_prompt if node_state.thumbnail_prompt_candidate else ''}
- Panels:
"""
        for p in node_state.prompt_candidates:
            prompt += f"  - panel_id: {p.panel_id}\n    prompt: {p.prompt}\n    negative_prompt: {p.negative_prompt}\n"
        prompt += (
            f"""

[User's Feedback (Korean)]
{user_response}

[INSTRUCTIONS]
- You MUST return a JSON object with these keys:
  - 'thumbnail': the updated thumbnail prompt (or null if unchanged). This MUST be a JSON object, not a string.
  - 'panels': a list of 4 updated panel prompts (or null if unchanged). This MUST be a list of JSON objects, not a list of strings.
  - 'finalize': true if the user confirmed all, false otherwise
  - 'clarification_needed': true if the intent was unclear, false otherwise
  - 'clarification_message': (if clarification_needed) a Korean message to ask the user
- DO NOT return any value as a string. All values must be JSON objects or lists as specified.
- Example of correct output:
{{
  "thumbnail": {{"panel_id": 0, "prompt": "...", "negative_prompt": "..."}},
  "panels": [
    {{"panel_id": 1, "prompt": "...", "negative_prompt": "..."}},
    {{"panel_id": 2, "prompt": "...", "negative_prompt": "..."}},
    {{"panel_id": 3, "prompt": "...", "negative_prompt": "..."}},
    {{"panel_id": 4, "prompt": "...", "negative_prompt": "..."}}
  ],
  "finalize": true,
  "clarification_needed": false,
  "clarification_message": ""
}}
- Do NOT return any value as a string (e.g., "thumbnail": "some string" is NOT allowed).
"""
        )
        try:
            response = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"prompts-thumbnail-feedback-{work_id}",
                max_tokens=3000,
                temperature=0.7
            )
            json_string = response.get("generated_text", "{}").strip()
            if json_string.startswith("```json"): json_string = json_string[7:-3].strip()
            data = json.loads(json_string)
            # 썸네일 처리
            if data.get('thumbnail'):
                node_state.thumbnail_prompt_candidate = ImagePromptItemPydantic(**data['thumbnail'])
            # 패널 처리
            if data.get('panels'):
                node_state.prompt_candidates = [ImagePromptItemPydantic(**p) for p in data['panels']]
            # 확정 처리
            if data.get('finalize'):
                node_state.thumbnail_prompt = node_state.thumbnail_prompt_candidate
                node_state.panels = node_state.prompt_candidates
                node_state.is_ready = True
                node_state.question = None
            # 추가 질문
            elif data.get('clarification_needed'):
                node_state.question = data.get('clarification_message', '답변의 의도가 명확하지 않습니다. 어떤 부분을 수정/확정/재생성할지 구체적으로 말씀해 주세요.')
        except Exception as e:
            self.logger.error(f"통합 프롬프트 피드백 처리 중 오류: {e}", extra={"work_id": work_id})
            node_state.question = "피드백 처리 중 오류가 발생했습니다. 다시 시도해 주세요."

    def _formulate_choice_question(self, prompts: List[ImagePromptItemPydantic], prefix_message: str = "") -> str:
        options_text = "\n\n".join(
            [f"### {p.panel_id}번 프롬프트\n**Prompt:** `{p.prompt}`\n**Negative Prompt:** `{p.negative_prompt}`" for p in prompts]
        )
        return (
            f"{prefix_message}\n\n"
            "이미지 생성을 위해 다음과 같이 4개의 프롬프트를 만들었습니다.\n\n"
            f"{options_text}\n"
        )

    async def _finalize_and_save_state(self, workflow_state: OverallWorkflowState, log_extra: Dict) -> Dict[str, Any]:
        updated_state_dict = workflow_state.model_dump(mode='json')
        await self._save_workflow_state_to_redis(workflow_state.work_id, updated_state_dict)
        self.logger.info("N07 노드 처리 완료 및 상태 저장.", extra=log_extra)
        return updated_state_dict

    async def _save_workflow_state_to_redis(self, work_id: str, state_dict: Dict[str, Any]):
        key = f"workflow:{work_id}:full_state"
        try:
            json_compatible_state = json.loads(json.dumps(state_dict, default=str))
            await self.redis.set(key, json_compatible_state, expire=60 * 60 * 6)
        except Exception as e:
            self.logger.error(f"Redis 상태 저장 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})