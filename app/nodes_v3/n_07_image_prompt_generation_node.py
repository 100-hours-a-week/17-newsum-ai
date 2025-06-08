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
    LangGraph 비동기 노드 – 이미지 콘셉트를 SDXL 프롬프트로 변환하고 사용자와 상호작용하여 확정합니다. (n_07)
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

        if not node_state.prompt_candidates and not node_state.is_ready:
            self.logger.info("초기 실행 단계: 4개 이미지 프롬프트 생성 시작.", extra=log_extra)
            final_concepts = workflow_state.image_concept.final_concepts
            if final_concepts and workflow_state.image_concept.is_ready:
                prompts = await self._run_initial_prompt_generation(final_concepts, work_id)
                node_state.prompt_candidates = prompts
                node_state.question = self._formulate_choice_question(prompts)
            else:
                node_state.error_message = "참조할 최종 확정 콘셉트가 없습니다."

        elif node_state.question and user_response:
            self.logger.info("사용자 프롬프트 피드백 처리 시작.", extra=log_extra)
            await self._process_user_feedback(node_state, user_response, workflow_state.image_concept.final_concepts,
                                              work_id)

        workflow_state.image_prompts = node_state
        return await self._finalize_and_save_state(workflow_state, log_extra)

    # --- 핵심 수정: 피드백 처리 로직 ---
    async def _process_user_feedback(self, node_state: ImagePromptsPydanticState, user_response: str,
                                     concepts: List[ImageConcept], work_id: str):
        """사용자 피드백을 분류하고 프롬프트를 수정, 재생성 또는 확정합니다."""
        # 개선된 LLM 기반 의도 분류 함수 호출
        intent = await self._classify_user_response_intent(user_response, work_id)
        self.logger.info(f"사용자 프롬프트 피드백 의도 분류 결과: {intent}", extra={"work_id": work_id})

        if intent == 'CO':
            node_state.panels = node_state.prompt_candidates
            node_state.is_ready = True
            node_state.question = None
            self.logger.info("사용자가 모든 프롬프트를 최종 확정했습니다.", extra={"work_id": work_id})

        elif intent == 'RG':
            self.logger.info("사용자가 새로운 프롬프트 세트 재생성을 요청했습니다.", extra={"work_id": work_id})
            new_prompts = await self._run_initial_prompt_generation(concepts, work_id)
            node_state.prompt_candidates = new_prompts
            node_state.question = self._formulate_choice_question(new_prompts, "\n\n**[알림] 요청에 따라 프롬프트를 다시 생성했습니다.**")

        elif intent == 'RE':
            # LLM 기반 분류기를 사용하므로, 이제 '전체 수정' 요청도 더 잘 처리할 수 있습니다.
            # 코드는 이전과 동일하게 유지하되, 분류기의 성능 향상으로 더 정확하게 동작합니다.
            match = re.search(r"(\d+)\s*번", user_response)
            if match:  # 특정 프롬프트 수정
                try:
                    choice_index = int(match.group(1)) - 1
                    original_prompt_item = node_state.prompt_candidates[choice_index]
                    original_concept = concepts[choice_index]
                    revised_prompt = await self._revise_single_prompt(original_prompt_item, original_concept,
                                                                      user_response, work_id)
                    node_state.prompt_candidates[choice_index] = revised_prompt
                    node_state.question = self._formulate_choice_question(node_state.prompt_candidates,
                                                                          f"\n\n**[알림] {choice_index + 1}번 프롬프트가 수정되었습니다.**")
                except (ValueError, IndexError):
                    node_state.question = "잘못된 번호를 언급하셨습니다. 다시 확인해주세요."
            else:  # 전반적인 프롬프트 수정
                # 전반적인 수정 요청은 아직 지원하지 않는다는 메시지 유지 또는 전체 수정 로직 추가
                self.logger.warning("전반적인 프롬프트 수정 요청은 아직 별도 로직이 없습니다. 번호 지정이 필요합니다.", extra={"work_id": work_id})
                node_state.question = "전체 프롬프트를 한 번에 수정하는 것은 현재 지원되지 않습니다. 수정하고 싶은 프롬프트의 번호를 지정하여 한 번에 하나씩 수정해주세요. (예: 1번 프롬프트를 더 밝게)"

        else:  # 'UN'
            node_state.question = "답변의 의도가 명확하지 않습니다. 현재 프롬프트를 '확정'할까요, 특정 부분을 '수정'할까요, 아니면 '새로 생성'할까요?"

    # --- 교체된 함수: _classify_user_response_intent ---
    async def _classify_user_response_intent(self, user_answer: str, work_id: str) -> str:
        """사용자 피드백 의도를 'CO'(확정), 'RE'(수정), 'RG'(재생성), 'UN'(불명확)으로 분류합니다."""
        prompt_sys = (
            "Your critical task is to classify a user's Korean feedback into one of four categories: 'CO', 'RE', 'RG', or 'UN'.\n"
            "1. 'CO' (Confirm): User is satisfied and agrees to proceed. (e.g., '네 좋아요', '이대로 진행해주세요', '확정')\n"
            "2. 'RE' (Revise): User wants specific modifications to the current proposals. (e.g., '2번을 좀 더 어둡게 바꿔주세요', '대사를 수정해주세요')\n"
            "3. 'RG' (Regenerate): User is dissatisfied and wants a completely new set. (e.g., '새로 생성', '다시생성', '다 별로네요, 새로 만들어주세요', '다른 콘셉트는 없나요?')\n"
            "4. 'UN' (Unclear): The user's intent is ambiguous.\n\n"
            "**RESPONSE INSTRUCTIONS:**\n"
            "- Your response MUST BE ONLY ONE of these exact English strings: 'CO', 'RE', 'RG', or 'UN'.\n"
            "- DO NOT include any other text or explanations."
        )
        prompt_user = (f"[User's Feedback (Korean)]\n{user_answer}\n\nClassify the intent:")
        messages = [{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt_user}]

        valid_labels = ["CO", "RE", "RG", "UN"]
        try:
            resp = await self.llm.generate_text(
                messages=messages,
                request_id=f"intent-classify-v3-{work_id}",
                max_tokens=50,
                temperature=0.0,
                timeout=LLM_TIMEOUT_SEC / 4
            )
            raw_classification = resp.get("generated_text", "").strip().upper()

            if raw_classification in valid_labels:
                return raw_classification
            else:
                for label in valid_labels:
                    if label in raw_classification:
                        self.logger.warning(f"의도 분류 시 라벨 외 텍스트 포함: '{raw_classification}'. '{label}'로 처리.",
                                            extra={"work_id": work_id})
                        return label
                self.logger.error(f"의도 분류 결과가 유효 라벨 미포함: '{raw_classification}'. 'UN'으로 처리.",
                                  extra={"work_id": work_id})
                return "UN"
        except Exception as e:
            self.logger.error(f"LLM 의도 분류 중 오류: {e}", exc_info=True, extra={"work_id": work_id})
            return "UN"

    # --- 나머지 헬퍼 함수들은 이전 답변과 동일하게 유지됩니다 ---
    # ... ( _run_initial_prompt_generation, _convert_concept_to_prompt, _revise_single_prompt, _formulate_choice_question 등) ...
    async def _run_initial_prompt_generation(self, concepts: List[ImageConcept], work_id: str) -> List[
        ImagePromptItemPydantic]:
        prompt_tasks = [self._convert_concept_to_prompt(concept, work_id) for concept in concepts]
        generated_prompts = await asyncio.gather(*prompt_tasks)
        return [prompt for prompt in generated_prompts if prompt]

    async def _convert_concept_to_prompt(self, concept: ImageConcept, work_id: str) -> Optional[
        ImagePromptItemPydantic]:
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
                negative_prompt=prompt_data.get("negative_prompt", "text, watermark, ugly, deformed, blurry"))
        except Exception as e:
            self.logger.error(f"{concept.panel_id}번 패널 프롬프트 변환 중 오류: {e}", extra={"work_id": work_id})
            return None

    def _build_prompt_conversion_prompt(self, concept: ImageConcept) -> str:
        return f"""You are a professional prompt engineer for advanced text-to-image models like Stable Diffusion XL (SDXL).
            Your task is to convert a detailed visual concept into a high-quality, effective English prompt.
            [VISUAL CONCEPT TO CONVERT]
            - Narrative Step (Korean): "{concept.narrative_step}"
            - Concept Description (Korean): "{concept.concept_description}"
            - Caption (Korean): "{concept.caption}"
            [TASK INSTRUCTIONS]
            1.  Read the Korean concept description and caption to fully understand the scene, mood, and message.
            2.  Translate and expand this concept into a detailed, comma-separated English prompt for SDXL.
            3.  The prompt must include keywords for:
                - Subject & Composition
                - Action & Emotion
                - Environment & Background
                - Art Style & Lighting
            4.  Create a standard `negative_prompt` to prevent common image flaws.
            [JSON OUTPUT FORMAT]
            Your response MUST be ONLY a JSON object with two keys: "prompt" and "negative_prompt".
            Example:
            {{
              "prompt": "masterpiece, best quality, cinematic digital painting of a middle-aged Korean cafe owner, head in hands, desperate expression, sitting at a wooden table in a dimly lit cafe at night, piles of bills with red text scattered on the table, dramatic shadows, moody, emotional",
              "negative_prompt": "text, watermark, signature, ugly, deformed, blurry, extra limbs, poorly drawn hands"
            }}
            Now, generate the JSON for the provided visual concept.
        """

    async def _revise_single_prompt(self, prompt_item: ImagePromptItemPydantic, concept: ImageConcept, feedback: str,
                                    work_id: str) -> ImagePromptItemPydantic:
        prompt_for_llm = f"""You are a prompt engineer revising a prompt for an SDXL model.
            [Original Visual Concept]
            - Narrative Step: {concept.narrative_step}
            - Description: {concept.concept_description}
            - Caption: {concept.caption}
            [Original Prompt to Revise]
            - "prompt": "{prompt_item.prompt}"
            - "negative_prompt": "{prompt_item.negative_prompt}"
            [User's Revision Request (Korean)]
            "{feedback}"
            [TASK]
            Revise the [Original Prompt to Revise] based on the user's feedback, while still respecting the [Original Visual Concept].
            Your response MUST be ONLY a JSON object with "prompt" and "negative_prompt" keys.
        """
        try:
            response = await self.llm.generate_text(messages=[{"role": "user", "content": prompt_for_llm}],
                                                    request_id=f"revise-prompt-{work_id}", temperature=0.5)
            json_string = response.get("generated_text", "{}").strip()
            if json_string.startswith("```json"): json_string = json_string[7:-3].strip()
            revised_data = json.loads(json_string)
            prompt_item.prompt = revised_data.get("prompt", prompt_item.prompt)
            prompt_item.negative_prompt = revised_data.get("negative_prompt", prompt_item.negative_prompt)
            return prompt_item
        except Exception as e:
            self.logger.error(f"프롬프트 수정 중 오류: {e}", extra={"work_id": work_id})
            return prompt_item

    def _formulate_choice_question(self, prompts: List[ImagePromptItemPydantic], prefix_message: str = "") -> str:
        options_text = "\n\n".join(
            [f"### {p.panel_id}번 프롬프트\n**Prompt:** `{p.prompt}`\n**Negative Prompt:** `{p.negative_prompt}`" for p in
             prompts]
        )
        return (
            f"{prefix_message}\n\n"
            "이미지 생성을 위해 다음과 같이 4개의 프롬프트를 만들었습니다.\n\n"
            f"{options_text}\n\n"
            "이 프롬프트들에 대해 어떻게 생각하시나요? 특정 번호를 골라 수정하거나, 의견을 말씀해주세요. "
            "마음에 드신다면 '이대로 진행', 전체를 새로 만들고 싶다면 '새로 생성'이라고 알려주세요."
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