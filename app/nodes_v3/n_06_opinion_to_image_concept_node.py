# app/nodes_v3/n_06_opinion_to_image_concept_node.py

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
    Opinion,
    ImageConceptState
)

# --- 로거 설정 ---
logger = get_logger("n_06_OpinionToImageConceptNode")


class N06OpinionToImageConceptNode:
    """
    LangGraph 비동기 노드 – 확정된 의견을 '기승전결' 구조의 4개 이미지 콘셉트로 변환합니다. (n_06)
    """

    def __init__(self, llm_service: LLMService, redis_client: DatabaseClient):
        """노드 초기화. 필요한 서비스 클라이언트들을 주입받습니다."""
        self.llm = llm_service
        self.redis = redis_client
        self.logger = logger

    async def __call__(self, current_state_dict: Dict[str, Any], user_response: Optional[str] = None) -> Dict[str, Any]:
        """LangGraph 노드의 메인 진입점 함수."""
        # 이 함수의 전체적인 로직은 이전 답변과 동일하게 유지됩니다.
        # 사용자 피드백을 처리하는 부분은 기승전결 구조에 맞게 수정된 _formulate_choice_question을 통해 자연스럽게 연계됩니다.
        work_id = current_state_dict.get("work_id", "UNKNOWN_WORK_ID_N06")
        log_extra = {"work_id": work_id}
        self.logger.info(
            f"N06_OpinionToImageConceptNode (기승전결) 시작. 사용자 응답: '{user_response if user_response else '[없음]'}'.",
            extra=log_extra)

        try:
            workflow_state = OverallWorkflowState(**current_state_dict)
            node_state = workflow_state.image_concept
        except ValidationError as e:
            self.logger.error(f"N06 State 유효성 검사 실패: {e}", extra=log_extra)
            return current_state_dict

        if not node_state.concept_candidates and not node_state.is_ready:
            self.logger.info("초기 실행 단계: 기승전결 콘셉트 생성 시작.", extra=log_extra)
            selected_opinion = workflow_state.persona_analysis.selected_opinion
            if selected_opinion and workflow_state.persona_analysis.is_ready:
                concepts = await self._run_initial_concept_generation(selected_opinion, work_id)
                node_state.concept_candidates = concepts
                node_state.question = self._formulate_choice_question(concepts)
            else:
                node_state.error_message = "참조할 최종 확정 의견이 없습니다."

        elif node_state.question and user_response:
            self.logger.info("사용자 콘셉트 피드백 처리 시작.", extra=log_extra)
            await self._process_user_feedback(node_state, user_response,
                                              workflow_state.persona_analysis.selected_opinion, work_id)

        workflow_state.image_concept = node_state
        return await self._finalize_and_save_state(workflow_state, log_extra)

    # --- 핵심 수정: 기승전결 프롬프트 ---
    def _build_image_concept_prompt(self, opinion: Opinion) -> str:
        """기승전결 구조의 이미지 콘셉트 생성을 위한 영문 프롬프트를 구성합니다."""
        prompt = f"""You are a creative webtoon writer and visual storyteller.
Your task is to take a written opinion and create a 4-panel narrative image series that follows the classic 'Gi-Seung-Jeon-Gyeol' (起承轉結) structure.

[INPUT OPINION]
- Persona: "{opinion.persona_name}"
- Opinion Text (in Korean): "{opinion.opinion_text}"

[TASK INSTRUCTIONS]
1.  Analyze the opinion to understand its core argument, conflict, and conclusion.
2.  Deconstruct the argument and map it onto the 4-part narrative structure below.
3.  Generate **exactly 4 visual concepts**, one for each part of the narrative, in the correct order.

[NARRATIVE STRUCTURE: Gi-Seung-Jeon-Gyeol (기승전결)]
- **Panel 1 (기: Introduction):** Introduce the core problem or the initial situation described in the opinion. Set the stage.
- **Panel 2 (승: Development):** Show the problem escalating, developing, or its direct consequences. Build upon the introduction.
- **Panel 3 (전: Turn/Climax):** Introduce a critical twist, a reversal, a deeper insight, or the main conflict's peak. This should be the core message of the opinion.
- **Panel 4 (결: Conclusion):** Present the final resolution, the ultimate message, a call to action, or the future outlook resulting from the 'Turn'.

[JSON OUTPUT FORMAT]
- You MUST provide the output as a JSON array of 4 objects, in the correct narrative order.
- Each object must have the following three keys:
  - `narrative_step` (string): The narrative stage of the panel (e.g., "기(起): 문제 제기"). This MUST be in Korean.
  - `concept_description` (string): A detailed, vivid description of the image content. This MUST be in Korean.
  - `caption` (string): A short, impactful text phrase or dialogue for the image. This MUST be in Korean.

Now, generate the JSON array for the provided opinion, following the 'Gi-Seung-Jeon-Gyeol' structure.
"""
        return prompt

    # --- 핵심 수정: 사용자 질문 포맷 ---
    def _formulate_choice_question(self, concepts: List[ImageConcept], prefix_message: str = "") -> str:
        """기승전결 구조의 콘셉트 후보를 제시하고 사용자에게 피드백을 요청하는 질문을 생성합니다."""
        options_text = "\n\n".join(
            [
                f"### {c.panel_id}. {c.narrative_step}\n- **콘셉트 설명:** {c.concept_description}\n- **대표 문구(대사):** \"{c.caption}\""
                for c in concepts]
        )
        return (
            f"{prefix_message}\n\n"
            "의견을 바탕으로 '기승전결' 구조의 4가지 이미지 콘셉트를 구상했습니다.\n\n"
            f"{options_text}\n\n"
            "이 서사 구조와 콘셉트에 대해 어떻게 생각하시나요? 특정 단계(번호)를 수정하거나, 전반적인 의견을 말씀해주세요. "
            "마음에 드신다면 '이대로 진행', 전체를 새로 구상하고 싶다면 '새로 생성'이라고 알려주세요."
        )

    # --- 핵심 수정: 단일 콘셉트 수정 프롬프트 ---
    async def _revise_all_concepts(self, concept: List[ImageConcept], feedback: str, work_id: str) -> List[
        ImageConcept]:
        """전체 콘셉트 리스트를 사용자의 전반적인 피드백을 반영하여 수정합니다. (프롬프트 최종 수정 버전)"""
        original_concepts_json = json.dumps([c.model_dump(exclude={'panel_id'}) for c in concept], ensure_ascii=False,
                                            indent=2)

        # --- 프롬프트 최종 개선 ---
        prompt = f"""You are an expert copywriter and Art Director. Your task is to fundamentally rewrite parts of 4 visual concepts based on a user's single piece of general feedback.

        [Original 4 Concepts (JSON Array)]
        {original_concepts_json}

        [User's General Feedback to Apply]
        - Feedback (in Korean): "{feedback}"

        [CRITICAL TASK & STEP-BY-STEP PROCESS]
        Your goal is to generate a **NEW version** of the JSON array that strictly follows the user's feedback. You must not return the original content unchanged. Follow these steps precisely:
        1.  **Analyze Feedback:** First, identify the core user request. For example, if the user says "대사가 좀 더 길어졌으면 좋겠어. 대사에 설명을 추가해줘." (I want longer captions. Add explanations to them.), your goal is to make every `caption` longer and more descriptive.
        2.  **Rewrite Each Panel's Caption:** Go through each of the 4 concepts one by one. For each concept, read its `concept_description` and then write a **brand new, longer, more explanatory caption** that reflects both the visual description and the user's request. DO NOT reuse the old caption.
            * *Example Thought Process for a rewrite:* "The user wants longer captions. For Panel 1, the description is a desperate owner surrounded by bills. The original caption was '시스템이 소규모 주체를 쫓아내는 구조'. A new, longer caption could be '수많은 청구서 아래, 내 꿈의 무게가 나를 짓누른다. 이 시스템에서 개인의 노력은 어디까지 유효한 걸까?' (Beneath countless bills, the weight of my dream crushes me. In this system, how far can an individual's effort go?)."
        3.  **Assemble Final JSON:** Construct a new JSON array. Use the original `narrative_step` and `concept_description` for each panel, but replace the `caption` with the new, longer one you just wrote for that panel.
        Your final output MUST be only the rewritten JSON array of 4 objects. All string values must be in Korean.
        """
        # --- 프롬프트 최종 개선 끝 ---
        try:
            response = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"revise-all-concepts-v3-{work_id}",
                temperature=0.7,
                max_tokens=3500
            )
            json_string = response.get("generated_text", "[]").strip()
            if json_string.startswith("```json"):
                json_string = json_string[7:-3].strip()

            revised_data_list = json.loads(json_string)
            if len(revised_data_list) != 4:
                self.logger.warning("LLM이 4개가 아닌 다른 수의 콘셉트를 반환했습니다. 원본을 유지합니다.", extra={"work_id": work_id})
                return concept

            return [ImageConcept(panel_id=i + 1, **data) for i, data in enumerate(revised_data_list)]
        except Exception as e:
            self.logger.error(f"전체 콘셉트 수정 중 오류: {e}", exc_info=True, extra={"work_id": work_id})
            return concept

    # _run_initial_concept_generation, _process_user_feedback, _revise_all_concepts, 및 나머지 함수들은
    # Pydantic 모델 필드명 변경(visual_style -> narrative_step)에 맞춰 내부적으로 수정되었으며,
    # 전체적인 로직은 이전 답변과 동일하게 유지됩니다.
    async def _run_initial_concept_generation(self, opinion: Opinion, work_id: str) -> List[ImageConcept]:
        prompt = self._build_image_concept_prompt(opinion)
        try:
            response = await self.llm.generate_text(messages=[{"role": "user", "content": prompt}],
                                                    request_id=f"image-concept-narrative-{work_id}", max_tokens=3000,
                                                    temperature=0.75)
            json_string = response.get("generated_text", "[]").strip()
            if json_string.startswith("```json"): json_string = json_string[7:-3].strip()
            concepts_data = json.loads(json_string)
            return [ImageConcept(panel_id=i + 1, **data) for i, data in enumerate(concepts_data)]
        except Exception as e:
            self.logger.error(f"초기 콘셉트 생성 중 오류: {e}", extra={"work_id": work_id})
            return []

    # --- 핵심 수정: 의도 분류 프롬프트 개선 ---
    async def _classify_user_response_intent(self, user_answer: str, work_id: str) -> str:
        """사용자 피드백 의도를 'CO'(확정), 'RE'(수정), 'RG'(재생성), 'UN'(불명확)으로 분류합니다."""
        prompt_sys = (
            "Your critical task is to classify a user's Korean feedback into one of four categories: 'CO', 'RE', 'RG', or 'UN'.\n"
            "1. 'CO' (Confirm): User is satisfied and agrees to proceed. (e.g., '네 좋아요', '이대로 진행해주세요', '확정')\n"
            "2. 'RE' (Revise): User wants specific modifications to the current proposals. (e.g., '2번을 좀 더 어둡게 바꿔주세요', '대사를 수정해주세요')\n"
            # 'RG' 예시 보강
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
            resp = await self.llm.generate_text(messages=messages, request_id=f"intent-classify-v3-{work_id}",
                                                max_tokens=200, temperature=0.0, timeout=30)
            raw_classification = resp.get("generated_text", "").strip().upper()
            if raw_classification in valid_labels:
                return raw_classification
            else:
                for label in valid_labels:
                    if label in raw_classification:
                        self.logger.warning(f"의도 분류 시 라벨 외 텍스트 포함: '{raw_classification}'. '{label}'로 처리.",
                                            extra={"work_id": work_id})
                        return label
                self.logger.error(f"의도 분류 결과가 유효 라벨 미포함: '{raw_classification}'.", extra={"work_id": work_id})
                return "UN"
        except Exception as e:
            self.logger.error(f"LLM 의도 분류 중 오류: {e}", exc_info=True, extra={"work_id": work_id})
            return "UN"

    # --- 핵심 수정: 피드백 처리 로직 단순화 ---
    async def _process_user_feedback(self, node_state: ImageConceptState, user_response: str, opinion: Opinion,
                                     work_id: str):
        """사용자 피드백을 분류하고 그에 따라 콘셉트를 수정, 재생성 또는 확정합니다."""
        intent = await self._classify_user_response_intent(user_response, work_id)
        self.logger.info(f"사용자 피드백 의도 분류 결과: {intent}", extra={"work_id": work_id})

        if intent == 'CO':
            node_state.final_concepts = node_state.concept_candidates
            node_state.is_ready = True
            node_state.question = None
            self.logger.info("사용자가 모든 콘셉트를 최종 확정했습니다.", extra={"work_id": work_id})

        elif intent == 'RG':
            self.logger.info("사용자가 새로운 콘셉트 세트 재생성을 요청했습니다.", extra={"work_id": work_id})
            new_concepts = await self._run_initial_concept_generation(opinion, work_id)
            node_state.concept_candidates = new_concepts
            node_state.question = self._formulate_choice_question(new_concepts,
                                                                  "\n\n**[알림] 요청에 따라 새로운 콘셉트들을 다시 생성했습니다.**")

        elif intent == 'RE':
            self.logger.info("사용자 수정 요청 처리 시작 (통합 방식).", extra={"work_id": work_id})
            # --- 단일 함수 호출로 로직 통합 ---
            revised_concepts = await self._run_revision_on_concepts(
                original_concepts=node_state.concept_candidates,
                feedback=user_response,
                work_id=work_id
            )
            node_state.concept_candidates = revised_concepts
            node_state.question = self._formulate_choice_question(revised_concepts,
                                                                  "\n\n**[알림] 요청하신 내용이 반영되었습니다.**")

        else:  # 'UN'
            node_state.question = "답변의 의도가 명확하지 않습니다. 현재 콘셉트를 '확정'할까요, 특정 부분을 '수정'할까요, 아니면 '새로 생성'할까요?"

        # --- 신규 통합 함수: _run_revision_on_concepts ---
    async def _run_revision_on_concepts(self, original_concepts: List[ImageConcept], feedback: str, work_id: str) -> \
        List[ImageConcept]:
            """
            사용자 피드백을 바탕으로 전체 콘셉트 리스트를 수정합니다.
            피드백이 특정 번호를 지목하면 해당 콘셉트만, 그렇지 않으면 전체를 수정합니다.
            """
            original_concepts_json = json.dumps([c.model_dump(exclude={'panel_id'}) for c in original_concepts],
                                                ensure_ascii=False, indent=2)

            prompt = f"""You are an expert Art Director. Your task is to intelligently revise a series of 4 visual concepts based on user feedback.

                [Original 4 Concepts (JSON Array)]
                {original_concepts_json}
            
                [User's Feedback (Korean)]
                "{feedback}"
            
                [CRITICAL TASK & INSTRUCTIONS]
                Your goal is to return a **new, revised JSON array of 4 concepts** that accurately incorporates the user's feedback.
            
                1.  **Analyze the Feedback's Scope:** First, determine if the user's feedback targets a specific panel number (e.g., "2번 그림이...", "1번 대사를...") or if it's a general request for all panels (e.g., "전반적으로...", "대사들을 길게...").
            
                2.  **Apply Revisions Accordingly:**
                    -   **If feedback targets a specific panel number:** Modify ONLY that specific concept object in the array. The other three concepts should remain unchanged.
                    -   **If feedback is general:** Apply the feedback cohesively across ALL 4 concepts in the array.
            
                3.  **Rewrite, Don't Just Echo:** You must generate a new version. Do not return the original content if a change was requested.
            
                Your final output MUST be only the rewritten JSON array of 4 objects. All string values must be in Korean.
            """
            try:
                response = await self.llm.generate_text(
                    messages=[{"role": "user", "content": prompt}],
                    request_id=f"unified-revision-{work_id}",
                    temperature=0.7,
                    max_tokens=3500
                )
                json_string = response.get("generated_text", "[]").strip()
                if json_string.startswith("```json"):
                    json_string = json_string[7:-3].strip()

                revised_data_list = json.loads(json_string)
                if len(revised_data_list) != 4:
                    self.logger.warning("LLM이 4개가 아닌 다른 수의 콘셉트를 반환했습니다. 원본을 유지합니다.", extra={"work_id": work_id})
                    return original_concepts

                return [ImageConcept(panel_id=i + 1, **data) for i, data in enumerate(revised_data_list)]
            except Exception as e:
                self.logger.error(f"통합 콘셉트 수정 중 오류: {e}", exc_info=True, extra={"work_id": work_id})
                return original_concepts

    async def _finalize_and_save_state(self, workflow_state: OverallWorkflowState, log_extra: Dict) -> Dict[str, Any]:
        updated_state_dict = workflow_state.model_dump(mode='json')
        await self._save_workflow_state_to_redis(workflow_state.work_id, updated_state_dict)
        self.logger.info("N06 노드 처리 완료 및 상태 저장.", extra=log_extra)
        return updated_state_dict

    async def _save_workflow_state_to_redis(self, work_id: str, state_dict: Dict[str, Any]):
        key = f"workflow:{work_id}:full_state"
        try:
            json_compatible_state = json.loads(json.dumps(state_dict, default=str))
            await self.redis.set(key, json_compatible_state, expire=60 * 60 * 6)
        except Exception as e:
            self.logger.error(f"Redis 상태 저장 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})