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
        work_id = current_state_dict.get("work_id", "UNKNOWN_WORK_ID_N06")
        log_extra = {"work_id": work_id}
        self.logger.info(
            f"N06_OpinionToImageConceptNode (기승전결+썸네일) 시작. 사용자 응답: '{user_response if user_response else '[없음]'}'",
            extra=log_extra)

        try:
            workflow_state = OverallWorkflowState(**current_state_dict)
            node_state = workflow_state.image_concept
        except ValidationError as e:
            self.logger.error(f"N06 State 유효성 검사 실패: {e}", extra=log_extra)
            return current_state_dict

        # 1. 기승전결 생성
        if not node_state.concept_candidates and not node_state.is_ready:
            self.logger.info("초기 실행 단계: 기승전결 콘셉트 생성 시작.", extra=log_extra)
            selected_opinion = workflow_state.persona_analysis.selected_opinion
            if selected_opinion and workflow_state.persona_analysis.is_ready:
                concepts = await self._run_initial_concept_generation(selected_opinion, work_id)
                node_state.concept_candidates = concepts
            else:
                node_state.error_message = "참조할 최종 확정 의견이 없습니다."

        # 2. 썸네일 생성
        if not node_state.thumbnail_candidate and not node_state.is_ready:
            selected_opinion = workflow_state.persona_analysis.selected_opinion
            if selected_opinion and workflow_state.persona_analysis.is_ready:
                thumbnail = await self._run_thumbnail_concept_generation(selected_opinion, work_id)
                node_state.thumbnail_candidate = thumbnail
            else:
                node_state.error_message = "참조할 최종 확정 의견이 없습니다."

        # 3. 사용자 피드백 처리
        if node_state.question and user_response:
            await self._process_concepts_and_thumbnail_feedback(node_state, user_response, workflow_state.persona_analysis.selected_opinion, work_id)

        # 4. 질문 메시지 생성(기승전결+썸네일 모두 포함)
        if not node_state.question:
            # 기승전결+썸네일을 모두 보여주는 질문 메시지
            panel_text = self._formulate_choice_question(
                node_state.concept_candidates,
                thumbnail=node_state.thumbnail_candidate
            )
            node_state.question = panel_text

        workflow_state.image_concept = node_state
        return await self._finalize_and_save_state(workflow_state, log_extra)

    # --- 핵심 수정: 기승전결 프롬프트 ---
    def _build_image_concept_prompt(self, opinion: Opinion) -> str:
        """기승전결 구조의 이미지 콘셉트 생성을 위한 영문 프롬프트를 구성합니다."""
        prompt = f"""You are a creative webtoon writer and visual storyteller.
Your task is to take a written opinion and create a 4-panel narrative image series that follows the classic 'Gi-Seung-Jeon-Gyeol' (起承轉結) structure.

[INPUT OPINION]
- Persona: \"{opinion.persona_name}\"
- Opinion Text (in Korean): \"{opinion.opinion_text}\"

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
  - `narrative_step` (string): The narrative stage of the panel (e.g., \"기(起): 문제 제기\"). This MUST be in Korean.
  - `concept_description` (string): A detailed, vivid description of the image content. This MUST be in Korean. **If the message or caption contains a specific phrase (e.g., '디지털 디톡스'), do NOT draw the phrase as text in the image. Instead, represent its meaning using symbolic objects, icons, or scenes that visually convey the message.**
  - `caption` (string): A short, impactful text phrase or dialogue for the image. This MUST be in Korean.

Now, generate the JSON array for the provided opinion, following the 'Gi-Seung-Jeon-Gyeol' structure.
"""
        return prompt

    # --- 핵심 수정: 사용자 질문 포맷 ---
    def _formulate_choice_question(self, concepts: List[ImageConcept], thumbnail: Optional[ImageConcept] = None, prefix_message: str = "") -> str:
        """기승전결 구조의 콘셉트 후보와 썸네일을 강조해서 보여주고, 사용자에게 피드백을 요청하는 질문을 생성합니다."""
        # 기승전결 패널 출력
        options_text = "\n\n".join(
            [
                f"### {c.panel_id}. {c.narrative_step}\n- **콘셉트 설명:** {c.concept_description}\n- **대표 문구(대사):** \"{c.caption}\""
                for c in concepts]
        )
        # 썸네일은 번호 없이 강조
        if thumbnail:
            thumbnail_text = (
                f"\n\n---\n\n"
                f"### 썸네일\n"
                f"- **설명:** {thumbnail.concept_description}\n"
                f"- **대표 문구:** \"{thumbnail.caption}\""
            )
        else:
            thumbnail_text = ""
        return (
            f"{prefix_message}\n\n"
            "아래는 기승전결 4컷과 썸네일 제안입니다.\n\n"
            "의견을 바탕으로 '기승전결' 구조의 4가지 이미지 콘셉트를 구상했습니다.\n\n"
            f"{options_text}"
            f"{thumbnail_text}\n\n"
            "각 항목에 대해 수정/확정/재생성 의견을 말씀해 주세요."
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
    async def _process_concepts_and_thumbnail_feedback(self, node_state: ImageConceptState, user_response: str, opinion: Opinion, work_id: str):
        """
        사용자 피드백이 썸네일/패널/전체 중 어디에 대한 것인지 LLM이 판단하여 해당 부분만 수정/재생성/확정하는 통합 피드백 처리 함수
        """
        # 통합 intent 분류 프롬프트
        prompt = f"""
You are an expert editorial cartoon planner. The user has provided feedback after seeing both a 4-panel narrative (기승전결) and a thumbnail image concept. Your job is to:
1. Classify the feedback as targeting (a) the thumbnail, (b) one of the 4 panels, or (c) all/overall.
2. For each target, determine if the user wants to confirm (CO), revise (RE), regenerate (RG), or if the intent is unclear (UN).
3. Apply the user's feedback to the correct part(s) and return the updated concepts.

[Current Concepts]
- Thumbnail:
  - panel_id: 0
  - narrative_step: {node_state.thumbnail_candidate.narrative_step if node_state.thumbnail_candidate else ''}
  - concept_description: {node_state.thumbnail_candidate.concept_description if node_state.thumbnail_candidate else ''}
  - caption: {node_state.thumbnail_candidate.caption if node_state.thumbnail_candidate else ''}
- Panels:
"""
        # thumbnail 내용에 이어서 panels 내용을 prompt에 추가하는 코드
        for c in node_state.concept_candidates:
            prompt += f"  - panel_id: {c.panel_id}\n    narrative_step: {c.narrative_step}\n    concept_description: {c.concept_description}\n    caption: {c.caption}\n"
        prompt += f"\n[User's Feedback (Korean)]\n{user_response}\n\n[INSTRUCTIONS]\n- If the feedback is about the thumbnail, only update the thumbnail.\n- If about a specific panel, only update that panel.\n- If about all, update all.\n- If the user confirms, copy the current concept(s) to the final field(s).\n- If unclear, return a message asking for clarification.\n\n[OUTPUT FORMAT]\nReturn a JSON object with these keys:\n- 'thumbnail': the updated thumbnail concept (or null if unchanged)\n- 'panels': a list of 4 updated panel concepts (or null if unchanged)\n- 'finalize': true if the user confirmed all, false otherwise\n- 'clarification_needed': true if the intent was unclear, false otherwise\n- 'clarification_message': (if clarification_needed) a Korean message to ask the user\n"

        try:
            response = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"concepts-thumbnail-feedback-{work_id}",
                max_tokens=3000,
                temperature=0.7
            )
            json_string = response.get("generated_text", "{}").strip()
            if json_string.startswith("```json"): json_string = json_string[7:-3].strip()
            data = json.loads(json_string)
            # 썸네일 처리
            if data.get('thumbnail'):
                node_state.thumbnail_candidate = ImageConcept(**data['thumbnail'])
            # 패널 처리
            if data.get('panels'):
                node_state.concept_candidates = [ImageConcept(**p) for p in data['panels']]
            # 확정 처리
            if data.get('finalize'):
                node_state.final_thumbnail = node_state.thumbnail_candidate
                node_state.final_concepts = node_state.concept_candidates
                node_state.is_ready = True
                node_state.question = None
            # 추가 질문
            elif data.get('clarification_needed'):
                node_state.question = data.get('clarification_message', '답변의 의도가 명확하지 않습니다. 어떤 부분을 수정/확정/재생성할지 구체적으로 말씀해 주세요.')
        except Exception as e:
            self.logger.error(f"통합 피드백 처리 중 오류: {e}", extra={"work_id": work_id})
            node_state.question = "피드백 처리 중 오류가 발생했습니다. 다시 시도해 주세요."

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

    async def _run_thumbnail_concept_generation(self, opinion: Opinion, work_id: str) -> Optional[ImageConcept]:
        """
        썸네일 이미지를 위한 별도 LLM 프롬프트로 ImageConcept 1개를 생성합니다.
        """
        prompt = f"""You are a creative editorial designer.
Your task is to create a single, representative thumbnail image concept for the following opinion.

[INPUT OPINION]
- Persona: \"{opinion.persona_name}\"
- Opinion Text (in Korean): \"{opinion.opinion_text}\"

[TASK INSTRUCTIONS]
1. Analyze the opinion and extract its core message and mood.
2. Imagine a single, visually impactful thumbnail image that best represents the overall opinion.
3. Write a detailed, vivid description of the image content (in Korean). **If the message or caption contains a specific phrase (e.g., '디지털 디톡스'), do NOT draw the phrase as text in the image. Instead, represent its meaning using symbolic objects, icons, or scenes that visually convey the message.**
4. Write a short, powerful caption or title for the thumbnail (in Korean).

[OUTPUT FORMAT]
Return a JSON object with the following keys:
- panel_id: 0
- narrative_step: "썸네일"
- concept_description: (Korean, detailed image description)
- caption: (Korean, short title or phrase)
"""
        try:
            response = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"thumbnail-concept-{work_id}",
                max_tokens=1000,
                temperature=0.7
            )
            json_string = response.get("generated_text", "{}").strip()
            if json_string.startswith("```json"): json_string = json_string[7:-3].strip()
            data = json.loads(json_string)
            return ImageConcept(**data)
        except Exception as e:
            self.logger.error(f"썸네일 콘셉트 생성 중 오류: {e}", extra={"work_id": work_id})
            return None