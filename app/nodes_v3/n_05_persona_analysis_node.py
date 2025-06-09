# app/nodes_v3/n_05_persona_analysis_node.py

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

# --- 애플리케이션 구성 요소 임포트 ---
from app.services.postgresql_service import PostgreSQLService
from app.services.llm_service import LLMService
from app.services.database_client import DatabaseClient
from app.utils.logger import get_logger
from app.workflows.state_v3 import (
    OverallWorkflowState,
    Opinion,
    PersonaAnalysisState
)

# --- 로거 및 상수 설정 ---
logger = get_logger("n_05_PersonaAnalysisNode")
LLM_TIMEOUT_SEC: int = 60  # 의도 분류 함수에서 사용


class N05PersonaAnalysisNode:
    """
    LangGraph 비동기 노드 – 여러 페르소나의 관점에서 의견을 생성하고 사용자와 상호작용하여 최종 의견을 확정합니다. (n_05)
    """

    def __init__(self, postgre_db_client: PostgreSQLService, llm_service: LLMService, redis_client: DatabaseClient):
        """노드 초기화. 필요한 서비스 클라이언트들을 주입받습니다."""
        self.db = postgre_db_client
        self.llm = llm_service
        self.redis = redis_client
        self.logger = logger

    async def __call__(self, current_state_dict: Dict[str, Any], user_response: Optional[str] = None) -> Dict[str, Any]:
        """LangGraph 노드의 메인 진입점 함수."""
        work_id = current_state_dict.get("work_id", "UNKNOWN_WORK_ID_N05")
        log_extra = {"work_id": work_id}
        self.logger.info(f"N05_PersonaAnalysisNode 시작. 사용자 응답: '{user_response if user_response else '[없음]'}'.",
                         extra=log_extra)

        try:
            workflow_state = OverallWorkflowState(**current_state_dict)
            node_state = workflow_state.persona_analysis
        except ValidationError as e:
            self.logger.error(f"N05 State 유효성 검사 실패: {e}", extra=log_extra)
            current_state_dict.setdefault('persona_analysis', {})['error_message'] = str(e)
            return current_state_dict

        # --- 상태에 따른 분기 처리 ---

        # 1. 초기 실행: 4개 의견 후보 생성 및 첫 질문 제시
        if not node_state.opinion_candidates and not node_state.is_ready:
            self.logger.info("초기 실행 단계: 4개 의견 후보 생성 시작.", extra=log_extra)
            report_draft = workflow_state.report_draft.draft
            if report_draft and workflow_state.report_draft.is_ready:
                opinions = await self._run_initial_analysis(report_draft, work_id)
                node_state.opinion_candidates = opinions
                node_state.question = self._formulate_choice_question(opinions)
            else:
                self.logger.warning("참조할 보고서 초안이 없어 의견 생성을 건너뜁니다.", extra=log_extra)
                node_state.error_message = "보고서 초안이 없습니다."

        # 2. 사용자 선택 처리: 사용자가 의견 후보 중 하나를 선택하고 피드백을 줬을 때
        elif node_state.question and user_response and not node_state.selected_opinion:
            self.logger.info("사용자 의견 선택 및 피드백 처리 시작.", extra=log_extra)
            await self._process_user_choice(node_state, user_response, work_id)

        # 3. 최종 확인: 수정된 최종안에 대해 사용자가 마지막 피드백을 줬을 때
        elif node_state.selected_opinion and not node_state.is_ready and user_response:
            self.logger.info("수정된 최종안에 대한 사용자 피드백 처리 시작.", extra=log_extra)

            # --- 수정된 로직: 의도 분류 함수 사용 ---
            intent = await self._classify_user_response_intent(
                current_draft=node_state.selected_opinion.opinion_text,
                user_answer=user_response,
                work_id=work_id
            )

            if intent == 'CO':  # 사용자가 최종 확정한 경우
                self.logger.info("사용자가 최종 의견을 확정(CO)했습니다.", extra=log_extra)
                node_state.is_ready = True
                node_state.question = None
            elif intent == 'RE':  # 사용자가 추가 수정을 요청한 경우
                self.logger.info("사용자가 최종 의견에 추가 수정(RE)을 요청했습니다.", extra=log_extra)
                await self._revise_final_opinion(node_state, user_response, work_id)
            else:  # 'UN' - 의도가 불명확한 경우
                self.logger.info("사용자 의도가 불명확(UN)하여 추가 질문을 생성합니다.", extra=log_extra)
                node_state.question = "답변의 의도가 명확하지 않습니다. 현재 내용으로 확정하기를 원하시면 '네, 확정합니다.'와 같이, 수정을 원하시면 구체적인 수정 사항을 다시 한번 말씀해주시겠어요?"
            # --- 수정 끝 ---

        workflow_state.persona_analysis = node_state
        return await self._finalize_and_save_state(workflow_state, log_extra)

    # --- 신규 추가: 사용자 의도 분류 함수 ---
    async def _classify_user_response_intent(self, current_draft: str, user_answer: str, work_id: str) -> str:
        """사용자 피드백의 의도를 'CO'(확정), 'RE'(수정), 'UN'(불명확)으로 분류합니다."""
        prompt_sys = (
            "Your **sole and critical task** is to classify a user's Korean feedback on a given Korean draft text into one of three categories. "
            "The categories are: 'CO', 'RE', or 'UN'.\n"
            "1. 'CO': Indicates the user is satisfied with the current draft and implies no further changes are needed (e.g., '네 좋아요', '이대로 진행해주세요').\n"
            "2. 'RE': Indicates the user wants specific modifications, additions, or deletions to the current draft (e.g., 'A를 B로 수정해주세요', '좀 더 자세히 설명해주세요').\n"
            "3. 'UN': Indicates the user's intent isn't clearly one of the above or is a general non-committal comment (e.g., '흠...', '글쎄요').\n\n"
            "**IMPORTANT INSTRUCTIONS FOR YOUR RESPONSE:**\n"
            "- Your response MUST BE **ONLY ONE** of these exact English strings: 'CO', 'RE', or 'UN'.\n"
            "- **DO NOT** include any other text, reasoning, explanations, translations, or conversational filler.\n"
            "- **ABSOLUTELY NO XML-like tags (e.g., <think>, </think>, <REASONING>, <TRANSLATION_ATTEMPT>) are allowed in your output.**\n"
            "Output the single, precise classification label and nothing else."
        )
        prompt_user = (
            f"[User's Feedback (Korean)]\n{user_answer}\n\n"
            "Based *only* on the [User's Feedback], classify the user's intent. "
            "Choose and respond with only one of the following labels: 'CO', 'RE', 'UN'."
        )
        messages = [{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt_user}]

        valid_labels = ["CO", "RE", "UN"]
        try:
            resp = await self.llm.generate_text(
                messages=messages,
                request_id=f"intent-classify-strict-{work_id}",
                max_tokens=50,
                temperature=0.0,
                timeout=LLM_TIMEOUT_SEC / 4
            )
            raw_classification = resp.get("generated_text", "").strip().upper()

            if raw_classification in valid_labels:
                return raw_classification
            else:  # LLM이 지시를 어기고 다른 텍스트를 포함했을 경우 대비
                for label in valid_labels:
                    if label in raw_classification:
                        self.logger.warning(f"LLM 의도 분류 시 라벨 외 텍스트 포함: '{raw_classification}'. '{label}'로 처리.",
                                            extra={"work_id": work_id})
                        return label
                self.logger.error(f"LLM 의도 분류 결과가 유효 라벨 미포함: '{raw_classification}'. 'UN'으로 처리.",
                                  extra={"work_id": work_id})
                return "UN"
        except Exception as e:
            self.logger.error(f"LLM 의도 분류 중 오류: {e}", exc_info=True, extra={"work_id": work_id})
            return "UN"

    # --- 기존 헬퍼 함수들 ---
    async def _process_user_choice(self, node_state: PersonaAnalysisState, user_response: str, work_id: str):
        """사용자 선택을 파싱하고, 선택된 의견을 피드백에 따라 수정합니다."""
        try:
            match = re.match(r"^\s*(\d+)", user_response)
            if not match:
                node_state.question = "죄송합니다. 몇 번 의견을 선택하셨는지 번호를 포함하여 다시 말씀해주시겠어요?"
                return

            choice_index = int(match.group(1)) - 1
            if not (0 <= choice_index < len(node_state.opinion_candidates)):
                node_state.question = f"잘못된 번호를 선택하셨습니다. 1에서 {len(node_state.opinion_candidates)} 사이의 번호를 선택해주세요."
                return

            chosen_opinion = node_state.opinion_candidates[choice_index]
            feedback_text = re.sub(r"^\s*\d+[\.\)\s]*", "", user_response).strip()

            # 피드백 텍스트의 의도를 분류하여 처리
            intent = await self._classify_user_response_intent(
                current_draft=chosen_opinion.opinion_text,
                user_answer=feedback_text if feedback_text else "좋아요",  # 피드백 없으면 긍정으로 간주
                work_id=work_id
            )

            if intent == 'RE':  # 수정 요청이 명확할 경우
                self.logger.info(f"사용자가 '{chosen_opinion.persona_name}' 의견에 대한 수정(RE)을 요청했습니다.",
                                 extra={"work_id": work_id})
                revised_opinion_text = await self._revise_opinion_with_feedback(chosen_opinion, feedback_text, work_id)
                node_state.selected_opinion = Opinion(
                    persona_id=chosen_opinion.persona_id,
                    persona_name=chosen_opinion.persona_name,
                    opinion_text=revised_opinion_text
                )
            else:  # CO 또는 UN (수정 없이 선택한 것으로 간주)
                self.logger.info(f"사용자가 '{chosen_opinion.persona_name}' 의견을 수정 없이 선택(CO/UN)했습니다.",
                                 extra={"work_id": work_id})
                node_state.selected_opinion = chosen_opinion

            node_state.question = self._formulate_final_confirmation_question(node_state.selected_opinion)

        except (ValueError, IndexError) as e:
            self.logger.error(f"사용자 응답 처리 중 오류 발생: {e}", extra={"work_id": work_id})
            node_state.question = "응답을 처리하는 중 오류가 발생했습니다. 다시 시도해 주시겠어요?"

    # _run_initial_analysis, _revise_opinion_with_feedback, _revise_final_opinion 등 다른 모든 헬퍼 함수는 이전 답변과 동일하게 유지됩니다.
    # ... (이전 답변의 나머지 모든 헬퍼 함수들을 여기에 붙여넣기) ...
    async def _run_initial_analysis(self, report_text: str, work_id: str) -> List[Opinion]:
        personas = await self._get_unique_random_personas(count=4)
        if not personas:
            self.logger.error("DB에서 페르소나를 가져오지 못해 분석을 중단합니다.", extra={"work_id": work_id})
            return []
        opinion_tasks = [self._generate_single_opinion(persona, report_text, work_id) for persona in personas]
        generated_opinions = await asyncio.gather(*opinion_tasks)
        return [op for op in generated_opinions if op]

    async def _revise_opinion_with_feedback(self, original_opinion: Opinion, feedback: str, work_id: str) -> str:
        prompt = f"""You are a helpful assistant that revises a text based on user feedback.
[Original Persona and Opinion]
- Persona Name: {original_opinion.persona_name}
- Original Text (in Korean): {original_opinion.opinion_text}
[User's Revision Request (in Korean)]
- Feedback: {feedback}
[TASK]
Revise the [Original Text] to reflect the user's feedback. Maintain the persona's original tone and core perspective as much as possible while incorporating the changes.
Your response MUST be ONLY the revised, final text in KOREAN.
"""
        try:
            response = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"revise-opinion-{original_opinion.persona_id}-{work_id}",
                temperature=0.5)
            return response.get("generated_text", original_opinion.opinion_text).strip()
        except Exception as e:
            self.logger.error(f"의견 수정 중 LLM 오류: {e}", extra={"work_id": work_id})
            return f"{original_opinion.opinion_text}\n\n[수정 요청 처리 중 오류가 발생했습니다: {feedback}]"

    async def _revise_final_opinion(self, node_state: PersonaAnalysisState, feedback: str, work_id: str):
        revised_text = await self._revise_opinion_with_feedback(node_state.selected_opinion, feedback, work_id)
        node_state.selected_opinion.opinion_text = revised_text
        node_state.question = self._formulate_final_confirmation_question(node_state.selected_opinion)

    def _formulate_choice_question(self, opinions: List[Opinion]) -> str:
        options_text = "\n\n".join(
            [f"### {i + 1}. {op.persona_name}의 관점\n{op.opinion_text}" for i, op in enumerate(opinions)])
        return (
            "보고서에 대해 다음과 같은 4가지 다른 관점의 의견을 생성했습니다.\n\n"
            f"{options_text}\n\n"
            "어떤 의견을 바탕으로 발전시켜 볼까요? 번호를 선택하여 의견을 말씀해주세요. (예: 2번 의견은 좋은데, 좀 더 날카롭게 다듬어주세요.)"
        )

    def _formulate_final_confirmation_question(self, final_opinion: Opinion) -> str:
        return (
            "요청하신 내용을 반영하여 다음과 같이 최종 의견을 수정했습니다.\n\n"
            f"### {final_opinion.persona_name}의 최종 관점\n{final_opinion.opinion_text}\n\n"
            "이 내용으로 최종 확정할까요? 더 수정할 부분이 있다면 말씀해주세요."
        )

    async def _get_unique_random_personas(self, count: int) -> List[Dict[str, Any]]:
        try:
            query = f"SELECT id, name, data FROM ai_test_personas ORDER BY RANDOM() LIMIT {count};"
            records = await self.db.fetch_all(query)
            return [dict(rec) for rec in records]
        except Exception as e:
            self.logger.error(f"여러 페르소나 조회 중 DB 오류 발생: {e}", exc_info=True)
            return []

    async def _generate_single_opinion(self, persona: Dict[str, Any], report_text: str, work_id: str) -> Opinion | None:
        prompt = self._build_english_prompt(persona, report_text)
        try:
            response = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"initial-opinion-{persona['id']}-{work_id}",
                max_tokens=1500,
                temperature=0.75)
            opinion_text = response.get("generated_text", "").strip()
            if opinion_text:
                return Opinion(persona_id=persona['id'], persona_name=persona['name'], opinion_text=opinion_text)
            return None
        except Exception as e:
            self.logger.error(f"단일 의견 생성 중 LLM 오류: {e}", extra={"work_id": work_id, "persona_id": persona.get('id')})
            return None

    def _build_english_prompt(self, persona: Dict[str, Any], report_text: str) -> str:
        p_data = {}
        p_data_raw = persona.get('data', {})
        if isinstance(p_data_raw, str):
            try:
                p_data = json.loads(p_data_raw)
            except json.JSONDecodeError:
                self.logger.error(f"페르소나 data 필드의 JSON 파싱에 실패했습니다. 내용: {p_data_raw}")
        elif isinstance(p_data_raw, dict):
            p_data = p_data_raw

        persona_name = p_data.get('name', persona.get('name', ''))
        summary = p_data.get('summary', 'No summary defined.')
        background = p_data.get('background', '')
        worldview = p_data.get('worldview', '')
        key_questions = '\n- '.join(p_data.get('key_questions', []))
        tone_manner = ', '.join(p_data.get('tone_manner', []))
        vocab = p_data.get('vocabulary', '')
        rhetoric = p_data.get('rhetoric', '')

        prompt = f"""You are now to act as the persona: '{persona_name}'.
            [ROLE]
            {summary}
            {background}
            [MINDSET]
            Worldview: "{worldview}"
            When analyzing an issue, you must ask yourself the following questions:
            - {key_questions}
            [COMMUNICATION STYLE]
            Your tone and manner are: {tone_manner}.
            Your vocabulary style is: {vocab}.
            Your rhetorical style is: {rhetoric}.
            [TASK]
            Now, analyze the following [REPORT TEXT] from your persona's perspective. You must generate a concise and insightful opinion.
            **IMPORTANT REQUIREMENT: Your entire response MUST be written in KOREAN.**
            [REPORT TEXT]
            ---
            {report_text[:8000]}
            ---
        """
        return prompt

    async def _finalize_and_save_state(self, workflow_state: OverallWorkflowState, log_extra: Dict) -> Dict[str, Any]:
        """최종 상태를 저장하고 반환합니다."""
        updated_state_dict = workflow_state.model_dump(mode='json')
        await self._save_workflow_state_to_redis(workflow_state.work_id, updated_state_dict)
        self.logger.info("N05 노드 처리 완료 및 상태 저장.", extra=log_extra)
        return updated_state_dict

    async def _save_workflow_state_to_redis(self, work_id: str, state_dict: Dict[str, Any]):
        """워크플로우 상태를 Redis에 저장합니다."""
        key = f"workflow:{work_id}:full_state"
        try:
            json_compatible_state = json.loads(json.dumps(state_dict, default=str))
            await self.redis.set(key, json_compatible_state, expire=60 * 60 * 6)
        except Exception as e:
            self.logger.error(f"Redis 상태 저장 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})