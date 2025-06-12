# ai/app/nodes_v3/n_02_report_search_planning_node.py
"""n_02_ReportSearchPlanningNode (Upgraded)

NewSum LangGraph 워크플로우의 **두 번째 노드**(순번 n_02).
'지능형 보고서 설계 에이전트'로서, 확정된 주제(`OverallWorkflowState.topic_clarification.draft`)를
심층적으로 분석 및 분해하고, 여러 보고서 구조(목차) 대안을 생성 및 평가하며,
자가 검증을 통해 최종적으로 '참고 계획 수준'의 보고서 구조를 설계한다.

주요 워크플로우:
1.  **심층 질의 분석**: 사용자 질의를 개체, 관계, 하위 질문으로 구조화하여 분해한다.
2.  **다중 목차 생성/선택**: 분해된 질문을 바탕으로 여러 목차 초안을 생성하고, 그중 최적안을 선택/종합한다.
3.  **자가 검증 (Reflection)**: 선택된 목차를 비판적으로 검토하고 스스로 개선하여 완성도를 높인다.
4.  **사용자 피드백 (Optional)**: 최종 확정된 계획을 사용자에게 제시하고 필요시 피드백을 받아 최종 확정한다.

State 모델 (`ReportPlanningPydanticState`) 요구사항:
---------------------------------------------
- `query_analysis` (Dict): 1단계의 질의 분석 결과 저장.
  - e.g., `{'entities': [...], 'relationships': [...], 'sub_questions': [...]}`
- `outline_candidates` (List[Dict]): 2단계에서 생성된 복수 목차 후보 저장.
- `structure` (Dict): 3단계 자가 검증을 거쳐 최종 확정된 보고서 구조(목차).
- `search_plan` 필드는 이 노드에서 더 이상 사용되지 않음.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.config.settings import Settings
from app.services.database_client import DatabaseClient
from app.services.llm_service import LLMService
from app.utils.logger import get_logger
from app.workflows.state_v3 import OverallWorkflowState, ReportPlanningPydanticState

# ──────────────────────────────────────────────────────────────────────────────
# 설정 및 상수
# ──────────────────────────────────────────────────────────────────────────────
settings = Settings()
REDIS_WORKFLOW_KEY_TMPL = "workflow:{work_id}:full_state"
LLM_TIMEOUT_SEC = 90
SECTION_MIN = 3
logger = get_logger("n_02_ReportSearchPlanningNode_v2")


# ──────────────────────────────────────────────────────────────────────────────
# 노드 구현
# ──────────────────────────────────────────────────────────────────────────────
class N02ReportSearchPlanningNode:
    """LangGraph async 노드 – 지능형 보고서 설계 (n_02)"""

    def __init__(
            self,
            redis_client: DatabaseClient, # redis_client 파라미터 다시 추가
            llm_service: LLMService
    ):
        self.redis = redis_client # self.redis에 할당
        self.llm = llm_service
        self.logger = logger

    async def __call__(
            self,
            current_state_dict: Dict[str, Any],
            user_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """노드 실행: 질의 분석 -> 목차 생성/선택 -> 자가 검증 -> 최종 확정"""
        raw_work_id = current_state_dict.get("work_id")
        log_extra = {"work_id": raw_work_id or "UNKNOWN_WORK_ID_N02"}
        self.logger.info("N02ReportSearchPlanningNode (Upgraded) 시작.", extra=log_extra)

        try:
            workflow_state = OverallWorkflowState(**current_state_dict)
        except ValidationError as e:
            self.logger.error(f"Pydantic state 유효성 검사 오류: {e}", extra=log_extra)
            current_state_dict["error_message"] = f"N02 State 유효성 검사 실패: {e}"
            return current_state_dict

        node_state = workflow_state.report_planning
        work_id = workflow_state.work_id
        topic_draft = workflow_state.topic_clarification.draft

        # 1️⃣ 선행 조건 확인
        if not workflow_state.topic_clarification.is_final or not topic_draft:
            self.logger.error("주제가 확정되지 않아 N02 노드 실행을 중단합니다.", extra=log_extra)
            workflow_state.error_message = "N02: Topic is not finalized or missing."
            return workflow_state.model_dump(exclude_none=True)

        # 2️⃣ 사용자 피드백 처리 (기존 계획에 대한 피드백이 있을 경우)
        if node_state.planning_question and user_response:
            self.logger.info(f"사용자 피드백 처리 시작: '{user_response[:50]}...'", extra=log_extra)
            await self._process_user_feedback(node_state, user_response, work_id,topic_draft=topic_draft)
            self.logger.info("사용자 피드백 처리 완료.", extra=log_extra)
            # 피드백 처리 후 상태를 반환하여 워크플로우가 다음 단계를 결정하게 함
            return workflow_state.model_dump(exclude_none=True)

        # 3️⃣ 새로운 보고서 설계 프로세스 실행
        if not node_state.is_ready and not node_state.planning_question:
            await self._design_report_structure_pipeline(node_state, topic_draft, work_id)

        updated_state_dict = workflow_state.model_dump(mode='json')
        await self._save_workflow_state_to_redis(work_id, updated_state_dict)

        # 4️⃣ 최종 상태 반환
        self.logger.info("N02ReportSearchPlanningNode (Upgraded) 종료.", extra=log_extra)
        return workflow_state.model_dump(exclude_none=True)

    async def _save_workflow_state_to_redis(self, work_id: str, state_dict: Dict[str, Any]):
        """워크플로우의 현재 상태를 Redis에 저장합니다."""
        key = f"workflow:{work_id}:full_state"
        log_extra = {"work_id": work_id, "redis_key": key}
        logger.info("워크플로우 상태를 Redis에 저장 시도.", extra=log_extra)
        try:
            # Pydantic 모델이 포함된 state_dict를 JSON 직렬화 가능한 dict로 변환
            # model_dump(mode='json')을 사용하여 datetime 등 비직렬화 객체 처리
            json_compatible_state = json.loads(json.dumps(state_dict, default=str))

            await self.redis.set(key, json_compatible_state, expire=60 * 60 * 6)  # 6시간 TTL
            logger.info("Redis 상태 저장 완료.", extra=log_extra)
        except Exception as e:
            logger.error(f"Redis 상태 저장 중 오류 발생: {e}", exc_info=True, extra=log_extra)

    async def _design_report_structure_pipeline(self, node_state: ReportPlanningPydanticState, topic: str,
                                                work_id: str):
        """
        [개선된 버전] 보고서 설계를 위한 전체 파이프라인. 실패 시 구체적인 피드백을 요청합니다.
        """
        log_extra = {"work_id": work_id}
        self.logger.info("보고서 설계 파이프라인 시작.", extra=log_extra)

        # --- 단계 1: 심층 질의 분석 및 분해 ---
        query_analysis = await self._analyze_and_decompose_query(topic, work_id)
        node_state.query_analysis = query_analysis

        # [실패 지점 1] 질의 분석 자체를 실패한 경우
        if not query_analysis or not query_analysis.get("sub_questions"):
            self.logger.error("1단계 (질의 분해)에 실패하여 파이프라인을 중단합니다.", extra=log_extra)

            # 사용자에게 전달할 구체적인 질문 생성
            node_state.planning_question = """AI가 보고서의 핵심 내용을 분석하는 데 어려움을 겪고 있습니다.
    이는 보고서의 주제가 다소 추상적이거나, 분석해야 할 핵심 대상과 관점이 명확하지 않기 때문일 수 있습니다.

    보고서의 방향을 명확히 하기 위해, 아래 내용 중 하나를 구체적으로 알려주시겠어요?

    - **핵심 질문**: 이 보고서를 통해 가장 알고 싶은 핵심 질문은 무엇인가요? (예: "AI 환각 현상의 가장 큰 원인은?")
    - **주요 대상/키워드**: 보고서가 반드시 다루어야 할 인물, 기술, 사건을 지정해주세요. (예: "오픈AI의 GPT-4", "검색 증강 생성(RAG) 기술")
    - **분석 관점**: 어떤 관점에 집중할까요? (예: "기술적 원인 분석에 집중", "산업에 미치는 경제적 영향 중심")"""
            return  # 파이프라인 중단

        # --- 단계 2: 다중 목차 생성 및 최적안 선택 ---
        selected_outline, candidates = await self._generate_and_select_outline(query_analysis, work_id)
        node_state.outline_candidates = candidates

        # [실패 지점 2] 질의 분석은 성공했으나, 목차 생성을 실패한 경우
        if not selected_outline:
            self.logger.error("2단계 (목차 생성/선택)에 실패하여 파이프라인을 중단합니다.", extra=log_extra)

            # 사용자에게 전달할 구체적인 질문 생성
            node_state.planning_question = """AI가 보고서의 핵심 내용은 파악했지만, 논리적인 목차를 구성하는 데 실패했습니다.
    이는 분석된 내용들을 어떤 순서로 배치해야 할지 판단하기 어렵기 때문입니다.

    보고서의 전체적인 틀을 잡을 수 있도록, 원하시는 보고서의 대략적인 구조나 포함되었으면 하는 섹션 제목들을 알려주시겠어요?

    - **예시 1**: 서론 / 기술적 배경 / 시장 현황 / 문제점 및 해결 방안 / 결론
    - **예시 2**: 1. AI 도입 현황, 2. 주요 성공 사례 분석, 3. 도입 시 고려사항, 4. 향후 전망
    - 또는, 가장 중요하게 생각하는 섹션 2~3개만 알려주셔도 좋습니다."""
            return  # 파이프라인 중단

        # --- 단계 3: 자가 검증 및 정제 (Reflection) ---
        final_outline = await self._reflect_and_refine_outline(selected_outline, query_analysis, work_id)
        node_state.structure = final_outline

        # --- 단계 4: 최종 계획 확정 및 사용자 피드백 요청 ---
        if self._check_planning_sufficiency(final_outline):
            self.logger.info("시스템이 설계한 보고서 구조가 충분하다고 판단되어 사용자에게 최종 확인을 요청합니다.", extra=log_extra)
            # is_ready는 False로 유지하여 사용자 입력을 기다림
            node_state.is_ready = False
            # '확인'을 위한 질문을 생성
            node_state.planning_question = self._create_confirmation_question(final_outline)
        else:
            self.logger.info("최종 목차 설계안에 대해 시스템 테스트 결과 개선이 필요하다고 판단됩니다.\n 다음 질문에 답변 부탁드립니다.", extra=log_extra)
            node_state.is_ready = False
            # '개선'을 위한 질문을 생성 (기존 로직 재사용)
            node_state.planning_question = self._generate_planning_question(node_state)
        # --- [수정 끝] ---

    async def _analyze_and_decompose_query(self, topic: str, work_id: str) -> Dict[str, Any]:
        """(1단계) 질의를 분석하고 구조화된 하위 질문으로 분해합니다. """
        log_extra = {"work_id": work_id}
        self.logger.info(f"질의 분석 및 분해 시작. Topic: '{topic}'", extra=log_extra)
        prompt = f"""You are an expert analyst. Your task is to deeply analyze a user's query and decompose it into structured components for a comprehensive report.
Analyze the following query: "{topic}"

Decompose it into:
1.  **Key Entities**: Identify the main subjects, organizations, locations, and concepts.
2.  **Core Relationship**: Describe the central cause-and-effect or relationship being investigated.
3.  **Sub-questions**: Formulate 5-7 specific, answerable questions that must be addressed to fully cover the query's scope. These questions will form the basis of the report's structure.

Provide the output strictly in JSON format with the keys "entities", "relationship", and "sub_questions".
Example for "Impact of tariffs on Samsung's US market":
{{
  "entities": ["Samsung", "US market", "tariffs"],
  "relationship": "Analyzing how tariffs are impacting Samsung's position in the US market.",
  "sub_questions": [
    "What are the current tariffs applicable to Samsung's products in the US?",
    "How do these tariffs affect Samsung's production costs and pricing strategy?",
    "What is the trend of Samsung's market share and profitability in the US?",
    "How are competitors like Apple affected by or reacting to these tariffs?",
    "What are the overall strategic implications for Samsung in the US market?"
  ]
}}
"""
        try:
            resp = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"decompose-query-{work_id}",
                max_tokens=1024,
                temperature=0.2
            )
            parsed_json = json.loads(resp.get("generated_text", "{}").strip())
            self.logger.info("질의 분해 성공.", extra=log_extra)
            self.logger.debug(f"분해된 질의: {parsed_json}", extra=log_extra)
            return parsed_json
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"질의 분해 중 오류 발생: {e}", extra=log_extra)
            return {}

    async def _generate_and_select_outline(self, query_analysis: Dict[str, Any], work_id: str) -> (
    Optional[Dict], List[Dict]):
        """(2단계) 여러 목차 후보를 생성하고 최적안을 선택/종합합니다. """
        log_extra = {"work_id": work_id}
        sub_questions_str = "\n".join([f"- {q}" for q in query_analysis.get("sub_questions", [])])
        self.logger.info("다중 목차 후보 생성 시작.", extra=log_extra)

        # 2a. 복수 후보 생성
        prompt_generate = f"""Based on the following key analytical questions, please propose 3 distinct and logical report outlines (in Korean). Each outline should be a JSON object with section titles as keys and a list of sub-headings as values.

            **Analytical Questions:**
            {sub_questions_str}
            
            **Output Format:**
            Provide a JSON object with a key "outlines", which is a list containing the 3 outline JSON objects.
            Example:
            {{
              "outlines": [
                {{ "서론": [], "기술 동향 분석": ["핵심 기술", "경쟁 기술"], "결론": [] }},
                {{ "도입": [], "시장 환경 분석": ["시장 규모", "주요 플레이어"], "전망": [] }},
                {{ "문제 제기": [], "원인 분석": ["정책적 원인", "기술적 원인"], "해결 방안": [], "결론": [] }}
              ]
            }}
        """
        try:
            resp_gen = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt_generate}],
                request_id=f"generate-outlines-{work_id}",
                max_tokens=2048,
                temperature=0.5
            )
            candidates_data = json.loads(resp_gen.get("generated_text", "{}").strip())
            candidates = candidates_data.get("outlines", [])
            if not candidates or len(candidates) < 1:
                self.logger.warning("충분한 수의 목차 후보를 생성하지 못했습니다.", extra=log_extra)
                return None, []
            self.logger.info(f"{len(candidates)}개의 목차 후보 생성 완료.", extra=log_extra)
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"목차 후보 생성 중 오류 발생: {e}", extra=log_extra)
            return None, []

        # 2b. 최적안 선택
        self.logger.info("최적 목차 선택 시작.", extra=log_extra)
        candidates_str = json.dumps(candidates, ensure_ascii=False, indent=2)
        prompt_select = f"""You are a senior editor. Below are several draft outlines for a report. Your task is to select the single best outline or synthesize a new, superior outline by combining the strengths of the given candidates. The final outline should be the most logical, comprehensive, and clear for a professional report.

**Candidate Outlines:**
{candidates_str}

**Task:**
Return a single JSON object representing the final, chosen outline. Do not add any explanatory text.
"""
        try:
            resp_select = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt_select}],
                request_id=f"select-outline-{work_id}",
                max_tokens=1024,
                temperature=0.1
            )
            selected_outline = json.loads(resp_select.get("generated_text", "{}").strip())
            self.logger.info("최적 목차 선택 완료.", extra=log_extra)
            self.logger.debug(f"선택된 목차: {selected_outline}", extra=log_extra)
            return selected_outline, candidates
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"목차 선택 중 오류 발생: {e}", extra=log_extra)
            return None, candidates

    async def _reflect_and_refine_outline(self, outline: Dict[str, Any], query_analysis: Dict[str, Any],
                                          work_id: str) -> Dict[str, Any]:
        """(3단계) 자가 검증을 통해 목차를 비판적으로 검토하고 개선합니다. """
        log_extra = {"work_id": work_id}
        self.logger.info("목차 자가 검증 (Reflection) 시작.", extra=log_extra)
        outline_str = json.dumps(outline, ensure_ascii=False, indent=2)
        sub_questions_str = "\n".join(query_analysis.get("sub_questions", []))

        # --- [수정된 프롬프트] ---
        prompt = f"""As a critical expert analyst, review the following report outline. Your task is to provide a refined and improved version that is perfectly structured for a subsequent research and writing phase.

            **Core Analytical Questions:**
            {sub_questions_str}
            
            **Current Outline to Review:**
            {outline_str}
            
            **Task:**
            Return a single JSON object of the final, refined outline. The keys of the object must be the main section titles. The value for each key **must be another JSON object** containing three keys:
            1.  `"role"`: A **concrete descriptive sentence** in Korean that explains **what this section does for the reader or for the report's overall logical flow** (e.g., "This section serves as the introduction, explaining the background and purpose of the report to help the reader's understanding," or "This section analyzes the core problem by presenting statistical data and case studies.").
            2.  `"description"`: A single Korean sentence (around 40-50 characters) that summarizes the core purpose of this section. This will be used as a high-context search query.
            3.  `"sub_topics"`: A list of strings representing the sub-headings for that section.
            
            **Example Output Format:**
            {{
              "서론": {{
                "role": "보고서의 연구 배경과 핵심 질문을 제시하여, 논의의 필요성과 방향을 설정하는 도입부 역할을 합니다.",
                "description": "AI 기술 확산에 따른 언론사의 위기 상황과 보고서의 목적을 설명합니다.",
                "sub_topics": [ "AI 기반 검색 기술 확산의 배경", "언론사의 전통적 수익 모델 개요" ]
              }}
            }}
            
            Your output must be **only the JSON** and nothing else.
        """
        try:
            resp = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"reflect-outline-{work_id}",
                max_tokens=1536,
                temperature=0.1
            )
            refined_outline = json.loads(resp.get("generated_text", "{}").strip())
            self.logger.info("자가 검증 및 정제 완료 (role 포함된 고도화 구조).", extra=log_extra)
            self.logger.debug(f"정제된 최종 목차: {refined_outline}", extra=log_extra)
            return refined_outline
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"자가 검증 중 오류 발생: {e}. 기존 목차를 반환합니다.", extra=log_extra)
            # 오류 발생 시, 새 구조에 맞게 기존 목차를 변환 시도
            if isinstance(outline, dict):
                return {
                    sec: {"role": "역할 미정의", "description": "내용 미정의", "sub_topics": sub}
                    for sec, sub in outline.items()
                }
            return outline

    def _create_confirmation_question(self, structure: Dict[str, Any]) -> str:
        """
        [가독성 개선 버전]
        성공적으로 생성된 목차를 사용자에게 제시하고 최종 동의를 구하는 질문을 생성합니다.
        """
        self.logger.info("최종 확인 질문 생성 시작 (가독성 개선).")

        # 각 대목차 사이에 두 줄의 줄바꿈을 넣어 시각적으로 분리합니다.

        question = f"""AI가 아래와 같이 조사 계획을 제안합니다.

            [ **제안된 최종 계획안** ]
            {json.dumps(structure, indent=2, ensure_ascii=False)}
            
            이 계획대로 진행할까요?
            '네' 또는 '진행'이라고 입력하여 확정하거나, 수정하고 싶은 내용을 알려주세요.
        """

        return question

    def _check_planning_sufficiency(self, structure: Dict[str, Any]) -> bool:
        """최종적으로 설계된 구조가 충분히 상세하고 논리적인지 검사합니다."""
        if not structure or not isinstance(structure, dict):
            return False
        # 최소 섹션 수 확인 및 각 섹션에 소제목이 하나 이상 있는지 확인하는 등의 논리 추가 가능
        is_sufficient = len(structure.keys()) >= SECTION_MIN
        self.logger.info(f"계획 충분성 검사 결과: {is_sufficient} (섹션 수: {len(structure.keys())})")
        return is_sufficient

    def _generate_planning_question(self, node_state: ReportPlanningPydanticState) -> str:
        """
        State를 분석하여 상황에 맞는 사용자 질문을 생성합니다.
        """
        log_extra = {"work_id": node_state.work_id if hasattr(node_state, 'work_id') else "UNKNOWN"}
        self.logger.info("상황별 사용자 확인 질문 생성 시작.", extra=log_extra)

        # --- [개선안 2] 목차 후보는 있으나 최종안 선택에 실패한 경우 ---
        # 사용자에게 직접 선택권을 제공합니다.
        if not node_state.structure and node_state.outline_candidates:
            self.logger.warning("최종 목차는 없으나 후보가 있어 사용자에게 선택을 요청합니다.", extra=log_extra)

            question_parts = [
                "AI가 아래와 같이 3개의 보고서 목차 후보를 생성했으나, 최종안을 확정하는 데 어려움을 겪고 있습니다.",
                "어떤 안으로 보고서를 구성할까요? 번호를 선택해주시거나, 직접 수정하고 싶은 내용을 알려주세요.\n"
            ]

            for i, candidate in enumerate(node_state.outline_candidates):
                # 후보 목차를 읽기 쉽게 요약하여 문자열로 변환
                structure_summary = "\n".join(
                    f"  - {sec}: {', '.join(sub) if sub else ''}" for sec, sub in list(candidate.items())[:4]
                    # 최대 4개 섹션만 표시
                )
                if len(candidate) > 4:
                    structure_summary += "\n  ..."

                question_parts.append(f"**[후보 {i + 1}]**\n{structure_summary}\n")

            return "\n".join(question_parts)

        # --- 기존 Fallback 질문 (주제 자체를 분석하기 어려운 경우) ---
        # (예: query_analysis 단계부터 실패했을 때 사용)
        return """
        죄송합니다, 보고서의 구체적인 조사 계획을 세우는 데 어려움을 겪고 있습니다.
        해당 메세지는 현재 주제는 1차적으로 세분화되었지만, 자동으로 목차를 생성하기에는 다루어야 할 범위가 넓거나 분석의 핵심 초점이 명확하지 않을 수 있습니다.
        보고서가 원하시는 방향으로 구성될 수 있도록, 아래 내용 중 한두 가지만이라도 구체적으로 알려주시겠어요?
        
        핵심 질문: 이 보고서를 통해 가장 알고 싶은 핵심 질문 2~3개를 알려주세요.
            (예시: "AI 환각 현상의 가장 큰 원인은 무엇인가?", "미디어는 이 문제를 해결하기 위해 어떤 기술을 도입해야 하는가?")
            
            주요 키워드 또는 대상: 보고서가 반드시 다루어야 할 중요 인물, 기술, 사건을 지정해주세요.
            (예시: "오픈AI의 GPT-4 모델", "검색 증강 생성(RAG) 기술", "뉴욕타임스 AI 소송 사례")
            
            원하는 분석 관점: 보고서가 집중했으면 하는 분석의 방향이나 관점을 알려주세요.
            (예시: "기술적 원인 분석에 집중", "산업에 미치는 경제적 영향 중심으로", "법적 및 윤리적 쟁점 위주로")
        """

    async def _process_user_feedback(
            self,
            node_state: ReportPlanningPydanticState,
            feedback: str,
            work_id: str,
            topic_draft: str
    ) -> None:
        """
        [수정된 버전]
        사용자 피드백을 처리합니다. '확정' 의도 외의 모든 피드백은
        기존 주제와 결합하여 전체 계획을 '재생성'하는 데 사용됩니다.
        """
        log_extra = {"work_id": work_id}
        self.logger.info("사용자 피드백 처리 시작 (재생성 기반 로직).", extra=log_extra)
        node_state.planning_answer = feedback

        # 1. 사용자 피드백 의도 분류 ('CO': 확정, 'RE': 수정, 'UN': 불명확)
        intent = await self._classify_user_response_intent(feedback, work_id)

        # 2. [확정] 사용자가 계획에 동의한 경우
        if intent == 'CO':
            self.logger.info("사용자 의도: '확정(CO)'. 계획을 최종 확정합니다.", extra=log_extra)
            node_state.is_ready = True
            node_state.planning_question = None  # 질문 상태 초기화
            return

        # 3. [수정 또는 불명확] 그 외 모든 경우 (수정, 추가 요청, 불명확한 답변 등)
        #    -> 주제를 보강하여 계획 수립 파이프라인 전체를 재실행합니다.
        self.logger.info(f"사용자 의도 감지: '{intent}'. 피드백을 반영하여 계획 전체를 재생성합니다.", extra=log_extra)

        # --- [사용자 요청에 따른 수정 지점] ---
        # `refined_topic`에 할당되는 프롬프트 문자열을 영문으로 변경합니다.
        # 이 문자열은 _design_report_structure_pipeline의 첫 단계인 _analyze_and_decompose_query에
        # 전달될 새로운 '쿼리'가 됩니다.
        refined_topic = f"""Initial Topic: "{topic_draft}"
            User Feedback/Request: "{feedback}"
            Synthesize the initial topic with the user's feedback to form a new, refined topic. 
            Your main task is to analyze this synthesized topic to create a report outline. 
            Treat the user feedback as a directive to adjust the scope and focus of the original topic.
        """
        # --- [수정 완료] ---

        self.logger.info(
            "A new, refined topic has been constructed in English. Rerunning the entire design pipeline.",
            extra=log_extra)
        # 3-2. 보강된 주제로 보고서 설계 파이프라인(_design_report_structure_pipeline)을 다시 호출
        #      이를 통해 질의 분석부터 목차 생성, 검증까지 모든 과정이 새로 실행됩니다.
        await self._design_report_structure_pipeline(node_state, refined_topic, work_id)

    async def _classify_user_response_intent(self, user_answer: str, work_id: str) -> str:
        # 시스템 프롬프트 (영문) - "<THINK>" 태그 등 불필요한 출력 금지 지시 강화
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

        # 사용자 프롬프트 (핵심 정보만 전달)
        prompt_user = (
            #f"[Current Draft (Korean)]\n{current_draft}\n\n"
            f"[User's Feedback (Korean)]\n{user_answer}\n\n"
            "Based *only* on the [User's Feedback], classify the user's intent. "
            "Choose and respond with only one of the following labels: 'CO', 'RE', 'UN'."
        )
        messages = [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt_user},
        ]
        self.logger.info(f"LLM 사용자 답변 의도 분류 요청 (강화된 프롬프트). User Answer: '{user_answer[:50]}...'",
                         extra={"work_id": work_id})

        valid_labels = ["CO", "RE", "UN"]

        try:
            resp = await self.llm.generate_text(
                messages=messages,
                request_id=f"intent-classify-strict-{work_id}",  # request_id 변경 가능
                max_tokens=200,  # 라벨 길이를 고려한 최소한의 토큰 수 (예: "CO"은 약 3-4 토큰)
                temperature=0.0,  # 결정적이고 일관된 출력 유도
                timeout=LLM_TIMEOUT_SEC  # 매우 짧은 시간 내 응답 기대
            )
            raw_classification = resp.get("generated_text", "").strip().upper()
            # LLMService에서 <think> 태그 등을 이미 제거한다고 가정.
            # 만약 제거하지 않는다면, 여기서 추가 제거 로직이 필요할 수 있으나,
            # 프롬프트에서 강력히 금지했으므로 우선 LLM이 지시를 따를 것으로 기대.
            self.logger.debug(f"LLM raw classification response (strict prompt): '{raw_classification}'",
                              extra={"work_id": work_id})

            classification = "UN"  # 기본값

            # 1. LLM이 정확히 라벨만 반환했는지 먼저 확인
            if raw_classification in valid_labels:
                classification = raw_classification
            else:
                # 2. 혹시 라벨 주변에 약간의 불필요한 공백이나 예측 못한 문자가 붙었을 경우를 대비하여,
                #    응답 문자열 내에 유효한 라벨이 "포함"되어 있는지 확인.
                #    (이 로직은 LLM이 프롬프트 지시를 완벽히 따르지 못할 경우를 위한 대비책)
                found_label = False
                for label in valid_labels:
                    if label in raw_classification:  # 대소문자 구분 없이 비교하려면 .upper() 등 활용
                        classification = label
                        found_label = True
                        self.logger.warning(
                            f"LLM 의도 분류 시 라벨 외의 텍스트가 포함되었으나, 유효 라벨 '{label}' 추출 성공: '{raw_classification}'.",
                            extra={"work_id": work_id})
                        break
                if not found_label:
                    self.logger.error(f"LLM 의도 분류 결과가 유효한 라벨을 포함하지 않음: '{raw_classification}'. 최종 'UN'로 처리.",
                                      extra={"work_id": work_id})
                    classification = "UN"  # 안전하게 UN로 처리

            self.logger.info(f"LLM 사용자 답변 의도 분류 최종 결과 (강화된 프롬프트): {classification}", extra={"work_id": work_id})
            return classification
        except Exception as e:
            self.logger.error(f"LLM 사용자 답변 의도 분류 중 오류 (강화된 프롬프트): {e}", exc_info=True, extra={"work_id": work_id})
            return "UN"  # 오류 발생 시 안전하게 UN로 처리