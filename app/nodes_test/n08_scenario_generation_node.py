# ai/app/nodes_v2/n08_scenario_generation_node.py

import json
from typing import Any, Dict, List
from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)

class N08ScenarioGenerationNode:
    """
    n07a의 simple_scenario를 읽어서, qwen3-awq 모델을 호출하여
    ‘thumbnail_details’와 ‘panel_details’(4컷 각각)를 생성합니다.

    제약사항:
    - 동일 캐릭터는 외형 묘사를 일관되게 유지할 것.
    - 장면 흐름이 자연스럽게 이어지도록 할 것.
    - 등장인물은 최대 3명 이내로 제한할 것(관중처럼 너무 많은 인물 지양).
    - 스타일 변화가 과도하지 않도록, lora 미적용 상태에서도 일관된 톤을 유지.
    - 출력은 반드시 JSON 형태로만 반환하며, think 태그나 부가 설명은 금지.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_system_prompt(self) -> str:
        # JSON 스키마 예시: thumbnail_details + 4개의 panel_details
        schema_example = {
            "thumbnail_details": {
                "title": "<string>",                 # 한글 또는 영어 제목(사용자 화면용)
                "location_and_time": "<string>",
                "characters": "<string>",            # 등장인물 최대 3명, 동일 캐릭터는 일관된 묘사
                "camera_angle": "<string>",
                "main_action": "<string>",
                "mood_and_lighting": "<string>",
                "props_and_effects": "<string>",
                "descriptive_clause": "<string>"
            },
            "panel_details": [
                {
                    "scene_id": "P1",
                    "panel_brief": "<string>",       # 한 줄 설명(한글)
                    "location_and_time": "<string>",
                    "characters": "<string>",  
                    "camera_angle": "<string>",
                    "main_action": "<string>",
                    "mood_and_lighting": "<string>",
                    "props_and_effects": "<string>",
                    "dialogue_ko": "<string or '—'>",
                    "descriptive_clause": "<string>"
                },
                # P2, P3, P4 동일 구조
                {"scene_id": "P2", "panel_brief": "<string>", "location_and_time": "<string>",
                 "characters": "<string>", "camera_angle": "<string>", "main_action": "<string>",
                 "mood_and_lighting": "<string>", "props_and_effects": "<string>",
                 "dialogue_ko": "<string or '—'>", "descriptive_clause": "<string>"},
                {"scene_id": "P3", "panel_brief": "<string>", "location_and_time": "<string>",
                 "characters": "<string>", "camera_angle": "<string>", "main_action": "<string>",
                 "mood_and_lighting": "<string>", "props_and_effects": "<string>",
                 "dialogue_ko": "<string or '—'>", "descriptive_clause": "<string>"},
                {"scene_id": "P4", "panel_brief": "<string>", "location_and_time": "<string>",
                 "characters": "<string>", "camera_angle": "<string>", "main_action": "<string>",
                 "mood_and_lighting": "<string>", "props_and_effects": "<string>",
                 "dialogue_ko": "<string or '—'>", "descriptive_clause": "<string>"}
            ]
        }
        json_schema = json.dumps(schema_example, ensure_ascii=False, indent=2)

        return (
            "You are an expert webtoon scenario writer and AI image prompt engineer.\n"
            "아래는 ‘단순화된 시나리오(simple_scenario)’를 바탕으로, Flux Dev용 image prompt를 생성하기 전 단계로,\n"
            "‘썸네일 이미지에 들어갈 상세 요소(thumbnail_details)’와 ‘4컷 각각의 시각적 상세 요소(panel_details)’를 작성해야 합니다.\n"
            "제약조건:\n"
            "1. 동일 캐릭터는 모든 패널에서 외형 묘사를 반드시 일관되게 유지하십시오.\n"
            "2. 장면 간 흐름이 자연스럽고 논리적으로 이어지도록 하십시오.\n"
            "3. 등장인물은 총 3명 이내로 제한하고, 관중처럼 사람을 너무 많이 그리는 것은 피하십시오.\n"
            "4. 현재는 LoRA가 적용되지 않은 상태이므로, 스타일이나 톤이 과도하게 변하지 않도록 일관성을 유지하십시오.\n"
            "5. 장르에 맞는 어휘를 선택하여 작성하되, 과장된 표현은 지양하십시오.\n"
            "6. 출력은 JSON 형태로만 반환하며, think 태그나 부가 설명은 절대 포함하지 마십시오.\n"
            "\n"
            "## 출력 스키마 (JSON only)\n"
            f"{json_schema}"
        )

    def _build_user_prompt(self, simple: Dict[str, Any]) -> str:
        # simple_scenario를 JSON 문자열로 그대로 전달
        return f"""
# Input Simple Scenario (한글)
{json.dumps(simple, ensure_ascii=False)}
"""

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta = state.meta
        scenario_sec = state.scenario

        # 1) simple_scenario 유효성 검사
        simple = scenario_sec.simple_scenario
        if not simple or not isinstance(simple, dict):
            logger.error("n08: simple_scenario가 없습니다.", extra={"trace_id": meta.trace_id})
            meta.current_stage = "ERROR"
            meta.error_message = "Missing simple_scenario in state.scenario"
            return {"scenario": scenario_sec.model_dump(), "meta": meta.model_dump()}

        # 2) 시스템/사용자 프롬프트 생성
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(simple)

        # 3) LLM 호출 (qwen3-awq-14b) + 재시도 로직
        max_retries = 2
        parsed: Dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            logger.info(f"n08: 상세 시나리오 생성 시도 {attempt+1}/{max_retries+1}", extra={"trace_id": meta.trace_id})
            llm_resp = await self.llm_service.generate_text(
                system_prompt_content=system_prompt,
                prompt=user_prompt,
                max_tokens=3500,
                temperature=0.7
            )
            raw = llm_resp.get("generated_text", "")
            logger.debug(f"n08: Raw scenario JSON: {raw}", extra={"trace_id": meta.trace_id})
            if not raw or "error" in llm_resp:
                logger.warning(f"n08: LLM 응답 오류 또는 빈값, 재시도 {attempt+1}", extra={"trace_id": meta.trace_id})
                continue

            # 4) JSON 파싱
            try:
                parsed = extract_json(raw)
            except Exception as e:
                logger.error(f"n08: JSON 파싱 실패: {e}", extra={"trace_id": meta.trace_id})
                continue

            # 5) 구조 검증: thumbnail_details가 dict, panel_details가 길이 4인 리스트
            thumbnail_details = parsed.get("thumbnail_details")
            panel_details = parsed.get("panel_details")
            if not isinstance(thumbnail_details, dict):
                logger.warning(f"n08: thumbnail_details가 유효하지 않음, 재시도 {attempt+1}", extra={"trace_id": meta.trace_id})
                continue
            if not isinstance(panel_details, list) or len(panel_details) != 4:
                logger.warning(f"n08: panel_details 개수 잘못됨 ({len(panel_details)}), 재시도 {attempt+1}", extra={"trace_id": meta.trace_id})
                continue

            # 6) 각 panel_details 내부 필드 검증 (필드가 모두 존재하는지 최소 확인)
            required_panel_keys = {
                "scene_id", "panel_brief", "location_and_time", "characters",
                "camera_angle", "main_action", "mood_and_lighting",
                "props_and_effects", "dialogue_ko", "descriptive_clause"
            }
            ok = True
            for pd in panel_details:
                if not required_panel_keys.issubset(set(pd.keys())):
                    logger.warning(
                        f"n08: panel_details에 필드 누락 {set(pd.keys())}, 재시도 {attempt+1}",
                        extra={"trace_id": meta.trace_id}
                    )
                    ok = False
                    break
            if not ok:
                continue

            # 성공
            break
        else:
            # 모든 재시도 실패
            logger.error("n08: 상세 시나리오 생성 실패", extra={"trace_id": meta.trace_id})
            meta.current_stage = "ERROR"
            meta.error_message = "Scenario generation failed after retries"
            return {"scenario": scenario_sec.model_dump(), "meta": meta.model_dump()}

        # 7) state.scenario에 결과 저장
        scenario_sec.thumbnail_details = parsed["thumbnail_details"]
        scenario_sec.panel_details = parsed["panel_details"]

        # 8) 다음 단계로 이동
        meta.current_stage = "n08a_image_prompt_refinement"
        logger.info("n08 완료: thumbnail_details 및 panel_details 저장, n08a로 이동", extra={"trace_id": meta.trace_id})

        # 9) Pydantic 필드 덮어쓰기
        state.scenario = scenario_sec
        state.meta = meta

        return {
            "scenario": scenario_sec.model_dump(),
            "meta": meta.model_dump()
        }