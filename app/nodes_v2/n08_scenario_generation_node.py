# ai/app/nodes_v2/n08_scenario_generation_node.py
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)

class N08ScenarioGenerationNode:
    """
    (업그레이드됨) 단일 LLM 호출로 썸네일 프롬프트와 4패널 웹툰 시나리오를
    JSON으로 생성하여 state.scenario에 저장하는 노드.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_system_prompt(self) -> str:
        """
        4개의 패널 예시를 모두 나열한 JSON 스키마를 시스템 프롬프트에 포함합니다.
        """
        schema_example = {
            "thumbnail_prompt": "<string>",
            "panels": [
                {
                    "panel_number": 1,
                    "scene_identifier": "S01P01",
                    "setting": "<string>",
                    "characters_present": "<string>",
                    "camera_shot_and_angle": "<string>",
                    "key_actions_or_events": "<string>",
                    "lighting_and_atmosphere": "<string>",
                    "dialogue_summary_for_image_context": "<string>",
                    "visual_effects_or_key_props": "<string>",
                    "image_style_notes_for_scene": "<string>",
                    "final_image_prompt": "<string>"
                },
                {
                    "panel_number": 2,
                    "scene_identifier": "S01P02",
                    "setting": "<string>",
                    "characters_present": "<string>",
                    "camera_shot_and_angle": "<string>",
                    "key_actions_or_events": "<string>",
                    "lighting_and_atmosphere": "<string>",
                    "dialogue_summary_for_image_context": "<string>",
                    "visual_effects_or_key_props": "<string>",
                    "image_style_notes_for_scene": "<string>",
                    "final_image_prompt": "<string>"
                },
                {
                    "panel_number": 3,
                    "scene_identifier": "S01P03",
                    "setting": "<string>",
                    "characters_present": "<string>",
                    "camera_shot_and_angle": "<string>",
                    "key_actions_or_events": "<string>",
                    "lighting_and_atmosphere": "<string>",
                    "dialogue_summary_for_image_context": "<string>",
                    "visual_effects_or_key_props": "<string>",
                    "image_style_notes_for_scene": "<string>",
                    "final_image_prompt": "<string>"
                },
                {
                    "panel_number": 4,
                    "scene_identifier": "S01P04",
                    "setting": "<string>",
                    "characters_present": "<string>",
                    "camera_shot_and_angle": "<string>",
                    "key_actions_or_events": "<string>",
                    "lighting_and_atmosphere": "<string>",
                    "dialogue_summary_for_image_context": "<string>",
                    "visual_effects_or_key_props": "<string>",
                    "image_style_notes_for_scene": "<string>",
                    "final_image_prompt": "<string>"
                }
            ]
        }
        json_schema = json.dumps(schema_example, ensure_ascii=False, indent=2)

        return (
            "You are a professional webtoon scenario writer and AI image prompt engineer.\n"
            "Generate both a thumbnail image prompt and exactly 4 detailed panels for the given comic idea.\n"
            "Write in clear Korean (한국어로 작성하세요).\n"
            "Return ONLY valid JSON matching the schema below, NO think tags, NO explanation, NO comments.\n"
            "# Output Format\n"
            f"{json_schema}"
        )


    def _build_user_prompt(self, idea: Dict[str, Any], summary: str, audience: str) -> str:
        """
        실제 입력 데이터(아이디어, summary, audience 등)는 user prompt에 넣습니다.
        """
        title = idea.get("title", "Untitled Comic")
        logline = idea.get("logline") or idea.get("summary", "No summary provided.")
        genre = idea.get("genre", "general")
        return (
            f"# Comic Idea\n"
            f"Title: {title}\n"
            f"Logline: {logline}\n"
            f"Genre: {genre}\n"
            f"Target Audience: {audience}\n\n"
            f"# Context Summary\n"
            f"{summary}"
        )

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta = state.meta
        scenario_sec = state.scenario
        idea_sec = state.idea
        report_sec = state.report

        # 1) 아이디어 선택
        selected = scenario_sec.selected_comic_idea_for_scenario or (idea_sec.comic_ideas or [None])[0]
        if not isinstance(selected, dict):
            logger.error("유효한 comic_idea가 없습니다.", extra={"trace_id": meta.trace_id})
            meta.current_stage = "ERROR"
            meta.error_message = "No comic idea provided for scenario generation"
            return {"scenario": scenario_sec.model_dump(), "meta": meta.model_dump()}

        # 2) 요약 또는 보고서 텍스트 준비
        summary = getattr(report_sec, "contextual_summary", None)
        if not summary:
            summary = re.sub(r'<[^>]+>', ' ', report_sec.report_content or '')
            summary = ' '.join(summary.split())[:4000]

        audience = state.config.config.get("target_audience", "general_public")

        # 3) system / user prompt 분리
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(selected, summary, audience)

        # 4) LLM 호출 + 재시도 로직 (파싱 후 패널 수 검증)
        max_retries = 2
        for attempt in range(max_retries + 1):
            logger.info(f"Generating scenario for {selected.get('title')}", extra={"trace_id": meta.trace_id})
            llm_resp = await self.llm_service.generate_text(
                system_prompt_content=system_prompt,
                prompt=user_prompt,
                max_tokens=3500,
                temperature=0.7
            )
            raw = llm_resp.get("generated_text", "")
            logger.debug(f"Raw scenario JSON: {raw}", extra={"trace_id": meta.trace_id})
            if not raw or "error" in llm_resp:
                logger.warning(f"LLM 응답 실패, 재시도 {attempt+1}/{max_retries+1}", extra={"trace_id": meta.trace_id})
                continue

            # JSON 파싱
            try:
                parsed = extract_json(raw)
            except Exception as e:
                logger.error(f"JSON 파싱 실패: {e}", extra={"trace_id": meta.trace_id})
                continue

            # 패널 수 검증: 반드시 4개여야 함
            panels = parsed.get("panels", [])
            if not isinstance(panels, list) or len(panels) != 4:
                logger.warning(
                    f"잘못된 패널 수({len(panels)}) – 정확히 4개의 패널 필요, 재시도 {attempt+1}/{max_retries+1}",
                    extra={"trace_id": meta.trace_id}
                )
                continue

            # 썸네일도 반드시 문자열이어야 함
            thumbnail = parsed.get("thumbnail_prompt")
            if not isinstance(thumbnail, str) or not thumbnail.strip():
                logger.warning(
                    "썸네일 프롬프트가 비어있거나 유효하지 않음 – 재시도 "
                    f"{attempt+1}/{max_retries+1}", extra={"trace_id": meta.trace_id}
                )
                continue

            # 성공
            break
        else:
            # 모든 시도 실패 시 에러
            meta.current_stage = "ERROR"
            meta.error_message = "Scenario generation failed after retries (invalid structure)"
            return {"scenario": scenario_sec.model_dump(), "meta": meta.model_dump()}

        # 5) State 업데이트
        scenario_sec.thumbnail_image_prompt = thumbnail
        scenario_sec.comic_scenarios = panels

        meta.current_stage = "n09_image_generation"
        meta.error_log = list(meta.error_log or [])
        logger.info(
            f"Scenario 생성 완료: {len(panels)} panels, thumbnail length {len(thumbnail)}",
            extra={"trace_id": meta.trace_id}
        )
        return {"scenario": scenario_sec.model_dump(), "meta": meta.model_dump()}
