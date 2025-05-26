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

    def _build_system_prompt(
        self,
        idea: Dict[str, Any],
        summary: str,
        audience: str
    ) -> str:
        """
        시나리오 및 썸네일 프롬프트를 생성할 LLM system prompt 구성
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
                }
            ]
        }
        json_schema = json.dumps(schema_example, ensure_ascii=False, indent=2)

        title = idea.get("title", "Untitled Comic")
        logline = idea.get("logline") or idea.get("summary", "No summary provided.")
        genre = idea.get("genre", "general")

        return f"""
You are a professional webtoon scenario writer and AI image prompt engineer.
Generate both a thumbnail image prompt and a detailed 4-panel scenario for the given comic idea.
Write in clear Korean (한국어로 작성하세요).

# Comic Idea
Title: {title}
Logline: {logline}
Genre: {genre}
Target Audience: {audience}

# Context Summary
{summary}

# Output Format (Strict JSON only)
Return ONLY valid JSON matching this schema exactly, without explanations, comments, or markdown:
{json_schema}
"""  # noqa: E501

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

        # 3) System prompt 생성 & LLM 호출
        system_prompt = self._build_system_prompt(selected, summary, audience)
        llm_resp = await self.llm_service.generate_text(
            system_prompt_content=system_prompt,
            prompt="",
            max_tokens=3500,
            temperature=0.7
        )
        raw = llm_resp.get("generated_text", "")
        logger.debug(f"Raw scenario JSON: {raw}", extra={"trace_id": meta.trace_id})

        # 4) JSON 파싱
        try:
            parsed = extract_json(raw)
        except Exception as e:
            logger.error(f"JSON 파싱 실패: {e}", extra={"trace_id": meta.trace_id})
            meta.current_stage = "ERROR"
            meta.error_message = f"Scenario JSON parsing error: {e}"
            return {"scenario": scenario_sec.model_dump(), "meta": meta.model_dump()}

        # 5) State 업데이트
        thumbnail = parsed.get("thumbnail_prompt", "")
        panels = parsed.get("panels", [])

        scenario_sec.thumbnail_image_prompt = thumbnail
        scenario_sec.comic_scenarios = panels

        meta.current_stage = "n09_image_generation"
        meta.error_log = list(meta.error_log or [])

        logger.info(
            f"Scenario 생성 완료: {len(panels)} panels, thumbnail length {len(thumbnail)}", 
            extra={"trace_id": meta.trace_id}
        )
        return {"scenario": scenario_sec.model_dump(), "meta": meta.model_dump()}
