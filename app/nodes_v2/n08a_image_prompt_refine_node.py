# ai/app/nodes_v2/n08a_image_prompt_refinement_node.py

import json
from typing import Dict, Any, List

from app.workflows.state_v2 import WorkflowState
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json
from app.utils.logger import get_logger
from app.config.image_style_config import IMAGE_STYLE_CONFIGS, DEFAULT_IMAGE_MODE

logger = get_logger(__name__)

class N08aImagePromptRefinementNode:
    """
    n08 에서 생성된 thumbnail_prompt, panels 시나리오를
    1) 정규화·표준화 → 2) Flux/XL 모드별로 재구성
    두 번의 LLM 호출로 처리하여 ImageSection.generated_comic_images에 저장합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_system_prompt_normalization(self) -> str:
        return """
You are an expert image-prompt normalizer.
- 입력된 한국어/영어 문장을 모두 영어로 통일하고
- 브랜드명, 고유명사, 모호한 단어 등을 일반화(normalize)하세요.
- 결과는 JSON으로, 아래 스키마만 반환합니다.

Schema:
{
  "thumbnail_prompt": "<string>",
  "panels": ["<string>", "<string>", "<string>", "<string>"]
}
"""

    def _build_user_prompt_normalization(self, thumbnail: str, panels: List[Dict[str, Any]]) -> str:
        panels_text = "\n".join(
            f"- {p.get('scene_identifier')}: {p.get('final_image_prompt','')}"
            for p in panels
        )
        return f"""
Return only JSON, NO thinks, NO comments, NO explanations.
# Original Thumbnail Prompt
{thumbnail}

# Original Panel Prompts
{panels_text}
"""

    def _build_user_prompt_mode(self, thumbnail: str, panels: List[str]) -> str:
        panels_text = "\n".join(f"{i+1}. {p}" for i, p in enumerate(panels))
        return f"""
Return only JSON, NO thinks, NO comments, NO explanations.
# Normalized Thumbnail
{thumbnail}

# Normalized Panels
{panels_text}
"""

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta = state.meta
        scenario_sec = state.scenario
        image_sec = state.image
        config = state.config.config or {}
        trace_id = meta.trace_id

        # 1) 정규화 호출
        logger.info(f"[n08a] 정규화 LLM 호출 시작", extra={"trace_id": trace_id})
        sys_norm = self._build_system_prompt_normalization()
        user_norm = self._build_user_prompt_normalization(
            scenario_sec.thumbnail_image_prompt or "",
            scenario_sec.comic_scenarios or []
        )
        # messages 리스트 생성
        messages_for_llm = [
            {"role": "system", "content": sys_norm},  # prompt를 system 역할로
            {"role": "user", "content": user_norm}  # original_query를 user 역할로
        ]

        # 수정된 방식으로 LLMService 호출
        resp_norm = await self.llm_service.generate_text(
            messages=messages_for_llm,
            temperature=0.3,
            max_tokens=1500
        )
        raw_norm = resp_norm.get("generated_text", "")
        logger.debug(f"[n08a] 정규화 LLM 응답: {raw_norm}", extra={"trace_id": trace_id})
        try:
            norm = extract_json(raw_norm)
            logger.info(f"[n08a] 정규화 JSON 파싱 성공", extra={"trace_id": trace_id})
        except Exception as e:
            logger.error(f"[n08a] 정규화 JSON 파싱 실패: {e}", extra={"trace_id": trace_id})
            meta.current_stage = "ERROR"
            meta.error_message = f"n08a normalization JSON error: {e}"
            return {"image": image_sec.model_dump(), "meta": meta.model_dump()}

        thumbnail_norm = norm.get("thumbnail_prompt", "").strip()
        panels_norm = norm.get("panels", [])
        logger.debug(f"[n08a] 정규화 결과 thumbnail: {thumbnail_norm}, panels: {panels_norm}", extra={"trace_id": trace_id})

        # 2) 모드별 프롬프트 빌더 호출 (image_style_config에서 관리)
        model_name = config.get("image_mode", DEFAULT_IMAGE_MODE).lower()
        style_conf = IMAGE_STYLE_CONFIGS.get(model_name, IMAGE_STYLE_CONFIGS[DEFAULT_IMAGE_MODE])
        prompt_builder = style_conf["prompt_builder"]
        sys_mode = prompt_builder()  # 실제 인자 필요시 맞게 전달

        user_mode = self._build_user_prompt_mode(thumbnail_norm, panels_norm)

        # messages 리스트 생성
        messages_for_llm = [
            {"role": "system", "content": sys_mode},  # prompt를 system 역할로
            {"role": "user", "content": user_mode}  # original_query를 user 역할로
        ]

        # 수정된 방식으로 LLMService 호출
        resp_mode = await self.llm_service.generate_text(
            messages=messages_for_llm,
            temperature=0.7,
            max_tokens=1500
        )
        raw_mode = resp_mode.get("generated_text", "")
        logger.debug(f"[n08a] 모드별 LLM 응답: {raw_mode}", extra={"trace_id": trace_id})
        try:
            final = extract_json(raw_mode)
            logger.info(f"[n08a] 모드별 JSON 파싱 성공", extra={"trace_id": trace_id})
        except Exception as e:
            logger.error(f"[n08a] 모드별 JSON 파싱 실패: {e}", extra={"trace_id": trace_id})
            meta.current_stage = "ERROR"
            meta.error_message = f"n08a mode-specific JSON error: {e}"
            return {"image": image_sec.model_dump(), "meta": meta.model_dump()}

        # 3) ImageSection 에 최종 프롬프트 저장
        entries: List[Dict[str, Any]] = []

        # 스타일 타입에 따라 key 결정 (확장성)
        prompt_type = style_conf.get("type", "flux")
        if prompt_type == "flux":
            thumb_prompt = final.get("thumbnail_prompt", "").strip()
            panel_prompts = final.get("panel_prompts", [])
        elif prompt_type == "xl":
            thumb_prompt = ", ".join(final.get("thumbnail_tokens", []))
            panel_prompts = [", ".join(tokens) for tokens in final.get("panel_tokens", [])]
        else:
            # 기본 fallback (확장성)
            thumb_prompt = final.get("thumbnail_prompt", "").strip()
            panel_prompts = final.get("panel_prompts", [])

        entries.append({
            "scene_identifier": "thumbnail",
            "model_name": model_name,
            "prompt_used": thumb_prompt
        })

        for idx, panel in enumerate(scenario_sec.comic_scenarios or []):
            scene_id = panel.get("scene_identifier", f"S0{idx+1}")
            prompt_text = panel_prompts[idx] if idx < len(panel_prompts) else ""
            entries.append({
                "scene_identifier": scene_id,
                "model_name": model_name,
                "prompt_used": prompt_text
            })

        image_sec.refined_prompts = entries
        logger.info(f"[n08a] refined_prompts 저장 완료: {len(entries)}개", extra={"trace_id": trace_id})
        meta.current_stage = "n09_image_generation"
        return {"image": image_sec.model_dump(), "meta": meta.model_dump()}
