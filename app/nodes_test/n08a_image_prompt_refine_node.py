# ai/app/nodes_v2/n08a_image_prompt_refinement_node.py

import json
from typing import Any, Dict, List
from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)

class N08aImagePromptRefinementNode:
    """
    n08에서 생성된 thumbnail_details와 panel_details를 읽어,
    Flux Dev 스타일(예: 나무·빈티지카 예시)을 참고하여
    qwen3-awq로부터 완성된 단일 문장 프롬프트를 생성합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_system_prompt(self) -> str:
        # Flux Dev 스타일 예시를 간략히 포함
        return (
            "You are an expert Flux Dev–style image-prompt engineer.\n"
            "Create one English sentence for the thumbnail and one for each of 4 panels using the given details.\n"
            "Requirements:\n"
            "1. Composition: Specify where main subjects appear (foreground, midground, background).\n"
            "2. Details: Use color, lighting, texture, and props. Avoid too many people (max 3).\n"
            "3. Fluent sentences: Multiple short English sentences, including 'with ...' clauses.\n"
            "4. Maintain consistency of character appearance and scene flow.\n"
            "5. Do not over-stylize; no LoRA applied. Output only JSON.\n"
            "\n"
            "Example Styles:\n"
            "- \"A single tree standing in the middle of the image. The left half has bright green leaves under a sunny blue sky, "
            "while the right half has bare branches covered in frost under a dark, thunderous sky. On the left is lush grass, on the right is thick snow. "
            "The split runs exactly down the middle of the tree.\"\n"
            "- \"In the foreground, a vintage car with a 'CLASSIC' plate is parked on a cobblestone street. Behind it is a bustling market with colorful awnings. "
            "In the background, the silhouette of an old castle on a hill is shrouded in mist.\"\n"
            "\n"
            "Output schema (JSON only):\n"
            "{\n"
            '  "thumbnail_prompt": "<Flux Dev–friendly English sentence>",\n'
            '  "panel_prompts": [\n'
            '    "<English sentence for P1>",\n'
            '    "<English sentence for P2>",\n'
            '    "<English sentence for P3>",\n'
            '    "<English sentence for P4>"\n'
            "  ]\n"
            "}"
        )

    def _build_user_prompt(self, thumbnail_details: Dict[str, Any], panel_details: List[Dict[str, Any]]) -> str:
        payload = {
            "thumbnail_details": thumbnail_details,
            "panel_details": panel_details
        }
        return f"""
# Input from n08 (JSON):
{json.dumps(payload, ensure_ascii=False)}
"""

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta = state.meta
        scenario_sec = state.scenario
        image_sec = state.image

        thumbnail_details = scenario_sec.thumbnail_details
        panel_details = scenario_sec.panel_details
        if not thumbnail_details or not isinstance(panel_details, list) or len(panel_details) != 4:
            logger.error("n08a: Invalid thumbnail_details or panel_details", extra={"trace_id": meta.trace_id})
            meta.current_stage = "ERROR"
            meta.error_message = "Missing or invalid details from n08"
            return {"image": image_sec.model_dump(), "meta": meta.model_dump()}

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(thumbnail_details, panel_details)

        max_retries = 2
        parsed: Dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            logger.info(f"n08a: Prompt refinement attempt {attempt+1}", extra={"trace_id": meta.trace_id})
            llm_resp = await self.llm_service.generate_text(
                system_prompt_content=system_prompt,
                prompt=user_prompt,
                max_tokens=1500,
                temperature=0.7
            )
            raw = llm_resp.get("generated_text", "")
            logger.debug(f"n08a: Raw prompt refinement JSON: {raw}", extra={"trace_id": meta.trace_id})
            if not raw or "error" in llm_resp:
                logger.warning(f"n08a: Empty or error response, retry {attempt+1}", extra={"trace_id": meta.trace_id})
                continue

            try:
                parsed = extract_json(raw)
            except Exception as e:
                logger.error(f"n08a: JSON parse failed: {e}", extra={"trace_id": meta.trace_id})
                continue

            tp = parsed.get("thumbnail_prompt")
            pp = parsed.get("panel_prompts")
            if not isinstance(tp, str) or not tp.strip():
                logger.warning(f"n08a: Invalid thumbnail_prompt, retry {attempt+1}", extra={"trace_id": meta.trace_id})
                continue
            if not isinstance(pp, list) or len(pp) != 4 or any(not isinstance(s, str) or not s.strip() for s in pp):
                logger.warning(f"n08a: Invalid panel_prompts list, retry {attempt+1}", extra={"trace_id": meta.trace_id})
                continue

            break
        else:
            logger.error("n08a: Failed to refine prompts", extra={"trace_id": meta.trace_id})
            meta.current_stage = "ERROR"
            meta.error_message = "Prompt refinement failed"
            return {"image": image_sec.model_dump(), "meta": meta.model_dump()}

        # config에서 image_mode를 읽어옴 (없으면 DEFAULT_IMAGE_MODE)
        model_name = getattr(state.config, 'config', {}).get('image_mode', 'flux')
        refined_list: List[Dict[str, Any]] = [{
            "scene_identifier": "thumbnail",
            "prompt_used": tp,
            "model_name": model_name
        }]
        for idx, sentence in enumerate(pp, start=1):
            refined_list.append({
                "scene_identifier": f"P{idx}",
                "prompt_used": sentence,
                "model_name": model_name
            })
        image_sec.refined_prompts = refined_list
        meta.current_stage = "n09_image_generation"
        logger.info("n08a complete: refined_prompts saved, moving to n09", extra={"trace_id": meta.trace_id})

        state.image = image_sec
        state.meta = meta
        return {"image": image_sec.model_dump(), "meta": meta.model_dump()}