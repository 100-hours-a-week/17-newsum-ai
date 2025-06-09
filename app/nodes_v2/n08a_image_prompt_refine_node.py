# ai/app/nodes_v2/n08a_image_prompt_refine_node.py
from typing import Dict, Any, List
from datetime import datetime, timezone  # <think> 태그 저장 시 timestamp용

from app.workflows.state_v2 import WorkflowState, ImageSection  # ImageSection 임포트
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json
from app.utils.logger import get_logger  # summarize_for_logging 사용
from app.config.image_style_config import IMAGE_STYLE_CONFIGS, DEFAULT_IMAGE_MODE  # 설정값 임포트

logger = get_logger(__name__)

# 이 노드의 워크플로우 내 순서 정의
NODE_ORDER = 12  # N08 다음 순서 (예시)


class N08aImagePromptRefinementNode:
    """
    (업그레이드됨) N08에서 생성된 thumbnail_prompt와 각 패널의 final_image_prompt를
    1) 정규화(영어 통일, 일반화) -> 2) 이미지 생성 모드(Flux/XL 등)별로 재구성합니다.
    두 번의 LLM 호출로 처리하며, <think> 태그 내용을 별도 저장하고 state_v2.py에 맞춰 상태를 관리합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_system_prompt_normalization(self) -> str:
        # (기존 프롬프트 유지 - 영어로 잘 작성되어 있음)
        return """
You are an expert image-prompt normalizer.
- Standardize all input sentences (Korean/English) into English.
- Generalize brand names, specific proper nouns (unless crucial for a well-known parody), and ambiguous words to broader terms suitable for image generation.
- Ensure prompts are descriptive and clear for image models.
- Return ONLY a valid JSON object strictly matching the schema below. No extraneous text, thoughts, or comments.

Schema:
{
  "normalized_thumbnail_prompt": "<string: English, normalized prompt for thumbnail>",
  "normalized_panel_prompts": ["<string: English, normalized prompt for panel 1>", "<string: panel 2>", "<string: panel 3>", "<string: panel 4>"]
}
"""

    def _build_user_prompt_normalization(self, original_thumbnail_prompt: str,
                                         original_panel_prompts: List[str]) -> str:
        # (기존 프롬프트 유지 - 입력값을 명확히 전달)
        panels_text = "\n".join(
            f"- Panel {i + 1}: {p_text}" for i, p_text in enumerate(original_panel_prompts)
        )
        return f"""
# Original Thumbnail Prompt (from N08):
{original_thumbnail_prompt}

# Original Panel Prompts (final_image_prompt from N08 panels):
{panels_text}

Normalize these prompts according to the instructions and schema in the system message.
"""

    def _build_user_prompt_mode_specific_refinement(self, normalized_thumbnail: str, normalized_panels: List[str],
                                                    mode_specific_system_prompt: str) -> str:  # mode_specific_system_prompt는 이제 사용 안함 (LLMService에서 처리)
        # 이 프롬프트는 LLM에게 정규화된 프롬프트를 모드별로 재구성하도록 요청합니다.
        # 모드별 시스템 프롬프트는 IMAGE_STYLE_CONFIGS에서 가져와 LLMService 호출 시 전달합니다.
        panels_text = "\n".join(f"Panel {i + 1}: {p}" for i, p in enumerate(normalized_panels))
        return f"""
# Normalized Thumbnail Prompt (English):
{normalized_thumbnail}

# Normalized Panel Prompts (English):
{panels_text}

Based on the system prompt (which defines the target image style and output JSON schema for that style),
refine and restructure these normalized prompts.
Return ONLY the JSON object as specified by the target style's schema.
"""

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        scenario_sec = state.scenario  # N08 결과 읽기
        image_sec = state.image  # 여기에 결과 저장
        config_sec_dict = state.config.config or {}

        work_id = meta_sec.work_id
        node_name = self.__class__.__name__
        extra_log_base = {"work_id": work_id, "node_name": node_name, "node_order": NODE_ORDER}

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info(f"{node_name} 진입: N08 프롬프트 정제 시작.", extra=extra_log_base)

        # --- 1. 입력 프롬프트 준비 ---
        original_thumb_prompt = scenario_sec.thumbnail_image_prompt
        # comic_scenarios 각 패널에서 'final_image_prompt' 추출
        original_panels_prompts_list = [
            panel.get("final_image_prompt", "") for panel in (scenario_sec.comic_scenarios or [])
        ]

        if not original_thumb_prompt or not all(original_panels_prompts_list) or len(original_panels_prompts_list) != 4:
            msg = "정제를 위한 원본 썸네일 프롬프트 또는 4개의 패널 프롬프트가 누락되었거나 유효하지 않습니다."
            logger.error(msg, extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"image": image_sec.model_dump(exclude_unset=True), "meta": meta_sec.model_dump()}

        # === 1단계 LLM 호출: 프롬프트 정규화 (영어화, 일반화) ===
        try:
            logger.info("1단계 LLM 호출: 프롬프트 정규화 시작.", extra=extra_log_base)
            norm_sys_prompt = self._build_system_prompt_normalization()
            norm_user_prompt = self._build_user_prompt_normalization(original_thumb_prompt,
                                                                     original_panels_prompts_list)
            request_id_norm = f"{work_id}_{node_name}_Normalization"

            llm_response_norm = await self.llm_service.generate_text(
                messages=[{"role": "system", "content": norm_sys_prompt},
                          {"role": "user", "content": norm_user_prompt}],
                request_id=request_id_norm, temperature=0.2, max_tokens=1500
            )

            cleaned_norm_output = llm_response_norm.get("generated_text", "")
            think_content_norm = llm_response_norm.get("think_content")

            if think_content_norm:
                meta_sec.llm_think_traces.append({
                    "node_name": f"{node_name}_Normalization", "request_id": request_id_norm,
                    "timestamp": datetime.now(timezone.utc).isoformat(), "log_content": think_content_norm
                })

            if not cleaned_norm_output or llm_response_norm.get("error"):
                raise ValueError(f"정규화 LLM 실패: {llm_response_norm.get('error', 'Empty output')}")

            parsed_norm_json = extract_json(cleaned_norm_output)
            if not isinstance(parsed_norm_json, dict):
                raise ValueError(f"정규화 LLM 결과가 JSON 객체가 아님. 파싱된 내용: {parsed_norm_json}")

            normalized_thumbnail = parsed_norm_json.get("normalized_thumbnail_prompt", "").strip()
            normalized_panels = parsed_norm_json.get("normalized_panel_prompts", [])

            if not normalized_thumbnail or not isinstance(normalized_panels, list) or len(
                    normalized_panels) != 4 or not all(isinstance(p, str) and p.strip() for p in normalized_panels):
                raise ValueError(
                    f"정규화된 프롬프트 결과가 유효하지 않습니다. 썸네일: '{normalized_thumbnail}', 패널 수: {len(normalized_panels)}")
            logger.info("1단계 LLM 호출: 프롬프트 정규화 성공.", extra=extra_log_base)

        except Exception as e_norm:
            logger.error(f"프롬프트 정규화 단계(1차 LLM) 중 오류: {e_norm}", extra=extra_log_base, exc_info=True)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"image": image_sec.model_dump(exclude_unset=True), "meta": meta_sec.model_dump()}

        # === 2단계 LLM 호출: 이미지 생성 모드별 프롬프트 재구성 ===
        try:
            logger.info("2단계 LLM 호출: 모드별 프롬프트 재구성 시작.", extra=extra_log_base)
            # 설정에서 현재 이미지 모드 가져오기 (예: 'flux', 'sdxl_base')
            image_generation_mode = config_sec_dict.get("image_mode", DEFAULT_IMAGE_MODE).lower()
            style_config = IMAGE_STYLE_CONFIGS.get(image_generation_mode, IMAGE_STYLE_CONFIGS[DEFAULT_IMAGE_MODE])

            # 모드별 시스템 프롬프트 (IMAGE_STYLE_CONFIGS에서 정의된 함수를 통해 생성)
            # 이 시스템 프롬프트는 해당 모드의 특성, JSON 출력 스키마 등을 LLM에게 지시해야 함.
            mode_specific_system_prompt_builder = style_config.get("prompt_builder")
            if not callable(mode_specific_system_prompt_builder):
                raise ValueError(f"설정된 이미지 모드 '{image_generation_mode}'에 대한 유효한 'prompt_builder' 함수가 없습니다.")

            # prompt_builder는 보통 (thumbnail_prompt: str, panel_prompts: List[str]) 같은 인자를 받아
            # 해당 스타일의 JSON 스키마를 포함한 전체 시스템 프롬프트를 반환하도록 설계될 수 있음.
            # 여기서는 간단히 인자 없이 시스템 프롬프트 문자열만 반환한다고 가정.
            # 또는, 프롬프트 빌더가 (LLMService가 사용할) messages 목록 자체를 반환할 수도 있음.
            # 현재 N08a 원본 코드는 prompt_builder가 시스템 프롬프트 "문자열"을 반환한다고 가정.
            mode_system_prompt_str = mode_specific_system_prompt_builder()  # 예: FluxStylePromptBuilder().build_system_prompt_for_refinement()

            mode_user_prompt = self._build_user_prompt_mode_specific_refinement(normalized_thumbnail, normalized_panels,
                                                                                mode_system_prompt_str)
            request_id_mode = f"{work_id}_{node_name}_ModeRefine_{image_generation_mode}"

            llm_response_mode = await self.llm_service.generate_text(
                messages=[{"role": "system", "content": mode_system_prompt_str},
                          {"role": "user", "content": mode_user_prompt}],
                request_id=request_id_mode, temperature=0.5, max_tokens=2000  # 모드별 재구성은 약간의 창의성 필요
            )

            cleaned_mode_output = llm_response_mode.get("generated_text", "")
            think_content_mode = llm_response_mode.get("think_content")

            if think_content_mode:
                meta_sec.llm_think_traces.append({
                    "node_name": f"{node_name}_ModeRefine_{image_generation_mode}", "request_id": request_id_mode,
                    "timestamp": datetime.now(timezone.utc).isoformat(), "log_content": think_content_mode
                })

            if not cleaned_mode_output or llm_response_mode.get("error"):
                raise ValueError(f"모드별 프롬프트 재구성 LLM 실패: {llm_response_mode.get('error', 'Empty output')}")

            final_refined_prompts_json = extract_json(cleaned_mode_output)
            if not isinstance(final_refined_prompts_json, dict):
                raise ValueError(f"모드별 재구성 LLM 결과가 JSON 객체가 아님. 파싱된 내용: {final_refined_prompts_json}")

            logger.info(f"2단계 LLM 호출: 모드 '{image_generation_mode}' 프롬프트 재구성 성공.", extra=extra_log_base)

        except Exception as e_mode:
            logger.error(f"모드별 프롬프트 재구성 단계(2차 LLM) 중 오류: {e_mode}", extra=extra_log_base, exc_info=True)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"image": image_sec.model_dump(exclude_unset=True), "meta": meta_sec.model_dump()}

        # --- 3. 최종 프롬프트 ImageSection에 저장 ---
        #    IMAGE_STYLE_CONFIGS의 'type' (flux, xl 등)에 따라 결과 JSON 구조가 다를 수 있음.
        #    N08a 원본 코드의 저장 로직 참조.
        refined_prompt_entries: List[Dict[str, Any]] = []
        prompt_structure_type = style_config.get("type", "default")  # 예: "flux", "sdxl_basic"

        # 썸네일 프롬프트 처리
        thumb_prompt_final = ""
        if prompt_structure_type == "flux_style_A":  # 예시 타입, 실제 설정에 맞게
            thumb_prompt_final = final_refined_prompts_json.get("refined_thumbnail", {}).get("flux_prompt", "")
        elif prompt_structure_type == "sdxl_style_B":
            thumb_prompt_final = ", ".join(final_refined_prompts_json.get("sdxl_thumbnail_tokens", []))
        else:  # 기본 또는 일반적인 경우 (예: 스키마에 thumbnail_prompt 키가 바로 있는 경우)
            thumb_prompt_final = final_refined_prompts_json.get("thumbnail_prompt",
                                                                normalized_thumbnail)  # 폴백으로 정규화된 것 사용

        if not thumb_prompt_final: logger.warning("최종 썸네일 프롬프트가 비어있습니다.", extra=extra_log_base)

        refined_prompt_entries.append({
            "scene_identifier": "thumbnail",  # 고정 식별자
            "model_name": image_generation_mode,  # 실제 사용할 모델 또는 모드명
            "prompt_used": thumb_prompt_final.strip()
        })

        # 패널 프롬프트 처리
        # LLM이 반환하는 JSON 내 패널 프롬프트 키 (예: "refined_panel_prompts", "sdxl_panel_tokens_list")는
        # 모드별 시스템 프롬프트에서 정의된 스키마에 따라 달라짐.
        # 여기서는 일반적인 "panel_prompts" 리스트를 가정 (실제로는 모드별로 분기 필요)
        raw_panel_prompts_from_llm = final_refined_prompts_json.get("panel_prompts", [])  # 예시 키

        if not isinstance(raw_panel_prompts_from_llm, list) or len(raw_panel_prompts_from_llm) != 4:
            logger.warning(
                f"LLM으로부터 받은 최종 패널 프롬프트가 4개가 아니거나 리스트가 아닙니다 (개수: {len(raw_panel_prompts_from_llm)}). 정규화된 프롬프트를 대신 사용합니다.",
                extra=extra_log_base)
            # 폴백: 정규화된 패널 프롬프트 사용
            panel_prompts_to_save = normalized_panels
        else:
            # 모드별로 프롬프트 구조가 다를 수 있음 (예: 문자열 리스트, 또는 각 패널이 딕셔너리 등)
            # 여기서는 문자열 리스트를 가정
            if prompt_structure_type == "flux_style_A" and all(isinstance(p, dict) for p in raw_panel_prompts_from_llm):
                panel_prompts_to_save = [p.get("flux_prompt", "") for p in raw_panel_prompts_from_llm]
            elif prompt_structure_type == "sdxl_style_B" and all(
                    isinstance(p, list) for p in raw_panel_prompts_from_llm):  # 토큰 리스트의 리스트
                panel_prompts_to_save = [", ".join(tokens_list) for tokens_list in raw_panel_prompts_from_llm]
            elif all(isinstance(p, str) for p in raw_panel_prompts_from_llm):  # 단순 문자열 리스트
                panel_prompts_to_save = raw_panel_prompts_from_llm
            else:  # 예상치 못한 구조면 정규화된 프롬프트 사용
                logger.warning(f"최종 패널 프롬프트의 내부 구조가 예상과 다릅니다. 정규화된 프롬프트를 대신 사용합니다.", extra=extra_log_base)
                panel_prompts_to_save = normalized_panels

        # scenario_sec.comic_scenarios에서 scene_identifier 가져와 매칭
        for idx, panel_info_from_n08 in enumerate(scenario_sec.comic_scenarios or []):
            scene_id_from_n08 = panel_info_from_n08.get("scene_identifier", f"S01P{idx + 1}")  # N08의 식별자
            prompt_text_for_panel = panel_prompts_to_save[idx].strip() if idx < len(panel_prompts_to_save) else ""
            if not prompt_text_for_panel: logger.warning(f"패널 {scene_id_from_n08}의 최종 프롬프트가 비어있습니다.",
                                                         extra=extra_log_base)

            refined_prompt_entries.append({
                "scene_identifier": scene_id_from_n08,
                "model_name": image_generation_mode,
                "prompt_used": prompt_text_for_panel
            })

        image_sec.refined_prompts = refined_prompt_entries  # 상태에 최종 저장
        logger.info(f"{node_name} 완료: 총 {len(refined_prompt_entries)}개의 정제된 프롬프트 저장 완료 (썸네일 포함).", extra=extra_log_base)
        meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"

        return {"image": image_sec.model_dump(exclude_unset=True), "meta": meta_sec.model_dump()}