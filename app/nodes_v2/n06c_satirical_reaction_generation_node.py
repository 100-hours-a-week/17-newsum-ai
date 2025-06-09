# ai/app/nodes_v2/n06c_satirical_reaction_generation_node.py
from __future__ import annotations
import json
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.workflows.state_v2 import WorkflowState, IdeaSection
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)
NODE_ORDER = 9  # N06B 다음 순서 (예시)


class N06CSatiricalReactionGenerationNode:
    """
    (N06C - 신규) N06의 이슈 분석과 N06B의 실제 커뮤니티 반응 분석 결과를 종합하여,
    각 플랫폼 스타일에 맞는 최종 풍자 반응을 LLM으로 생성합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_final_satire_prompt(self, platform_name: str,
                                   structured_issue_analysis: Dict[str, Any],
                                   platform_reaction_analysis: Optional[Dict[str, Any]]) -> str:
        """최종 풍자 반응 생성을 위한 LLM 프롬프트 (영문)"""

        issue_summary = (
            f"Core Problem: {structured_issue_analysis.get('core_problem_definition', 'N/A')}. "
            f"Key Satire Points from Report: {'; '.join(structured_issue_analysis.get('potential_satire_points', ['N/A']))}."
        )

        actual_reaction_summary = "No specific analysis of actual user reactions available for this platform."
        if platform_reaction_analysis:
            style_sum = platform_reaction_analysis.get('platform_style_summary_korean', 'N/A')
            key_args = "; ".join(platform_reaction_analysis.get('key_arguments_observed_korean', ['N/A']))
            add_satire = "; ".join(
                platform_reaction_analysis.get('additional_satire_points_from_reactions_korean', ['N/A']))
            actual_reaction_summary = (
                f"Analysis of actual user reactions on '{platform_name}' for this issue indicates:\n"
                f"- Style Summary (Korean): {style_sum}\n"
                f"- Key Arguments Observed (Korean): {key_args}\n"
                f"- Additional Satire Points from Reactions (Korean): {add_satire}"
            )

        # LLM이 반환할 JSON 스키마 (반응 타입과 내용)
        reaction_schema = {
            "reactions": [
                {"type": "<e.g., comment, post_title, tweet, image_idea_description>",
                 "content_korean": "<Satirical content in Korean, fitting the platform style and leveraging insights from actual user reactions.>"},
                # ... (2-3 more reactions)
            ]
        }
        json_schema_str = json.dumps(reaction_schema, indent=2, ensure_ascii=False)

        common_instruction = f"""
You are a highly creative satirist tasked with generating initial satirical reactions for an online platform.
Your reactions must be in KOREAN and tailored to the style of '{platform_name}'.

# Issue Context (from report analysis):
{issue_summary}

# Insights from Actual User Reactions on '{platform_name}' (use this to refine your satire):
{actual_reaction_summary}

# Task:
Based on BOTH the issue context AND the insights from actual user reactions (if available), generate 2-3 distinct, short, and impactful satirical reactions.
These reactions should reflect the analyzed style, tone, and common themes observed from real users on '{platform_name}'.
Be creative and ensure the satire is sharp and relevant.

# Output Format:
Return ONLY a single, valid JSON object that strictly adheres to the schema below.
The 'type' field should categorize the reaction (e.g., "comment", "sarcastic_question", "tweet_idea", "image_concept_text").
The 'content_korean' field must be your satirical KOREAN text.
Do NOT include any extraneous text, <think> tags, or comments outside this JSON structure.

# JSON Schema:
{json_schema_str}
"""
        # 플랫폼별 특화 지침은 이전 N06의 _build_community_reaction_prompt와 유사하게 추가 가능 (선택적)
        # 예: if platform_name.lower() == "dcinside": common_instruction += "\n# DCInside Specifics: Embrace bluntness..."
        return common_instruction

    async def _generate_final_platform_reactions(self, platform_name: str,
                                                 structured_issue_analysis: Dict[str, Any],
                                                 platform_reaction_analysis: Optional[Dict[str, Any]],
                                                 work_id: str, parent_node_name: str, extra_log_base: dict
                                                 ) -> Tuple[str, List[Dict[str, str]], Optional[str]]:
        """단일 플랫폼에 대한 최종 풍자 반응을 생성합니다."""
        request_id = f"{work_id}_{parent_node_name}_FinalSatire_{platform_name}"
        prompt = self._build_final_satire_prompt(platform_name, structured_issue_analysis, platform_reaction_analysis)

        llm_response = await self.llm_service.generate_text(
            messages=[{"role": "system", "content": f"You are generating satirical content for {platform_name}."},
                      {"role": "user", "content": prompt}],
            request_id=request_id, temperature=0.8, max_tokens=700  # 창의성 및 다양한 반응 유형 위해 온도 약간 높임
        )
        cleaned_output = llm_response.get("generated_text", "")
        think_content = llm_response.get("think_content")
        generated_reactions: List[Dict[str, str]] = []

        if cleaned_output and not llm_response.get("error"):
            try:
                parsed_json = extract_json(cleaned_output)
                if isinstance(parsed_json, dict) and "reactions" in parsed_json and isinstance(parsed_json["reactions"],
                                                                                               list):
                    # 각 reaction이 {"type": "...", "content_korean": "..."} 형태인지 추가 검증 가능
                    generated_reactions = [r for r in parsed_json["reactions"] if
                                           isinstance(r, dict) and "content_korean" in r]
                else:
                    logger.warning(f"플랫폼 '{platform_name}' 최종 풍자 반응 JSON 형식이 스키마와 다름: {cleaned_output}",
                                   extra=extra_log_base)
            except Exception as e:
                logger.error(f"플랫폼 '{platform_name}' 최종 풍자 반응 JSON 파싱 오류: {e}. 원본: {cleaned_output}",
                             extra=extra_log_base)
        else:
            logger.error(f"플랫폼 '{platform_name}' 최종 풍자 반응 생성 LLM 호출 실패: {llm_response.get('error')}",
                         extra=extra_log_base)

        return platform_name, generated_reactions, think_content

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        idea_sec = state.idea

        work_id = meta_sec.work_id
        node_name = self.__class__.__name__
        extra_log_base = {"work_id": work_id, "node_name": node_name, "node_order": NODE_ORDER}

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info(f"{node_name} 진입: 최종 풍자 반응 생성 시작.", extra=extra_log_base)

        structured_analysis = idea_sec.structured_issue_analysis
        community_analysis = idea_sec.community_reaction_analysis

        if not structured_analysis:
            logger.error("풍자 반응 생성을 위한 선행 이슈 분석 결과(N06)가 없습니다.", extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            idea_sec.generated_satirical_reactions = {}
            return {"meta": meta_sec.model_dump(), "idea": idea_sec.model_dump(exclude_unset=True)}

        # community_analysis는 Optional이므로 없을 수도 있음. 프롬프트에서 처리.
        if not community_analysis:
            logger.warning("선행 커뮤니티 실제 반응 분석 결과(N06B)가 없습니다. 보고서 분석 기반으로만 풍자 반응을 생성합니다.", extra=extra_log_base)
            community_analysis = {}  # 빈 딕셔너리로 처리하여 _build_final_satire_prompt에서 분기하도록 함

        platforms_to_generate_for = list(community_analysis.keys()) if community_analysis else ["dcinside", "reddit",
                                                                                                "x_twitter"]  # 분석된 플랫폼 또는 기본 플랫폼
        if not platforms_to_generate_for and idea_sec.scraped_community_reactions:  # 스크랩은 했으나 분석이 없는 경우
            platforms_to_generate_for = list(idea_sec.scraped_community_reactions.keys())

        final_satire_tasks = []
        for platform in platforms_to_generate_for:
            platform_specific_analysis = community_analysis.get(platform)  # 해당 플랫폼의 분석 결과 가져오기
            final_satire_tasks.append(
                self._generate_final_platform_reactions(
                    platform, structured_analysis, platform_specific_analysis, work_id, node_name, extra_log_base
                    # type: ignore
                )
            )

        final_generated_reactions: Dict[str, List[Dict[str, str]]] = {}
        if final_satire_tasks:
            logger.info(f"{len(final_satire_tasks)}개 플랫폼에 대한 최종 풍자 반응 생성 병렬 시작.", extra=extra_log_base)
            satire_results_tuples = await asyncio.gather(*final_satire_tasks, return_exceptions=True)

            for result_tuple_or_exc in satire_results_tuples:
                if isinstance(result_tuple_or_exc, Exception):
                    logger.error(f"최종 풍자 반응 생성 작업 중 예외: {result_tuple_or_exc}", extra=extra_log_base,
                                 exc_info=result_tuple_or_exc)
                    continue

                platform_name_res, reactions_data, think_content_platform = result_tuple_or_exc
                if think_content_platform:
                    meta_sec.llm_think_traces.append({
                        "node_name": f"{node_name}_GenerateSatire_{platform_name_res}",
                        "request_id": f"{work_id}_{node_name}_FinalSatire_{platform_name_res}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "log_content": think_content_platform
                    })
                if reactions_data:  # 유효한 반응 데이터만 저장
                    final_generated_reactions[platform_name_res] = reactions_data
        else:
            logger.info("생성할 최종 풍자 반응 작업이 없습니다 (플랫폼 목록 비어있음).", extra=extra_log_base)

        idea_sec.generated_satirical_reactions = final_generated_reactions
        meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"
        logger.info(f"{node_name} 완료: 최종 풍자 반응 생성 완료.", extra=extra_log_base)
        return {"meta": meta_sec.model_dump(), "idea": idea_sec.model_dump(exclude_unset=True)}