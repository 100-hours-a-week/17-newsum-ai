# ai/app/nodes_v2/n06b_community_reaction_analysis_node.py
from __future__ import annotations
import json, asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json

logger = get_logger(__name__)
NODE_ORDER = 8  # N06A 다음 순서 (예시)


class N06BCommunityReactionAnalysisNode:
    """
    (N06B - 신규) N06A에서 수집된 플랫폼별 실제 사용자 반응을 LLM으로 분석하여,
    각 플랫폼의 반응 스타일, 주요 논점, 감정, 추가 풍자 포인트 등을 도출합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _build_reaction_analysis_prompt(self, platform_name: str, actual_reactions: List[str],
                                        issue_core_problem: str) -> str:
        """수집된 실제 반응 분석을 위한 LLM 프롬프트 (영문)"""
        reactions_str = "\n".join([f"- {r[:200]}" for r in actual_reactions[:15]])  # 최대 15개 반응, 각 200자 제한

        analysis_schema = {
            "platform_style_summary_korean": "<한글 요약: 이 플랫폼 사용자들의 현재 이슈에 대한 전반적인 반응 스타일 (예: 냉소적, 비판적, 유머러스 등) 및 주로 사용하는 어투나 표현 특징>",
            "key_arguments_observed_korean": ["<한글 요약: 관찰된 주요 주장 또는 논점 1>", "<한글 요약: 관찰된 주요 주장 또는 논점 2>", "..."],
            "dominant_emotions_observed_korean": ["<한글 감정1 (예: 분노)>", "<한글 감정2 (예: 허탈함)>", "..."],
            "additional_satire_points_from_reactions_korean": ["<한글 설명: 실제 반응에서 발견된 추가적인 풍자 포인트나 아이디어 1>", "..."]
        }
        json_schema_str = json.dumps(analysis_schema, indent=2)

        return f"""
You are an expert in analyzing online community discussions and identifying socio-cultural trends and satirical elements.
The core issue being discussed is: "{issue_core_problem}"

# Task:
Analyze the following actual user reactions collected from the online platform '{platform_name}'. Based on these reactions, provide a structured analysis in KOREAN by identifying:
1.  `platform_style_summary_korean`: A summary of the overall reaction style, tone, and linguistic features observed for this platform regarding the issue.
2.  `key_arguments_observed_korean`: List the main arguments or points being made by users on this platform.
3.  `dominant_emotions_observed_korean`: List the dominant emotions expressed in these reactions.
4.  `additional_satire_points_from_reactions_korean`: Identify any new or reinforced points for satire that emerge from these actual user reactions.

# Collected User Reactions from '{platform_name}':
---
{reactions_str}
---
(Note: Reactions might be truncated and are a sample.)

# Output Format:
Return ONLY a single, valid JSON object that strictly adheres to the schema below. Do NOT include any extraneous text, thoughts (e.g., <think> tags), or comments outside this JSON structure.

# JSON Schema:
{json_schema_str}
"""

    async def _analyze_single_platform(self, platform_name: str, reactions: List[str], issue_core_problem: str,
                                       work_id: str, parent_node_name: str, extra_log_base: dict) -> Tuple[
        str, Optional[Dict[str, Any]], Optional[str]]:
        """단일 플랫폼의 수집된 반응을 분석합니다."""
        if not reactions:
            logger.info(f"플랫폼 '{platform_name}'에 분석할 실제 반응이 없습니다.", extra=extra_log_base)
            return platform_name, None, None  # 분석할 내용 없으면 None 반환

        request_id = f"{work_id}_{parent_node_name}_AnalyzeReaction_{platform_name}"
        prompt = self._build_reaction_analysis_prompt(platform_name, reactions, issue_core_problem)

        llm_response = await self.llm_service.generate_text(
            messages=[{"role": "system", "content": f"You are analyzing user reactions from {platform_name}."},
                      {"role": "user", "content": prompt}],
            request_id=request_id, temperature=0.2, max_tokens=1500
        )
        cleaned_output = llm_response.get("generated_text", "")
        think_content = llm_response.get("think_content")
        analysis_result: Optional[Dict[str, Any]] = None

        if cleaned_output and not llm_response.get("error"):
            try:
                parsed_json = extract_json(cleaned_output)
                if isinstance(parsed_json, dict):
                    analysis_result = parsed_json
                else:
                    logger.warning(f"플랫폼 '{platform_name}' 반응 분석 결과가 JSON 객체가 아님: {cleaned_output}",
                                   extra=extra_log_base)
            except Exception as e:
                logger.error(f"플랫폼 '{platform_name}' 반응 분석 JSON 파싱 오류: {e}. 원본: {cleaned_output}", extra=extra_log_base)
        else:
            logger.error(f"플랫폼 '{platform_name}' 반응 분석 LLM 호출 실패: {llm_response.get('error')}", extra=extra_log_base)

        return platform_name, analysis_result, think_content

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        idea_sec = state.idea

        work_id = meta_sec.work_id
        node_name = self.__class__.__name__
        extra_log_base = {"work_id": work_id, "node_name": node_name, "node_order": NODE_ORDER}

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info(f"{node_name} 진입: 수집된 커뮤니티 반응 분석 시작.", extra=extra_log_base)

        scraped_reactions = idea_sec.scraped_community_reactions
        structured_analysis = idea_sec.structured_issue_analysis

        if not scraped_reactions:
            logger.warning("분석할 스크랩된 커뮤니티 반응 데이터가 없습니다. N06A 결과를 확인하세요.", extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"  # 또는 SKIPPED
            idea_sec.community_reaction_analysis = {}
            return {"meta": meta_sec.model_dump(), "idea": idea_sec.model_dump(exclude_unset=True)}

        if not structured_analysis or not structured_analysis.get("core_problem_definition"):
            logger.warning("핵심 문제 정의가 없어 반응 분석의 컨텍스트가 부족합니다. N06 결과를 확인하세요.", extra=extra_log_base)
            # 핵심 문제 정의 없이 진행하거나, 오류 처리 가능. 여기서는 진행.
            issue_core_problem = "정의되지 않은 이슈"
        else:
            issue_core_problem = structured_analysis.get("core_problem_definition", "정의되지 않은 이슈")

        analysis_tasks = []
        for platform, reactions_list in scraped_reactions.items():
            if reactions_list:  # 실제 반응이 있는 플랫폼에 대해서만 분석
                analysis_tasks.append(
                    self._analyze_single_platform(platform, reactions_list, issue_core_problem, work_id, node_name,
                                                  extra_log_base)  # type: ignore
                )

        final_reaction_analysis: Dict[str, Dict[str, Any]] = {}
        if analysis_tasks:
            logger.info(f"{len(analysis_tasks)}개 플랫폼 반응 분석 병렬 시작.", extra=extra_log_base)
            analysis_results_tuples = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            for result_tuple_or_exc in analysis_results_tuples:
                if isinstance(result_tuple_or_exc, Exception):
                    logger.error(f"커뮤니티 반응 분석 작업 중 예외 발생: {result_tuple_or_exc}", extra=extra_log_base,
                                 exc_info=result_tuple_or_exc)
                    continue

                platform_name_res, analysis_data, think_content_platform = result_tuple_or_exc
                if think_content_platform:
                    meta_sec.llm_think_traces.append({
                        "node_name": f"{node_name}_Analyze_{platform_name_res}",
                        "request_id": f"{work_id}_{node_name}_AnalyzeReaction_{platform_name_res}",  # request_id 일관성
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "log_content": think_content_platform
                    })
                if analysis_data:  # 유효한 분석 결과만 저장
                    final_reaction_analysis[platform_name_res] = analysis_data
        else:
            logger.info("분석할 커뮤니티 반응이 없습니다.", extra=extra_log_base)

        idea_sec.community_reaction_analysis = final_reaction_analysis
        meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"
        logger.info(f"{node_name} 완료: 수집된 커뮤니티 반응 분석 완료.", extra=extra_log_base)
        return {"meta": meta_sec.model_dump(), "idea": idea_sec.model_dump(exclude_unset=True)}