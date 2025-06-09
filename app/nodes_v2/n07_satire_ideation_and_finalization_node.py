# ai/app/nodes_v2/n07_satire_ideation_and_finalization_node.py

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.workflows.state_v2 import WorkflowState, IdeaSection, MetaSection  # FinalSatiricalIdea 타입 정의 필요
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json  # 또는 직접 json.loads 사용

logger = get_logger(__name__)

# 이 노드의 워크플로우 내 순서 정의
NODE_ORDER = 10  # N06 시리즈 다음

# LLM이 생성할 아이디어 개수 관련 상수
INITIAL_IDEAS_TO_GENERATE = 5  # 풍자 기법 적용 시 생성할 아이디어 수
FINAL_IDEAS_TO_SELECT = 3  # 윤리적 검토 후 최종 선택할 아이디어 수 (3~5개 범위 내)


class N07SatireIdeationAndFinalizationNode:
    """
    (N07 - 신규/업그레이드) 풍자 아이디어 개발 및 최종화 노드.
    N06 단계의 분석 정보와 초기 반응을 바탕으로, 다양한 풍자 기법을 적용하여
    만평/밈/웹툰 컨셉 아이디어를 개발하고, 매력도 강화 및 윤리적 검토를 거쳐
    최종 아이디어를 선정합니다.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        # 사용 가능한 풍자 기법 목록 (LLM 프롬프트에 활용)
        self.satire_techniques = [
            "Exaggeration (과장)", "Understatement (절제된 표현)", "Irony (아이러니 - 상황적/언어적)",
            "Parody (패러디)", "Juxtaposition (병치/대조)", "Symbolism (상징화)",
            "Wordplay/Puns (언어유희)", "Characterization (캐릭터화/의인화)",
            "Reversal (반전)", "Satirical Analogy (풍자적 유추)", "Absurdity (부조리함 강조)"
        ]
        # 윤리적 검토 가이드라인 요약 (실제로는 더 상세한 내용을 설정에서 가져오거나 주입)
        self.ethical_guidelines_summary = """
Ethical Guidelines for Satire:
1. Avoid hate speech, discrimination, or inciting violence against any group or individual.
2. Do not spread misinformation or disinformation. Satire should be based on truth, even if exaggerated.
3. Avoid excessive personal attacks; focus on systems, actions, or ideologies.
4. Be mindful of vulnerable groups and avoid punching down.
5. Ensure satire does not trivialize serious harm or trauma.
6. Aim for constructive criticism or encouraging reflection, not just nihilistic cynicism.
7. Consider potential misinterpretations and the overall societal impact.
"""

    def _prepare_input_summary_for_llm(self, state: WorkflowState) -> str:
        """LLM에 전달할 1단계(N06 시리즈) 결과 요약 문자열을 생성합니다."""
        idea_sec = state.idea
        summary_parts = []

        if idea_sec.structured_issue_analysis:
            sa = idea_sec.structured_issue_analysis
            summary_parts.append("# Issue Analysis (from N06):")
            summary_parts.append(f"- Core Problem: {sa.get('core_problem_definition', 'N/A')}")
            summary_parts.append(f"- Key Actors/Targets: {sa.get('key_actors_targets', 'N/A')}")
            summary_parts.append(f"- Social Context: {sa.get('social_context_background', 'N/A')}")
            summary_parts.append(f"- Potential Satire Points: {'; '.join(sa.get('potential_satire_points', ['N/A']))}")
            summary_parts.append(f"- Keywords: {', '.join(sa.get('extracted_keywords', ['N/A']))}")
            summary_parts.append(f"- Dominant Emotion: {sa.get('dominant_public_emotion', 'N/A')}")
        else:
            summary_parts.append("# Issue Analysis (from N06): Not available.")

        if idea_sec.community_reaction_analysis:
            summary_parts.append("\n# Analysis of Actual Community Reactions (from N06B):")
            for platform, analysis in idea_sec.community_reaction_analysis.items():
                if isinstance(analysis, dict):
                    summary_parts.append(f"## {platform.upper()} Reactions Analysis:")
                    summary_parts.append(
                        f"  - Style Summary (Korean): {analysis.get('platform_style_summary_korean', 'N/A')}")
                    summary_parts.append(
                        f"  - Key Arguments (Korean): {'; '.join(analysis.get('key_arguments_observed_korean', ['N/A']))}")
                    summary_parts.append(
                        f"  - Additional Satire Points (Korean): {'; '.join(analysis.get('additional_satire_points_from_reactions_korean', ['N/A']))}")
        elif idea_sec.generated_satirical_reactions:  # N06B 결과가 없고 N06C 결과만 있다면 (흐름상 N06B가 선행)
            summary_parts.append("\n# Generated Initial Satirical Reactions (from N06C):")
            for platform, reactions in idea_sec.generated_satirical_reactions.items():
                summary_parts.append(f"## {platform.upper()} Style (Generated):")
                for reaction_item in reactions[:2]:  # 일부만 표시
                    summary_parts.append(
                        f"  - [{reaction_item.get('type', 'text')}] {reaction_item.get('content_korean', 'N/A')}")
        else:
            summary_parts.append("\n# Community Reactions/Analysis: Not available or not generated.")

        return "\n".join(summary_parts)

    # --- 1. 풍자 기법 적용 및 아이디어 확장 모듈 ---
    def _build_satire_expansion_prompt(self, previous_analysis_summary: str) -> str:
        """풍자 기법을 적용하여 만평/웹툰 아이디어 확장을 요청하는 LLM 프롬프트 (영문)."""
        # LLM이 반환할 아이디어의 JSON 스키마 정의
        idea_schema = {
            "idea_type": "<cartoon_panel | meme_concept | short_webtoon_synopsis>",
            "title_concept_korean": "<Korean: A catchy title or concept name for the idea.>",
            "detailed_description_korean": "<Korean: Detailed description. For cartoons/memes: visual elements, key characters, expressions, setting. For webtoon: brief plot, character interactions, punchline/twist.>",
            "key_dialogue_or_caption_korean": "<Korean: Core dialogue, narration, or meme caption.>",
            "applied_satire_techniques": ["<Name of a primary satire technique used (e.g., Exaggeration)>",
                                          "<(Optional) Secondary technique>"],
            "brief_rationale_for_technique_korean": "<Korean: Briefly explain why this technique is effective for this idea and issue.>"
        }
        # LLM에게 여러 개의 아이디어를 리스트 형태로 반환하도록 요청
        output_schema = {"satirical_ideas": [idea_schema]}  # 리스트로 감쌈
        json_schema_str = json.dumps(output_schema, indent=2, ensure_ascii=False)

        return f"""
You are a highly creative satirist and content ideation expert.
Your task is to develop {INITIAL_IDEAS_TO_GENERATE} distinct satirical content ideas (e.g., single-panel cartoons, meme concepts, short webtoon synopses up to 4 panels) based on the provided issue analysis and community reaction insights.

# Input: Summary of Previous Analysis (Phases N06, N06A, N06B):
---
{previous_analysis_summary}
---

# Task Requirements:
1.  For each idea, creatively apply one or more satire techniques from the list: {', '.join(self.satire_techniques)}.
2.  Develop ideas for different formats: aim for a mix of cartoon panel ideas, meme concepts (describe visual and text), and at least one short webtoon synopsis.
3.  Consider "what if" scenarios: "What if this situation escalates دعوت 극단적으로 치닫는다면?", "What if this public figure were in a different, ironic situation?" to expand ideas.
4.  All descriptive content within the ideas (title, description, dialogue/caption, rationale) MUST be in KOREAN. Only the 'idea_type' and 'applied_satire_techniques' names can be in English or as provided in the list.

# Output Format:
Return ONLY a single, valid JSON object containing a list of ideas, strictly adhering to the schema below. The main key should be "satirical_ideas".
Do NOT include any extraneous text, <think> tags, or comments outside this JSON structure.

# JSON Schema for Output ("satirical_ideas" should be a list of objects matching this):
{json_schema_str} 
(Note: The 'applied_satire_techniques' field should list names from: {', '.join(self.satire_techniques)})
"""

    # --- 2. 아이디어 구조화 및 매력도 강화 모듈 ---
    def _build_idea_refinement_prompt(self, generated_ideas_json_str: str) -> str:
        """생성된 아이디어들의 매력도 강화 및 분석을 요청하는 LLM 프롬프트 (영문)."""
        # LLM이 반환할 아이디어 강화 결과의 JSON 스키마 정의
        # 입력 아이디어 구조를 유지하면서 'appeal_analysis_korean' 필드 추가
        refined_idea_schema = {
            # (이전 _build_satire_expansion_prompt의 idea_schema 필드들 ... )
            "idea_type": "<cartoon_panel | meme_concept | short_webtoon_synopsis>",
            "title_concept_korean": "<Korean: Refined catchy title or concept name.>",
            "detailed_description_korean": "<Korean: Refined and more vivid detailed description.>",
            "key_dialogue_or_caption_korean": "<Korean: Refined and punchier core dialogue/caption.>",
            "applied_satire_techniques": ["<Technique1>", "<Technique2>"],
            "brief_rationale_for_technique_korean": "<Korean: Refined rationale.>",
            # 새로 추가/강화될 필드
            "suggested_catchphrase_korean": "<Korean: A very short, memorable catchphrase or meme text for this idea.>",
            "appeal_analysis_korean": {
                "target_emotion_korean": "<Korean: What emotion(s) does this idea primarily evoke in the audience (e.g., amusement, anger, sympathy, cringeworthy empathy)? रेफरेंसिंग dominant public emotion from N06.>",
                "main_appeal_points_korean": [
                    "<Korean: Why would this idea appeal to the public? (e.g., timeliness, relatable frustration, specific humor code, sharpness of critique). List 2-3 points.>",
                    "..."],
                "potential_virality_factors_korean": [
                    "<Korean: What elements could contribute to this idea's shareability or virality (e.g., shocking, funny, highly debatable)? List 1-2 factors.>",
                    "..."]
            }
        }
        output_schema = {"refined_satirical_ideas": [refined_idea_schema]}
        json_schema_str = json.dumps(output_schema, indent=2, ensure_ascii=False)

        return f"""
You are an expert content strategist and editor specializing in satirical content.
Your task is to refine and enhance the provided list of raw satirical ideas. For each idea, you need to:
1.  Sharpen the title/concept.
2.  Make the description more vivid and engaging.
3.  Refine key dialogues or captions to be more impactful.
4.  Propose a short, memorable catchphrase or meme text.
5.  Analyze its appeal: identify target emotions, main appeal points (why it would resonate with the public, considering factors like timeliness, relatability, humor, critique sharpness), and potential virality factors. All analysis MUST be in KOREAN.

# Input: List of Generated Satirical Ideas (JSON format):
---
{generated_ideas_json_str}
---

# Output Format:
Return ONLY a single, valid JSON object. The main key should be "refined_satirical_ideas", containing a list of refined idea objects. Each refined idea object must strictly adhere to the schema below.
Do NOT include any extraneous text, <think> tags, or comments outside this JSON structure.

# JSON Schema for Output ("refined_satirical_ideas" should be a list of objects matching this):
{json_schema_str}
"""

    # --- 3. 윤리적 검토 및 최종 아이디어 선정 필터 ---
    def _build_ethical_review_prompt(self, refined_ideas_json_str: str) -> str:
        """정제된 아이디어 목록에 대한 윤리적 검토 및 최종 선정을 요청하는 LLM 프롬프트 (영문)."""
        # LLM이 반환할 최종 아이디어 목록의 스키마 (입력 아이디어 스키마와 동일 + ethical_review_summary_korean)
        final_idea_schema_with_review = {
            # (이전 _build_idea_refinement_prompt의 refined_idea_schema 필드들 ... )
            "idea_type": "...", "title_concept_korean": "...", "detailed_description_korean": "...",
            "key_dialogue_or_caption_korean": "...", "applied_satire_techniques": [],
            "brief_rationale_for_technique_korean": "...",
            "suggested_catchphrase_korean": "...",
            "appeal_analysis_korean": {"target_emotion_korean": "...", "main_appeal_points_korean": [],
                                       "potential_virality_factors_korean": []},
            # 윤리적 검토 결과 필드 추가
            "ethical_review_summary_korean": "<Korean: Summary of ethical review. 'No major concerns', or if concerns exist: 'Potential issue: [type of issue], Suggestion: [how to mitigate/modify, or if it should be excluded and why]'.>"
        }
        output_schema = {
            "ethically_reviewed_and_selected_ideas": [final_idea_schema_with_review]}  # 최종 선정 아이디어 수 제한은 LLM 지시로
        json_schema_str = json.dumps(output_schema, indent=2, ensure_ascii=False)

        return f"""
You are an ethical review specialist with a deep understanding of satire and its societal impact.
Your task is to review the provided list of refined satirical ideas based on the following ethical guidelines. Then, select the best {FINAL_IDEAS_TO_SELECT} ideas that are both impactful and ethically sound.

# Ethical Guidelines Summary:
---
{self.ethical_guidelines_summary}
---

# Input: List of Refined Satirical Ideas (JSON format):
---
{refined_ideas_json_str}
---

# Task Requirements:
1.  For EACH idea in the input list, provide an 'ethical_review_summary_korean' (in KOREAN). This summary should state if there are 'No major concerns' or identify potential issues (e.g., hate speech, misinformation, excessive generalization, disproportionate attack on vulnerable groups). If issues are found, briefly suggest how to modify the idea to mitigate them, or if it should be excluded and why.
2.  From the reviewed list, select the top {FINAL_IDEAS_TO_SELECT} most effective AND ethically appropriate satirical ideas.
3.  The final output should be a list containing only these {FINAL_IDEAS_TO_SELECT} selected ideas, each including its 'ethical_review_summary_korean'.

# Output Format:
Return ONLY a single, valid JSON object. The main key should be "ethically_reviewed_and_selected_ideas", containing a list of exactly {FINAL_IDEAS_TO_SELECT} idea objects (or fewer if not enough pass the review). Each idea object must strictly adhere to the schema below, including all original fields plus your 'ethical_review_summary_korean'.
Do NOT include any extraneous text, <think> tags, or comments outside this JSON structure.

# JSON Schema for Output ("ethically_reviewed_and_selected_ideas" list items):
{json_schema_str}
"""

    async def _call_llm_with_think_handling(
            self,
            messages: List[Dict[str, str]],
            request_id: str,
            temperature: float,
            max_tokens: int,
            meta_sec: MetaSection,  # MetaSection 직접 전달
            log_context_node_name: str,  # 로깅용 노드/단계 이름
            extra_log_base: dict
    ) -> Tuple[Optional[Dict[str, Any]], str]:  # (parsed_json, cleaned_llm_output_str)
        """LLM을 호출하고, <think> 태그를 처리하며, JSON을 파싱하는 공통 로직"""

        llm_response = await self.llm_service.generate_text(
            messages=messages, request_id=request_id,
            temperature=temperature, max_tokens=max_tokens
        )
        cleaned_output = llm_response.get("generated_text", "")
        think_content = llm_response.get("think_content")

        if think_content:
            meta_sec.llm_think_traces.append({
                "node_name": log_context_node_name, "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(), "log_content": think_content
            })
            logger.debug(f"LLM <think> 내용 저장됨 (Request ID: {request_id}, Context: {log_context_node_name})",
                         extra=extra_log_base)

        if not cleaned_output or llm_response.get("error"):
            error_msg = f"LLM 호출 실패 (Context: {log_context_node_name}): {llm_response.get('error', 'Empty output')}"
            logger.error(error_msg, extra=extra_log_base)
            raise ValueError(error_msg)  # 예외를 발생시켜 호출부에서 처리

        logger.debug(f"LLM 응답 (Context: {log_context_node_name}, 정리 후): {summarize_for_logging(cleaned_output, 300)}",
                     extra=extra_log_base)

        try:
            parsed_json = extract_json(cleaned_output)  # extract_json은 <think> 태그가 이미 제거된 텍스트를 받는다고 가정
            if not isinstance(parsed_json, dict):  # LLM이 스키마의 최상위 키를 포함한 딕셔너리를 반환해야 함
                # 일부 LLM은 최상위 키 없이 바로 리스트를 반환할 수 있음. 이 경우 extract_json이나 후처리 조정 필요.
                # 여기서는 프롬프트에서 최상위 키(예: "satirical_ideas")를 갖는 객체를 요청했으므로 dict여야 함.
                raise ValueError(
                    f"LLM 응답 JSON이 최상위 딕셔너리 객체가 아님 (Context: {log_context_node_name}). 파싱된 내용: {type(parsed_json)}")
            return parsed_json, cleaned_output
        except Exception as e:
            error_msg = f"LLM 응답 JSON 파싱 실패 (Context: {log_context_node_name}): {e}. 원본(정리후): {cleaned_output}"
            logger.error(error_msg, extra=extra_log_base)
            raise ValueError(error_msg) from e

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        idea_sec = state.idea  # N06 시리즈의 결과 사용 및 N07 결과 저장

        work_id = meta_sec.work_id
        node_name = self.__class__.__name__
        extra_log_base = {"work_id": work_id, "node_name": node_name, "node_order": NODE_ORDER}

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info(f"{node_name} 진입: 풍자 아이디어 개발 및 최종화 시작.", extra=extra_log_base)

        # --- 입력 데이터 준비 (N06 시리즈 결과) ---
        if not idea_sec.structured_issue_analysis:
            logger.error("풍자 아이디어 생성을 위한 선행 이슈 분석 결과(N06)가 없습니다.", extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"meta": meta_sec.model_dump(), "idea": idea_sec.model_dump(exclude_unset=True)}

        llm_input_summary = self._prepare_input_summary_for_llm(state)
        logger.debug(f"아이디어 생성 LLM 입력용 요약 정보 준비 완료:\n{llm_input_summary[:500]}...", extra=extra_log_base)

        try:
            # === 단계 1: 풍자 기법 적용 및 아이디어 확장 ===
            logger.info("1단계: 풍자 기법 적용 및 아이디어 확장 LLM 호출 시작.", extra=extra_log_base)
            expansion_prompt = self._build_satire_expansion_prompt(llm_input_summary)
            request_id_expansion = f"{work_id}_{node_name}_SatireExpansion"

            parsed_expanded_ideas_json, _ = await self._call_llm_with_think_handling(
                messages=[{"role": "system", "content": "You are a creative satirist and content ideation expert."},
                          {"role": "user", "content": expansion_prompt}],
                request_id=request_id_expansion, temperature=0.7, max_tokens=3000,  # 창의성을 위해 온도 약간 높임
                meta_sec=meta_sec, log_context_node_name=f"{node_name}_Expansion", extra_log_base=extra_log_base
            )

            initial_generated_ideas = parsed_expanded_ideas_json.get("satirical_ideas", [])
            if not initial_generated_ideas or not isinstance(initial_generated_ideas, list):
                raise ValueError("LLM이 초기 풍자 아이디어 목록('satirical_ideas')을 생성하지 못했거나 형식이 올바르지 않습니다.")
            logger.info(f"{len(initial_generated_ideas)}개의 초기 풍자 아이디어 생성 성공.", extra=extra_log_base)

            # === 단계 2: 아이디어 구조화 및 매력도 강화 ===
            logger.info("2단계: 생성된 아이디어 구조화 및 매력도 강화 LLM 호출 시작.", extra=extra_log_base)
            # LLM에 전달하기 위해 다시 JSON 문자열로 변환
            initial_ideas_json_str = json.dumps({"satirical_ideas": initial_generated_ideas}, ensure_ascii=False,
                                                indent=2)
            refinement_prompt = self._build_idea_refinement_prompt(initial_ideas_json_str)
            request_id_refinement = f"{work_id}_{node_name}_IdeaRefinement"

            parsed_refined_ideas_json, _ = await self._call_llm_with_think_handling(
                messages=[{"role": "system", "content": "You are an expert content strategist and editor for satire."},
                          {"role": "user", "content": refinement_prompt}],
                request_id=request_id_refinement, temperature=0.4, max_tokens=3500,  # 내용을 다듬는 것이므로 온도 약간 낮춤
                meta_sec=meta_sec, log_context_node_name=f"{node_name}_Refinement", extra_log_base=extra_log_base
            )

            refined_ideas = parsed_refined_ideas_json.get("refined_satirical_ideas", [])
            if not refined_ideas or not isinstance(refined_ideas, list):
                raise ValueError("LLM이 정제된 풍자 아이디어 목록('refined_satirical_ideas')을 생성하지 못했거나 형식이 올바르지 않습니다.")
            logger.info(f"{len(refined_ideas)}개의 풍자 아이디어 정제 및 매력도 분석 성공.", extra=extra_log_base)

            # === 단계 3: 윤리적 검토 및 최종 아이디어 선정 ===
            logger.info("3단계: 정제된 아이디어 윤리적 검토 및 최종 선정 LLM 호출 시작.", extra=extra_log_base)
            refined_ideas_json_str = json.dumps({"refined_satirical_ideas": refined_ideas}, ensure_ascii=False,
                                                indent=2)
            ethical_review_prompt = self._build_ethical_review_prompt(refined_ideas_json_str)
            request_id_ethical_review = f"{work_id}_{node_name}_EthicalReview"

            parsed_final_ideas_json, _ = await self._call_llm_with_think_handling(
                messages=[{"role": "system", "content": "You are an ethical review specialist for satirical content."},
                          {"role": "user", "content": ethical_review_prompt}],
                request_id=request_id_ethical_review, temperature=0.1, max_tokens=4000,  # 신중한 판단을 위해 온도 매우 낮춤
                meta_sec=meta_sec, log_context_node_name=f"{node_name}_EthicalReview", extra_log_base=extra_log_base
            )

            final_selected_ideas = parsed_final_ideas_json.get("ethically_reviewed_and_selected_ideas", [])
            if not final_selected_ideas or not isinstance(final_selected_ideas, list):
                # LLM이 최종 아이디어를 선택하지 못했더라도, 정제된 아이디어 중 일부를 사용하거나 오류 처리
                logger.warning("LLM이 최종 아이디어를 선정하지 못했거나 형식이 올바르지 않습니다. 정제된 아이디어를 최종 결과로 사용합니다 (윤리 검토 요약은 없을 수 있음).",
                               extra=extra_log_base)
                # 이 경우, refined_ideas에 ethical_review_summary_korean 필드를 수동으로 추가하거나,
                # final_selected_ideas를 refined_ideas로 대체하고, 각 아이템에 "ethical_review_summary_korean": "LLM 자동 선정 실패" 추가
                for idea in refined_ideas:
                    idea.setdefault("ethical_review_summary_korean", "LLM 자동 최종 선정 단계에서 처리되지 않음. 개별 검토 필요.")
                final_selected_ideas = refined_ideas[:FINAL_IDEAS_TO_SELECT]  # 정제된 아이디어 중 상위 N개 사용

            # 최종 아이디어 목록을 IdeaSection의 final_comic_ideas 필드에 저장 (구조는 FinalSatiricalIdea TypedDict/Pydantic 모델과 일치해야 함)
            # 현재 LLM 반환 스키마가 FinalSatiricalIdea와 거의 일치하므로 바로 할당 시도
            idea_sec.final_comic_ideas = final_selected_ideas  # type: ignore
            logger.info(f"{len(final_selected_ideas)}개의 최종 풍자 아이디어 선정 완료.", extra=extra_log_base)

            meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"

        except ValueError as ve:  # _call_llm_with_think_handling 에서 발생시킨 명시적 오류
            logger.error(f"{node_name} 실행 중 예측된 오류: {ve}", extra=extra_log_base,
                         exc_info=False)  # 이미 로깅된 내용이므로 exc_info=False 가능
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
        except Exception as e:  # 그 외 예상치 못한 오류
            logger.error(f"{node_name} 실행 중 예상치 못한 전체 오류 발생: {e}", extra=extra_log_base, exc_info=True)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"

        return {
            "meta": meta_sec.model_dump(),
            "idea": idea_sec.model_dump(exclude_unset=True)  # 변경된 idea 섹션 전체 반환
        }