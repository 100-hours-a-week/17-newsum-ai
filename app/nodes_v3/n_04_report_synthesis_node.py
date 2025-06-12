# app/nodes_v3/n_04_report_drafting_node.py
"""n_04_ReportDraftingNode (Final Version)

NewSum LangGraph 워크플로우의 **네 번째 노드**(순번 n_04).
'전문 보고서 작성 에이전트'로서, n_02의 보고서 계획과 n_03의 수집된 자료를 바탕으로,
SLM의 토큰 제한을 고려하여 실제 보고서 초안을 작성한다.

주요 워크플로우:
1.  **콘텐츠 패킹**: 토큰 예산 내에서 우선순위(한국어 우선 등)에 따라 참고할 소스 자료를 선별 및 조합한다.
2.  **메타데이터 헤더 추가**: 각 소스 자료에 URL, 제목, 검색 스니펫을 헤더로 추가하여 LLM의 이해를 돕는다.
3.  **섹션 단위 순차 작성**: 보고서 계획에 따라 각 섹션을 독립적으로 작성하여 일관성과 품질을 유지한다.
4.  **최종 보고서 조립**: 작성된 모든 섹션을 하나로 합쳐 완전한 보고서 초안을 생성하고 State에 저장한다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pydantic import ValidationError

from app.config.settings import Settings
from app.services.database_client import DatabaseClient
from app.services.llm_service import LLMService
from app.utils.logger import get_logger
from app.workflows.state_v3 import (
    OverallWorkflowState,
)
from app.utils.report_html_renderer import render_report_html_from_markdown, save_report_html

# ──────────────────────────────────────────────────────────────────────────────
# 설정 및 상수
# ──────────────────────────────────────────────────────────────────────────────
settings = Settings()
MAX_SOURCE_TOKENS = 10000  # SLM에 전달할 소스 텍스트의 최대 토큰 수 (대략적)
logger = get_logger("n_04_ReportDraftingNode_Final")


# ──────────────────────────────────────────────────────────────────────────────
# 노드 구현
# ──────────────────────────────────────────────────────────────────────────────
class N04ReportDraftingNode:
    """LangGraph async 노드 – 보고서 초안 작성 (n_04)"""

    def __init__(self, redis_client: DatabaseClient, llm_service: LLMService):
        self.redis = redis_client
        self.llm = llm_service
        self.logger = logger

    async def __call__(self, current_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        work_id = current_state_dict.get("work_id")
        log_extra = {"work_id": work_id or "UNKNOWN_WORK_ID_N04"}
        self.logger.info("N04ReportDraftingNode (Final) 시작.", extra=log_extra)

        try:
            workflow_state = OverallWorkflowState(**current_state_dict)
        except ValidationError as e:
            self.logger.error(f"N04 State 유효성 검사 실패: {e}", extra=log_extra)
            current_state_dict["error_message"] = f"N04 State 유효성 검사 실패: {e}"
            return current_state_dict

        await self._run_drafting_pipeline(workflow_state)
        return await self._finalize_and_save_state(workflow_state, log_extra)

    def _estimate_token_count(self, text: str) -> int:
        """텍스트의 대략적인 토큰 수를 계산합니다."""
        return int(len(text) * 1.8)

    def _pack_source_content(self, sources: List[Dict[str, Any]], budget: int) -> str:
        """주어진 소스들을 우선순위에 따라 토큰 예산 내에서 '메타데이터 헤더'와 함께 패킹합니다."""
        sorted_sources = sorted(sources, key=lambda x: x.get('language', 'en') == 'ko', reverse=True)

        packed_texts = []
        current_token_count = 0

        for i, source in enumerate(sorted_sources):
            header = f"""--- Source [{i + 1}] ---
[URL]: {source.get('source_url', 'N/A')}
[Language]: {source.get('language', 'N/A')}
[Title]: {source.get('title', 'N/A')}
[Search Snippet]: {source.get('snippet', 'N/A')}

[Full Text]:
"""
            full_text = source.get('full_text', '')
            header_token_estimate = self._estimate_token_count(header)
            text_token_estimate = self._estimate_token_count(full_text)

            if current_token_count + header_token_estimate + 50 > budget:  # 헤더가 들어갈 공간도 없으면 중단
                break

            packed_texts.append(header)
            current_token_count += header_token_estimate

            # 남은 예산만큼만 본문 텍스트를 잘라서 추가
            remaining_budget = budget - current_token_count
            if text_token_estimate > remaining_budget:
                chars_to_add = int(remaining_budget / 1.8)
                packed_texts.append(full_text[:chars_to_add] + "...")
                current_token_count += remaining_budget
                break  # 예산 초과 시 루프 종료
            else:
                packed_texts.append(full_text)
                current_token_count += text_token_estimate

        self.logger.info(f"{len(packed_texts)}개의 소스 블록, 약 {current_token_count} 토큰으로 패킹 완료.")
        return "\n".join(packed_texts)

    async def _draft_single_section(self, overall_topic: str, sec_title: str, sec_plan: Dict, source_text: str,
                                    work_id: str, section_number: int, is_final_section: bool = False) -> str:
        """단일 섹션을 작성하는 LLM 호출 함수"""
        log_extra = {"work_id": work_id}
        self.logger.info(f"섹션 '{sec_title}' 작성 시작.", extra=log_extra)
        # --- 프롬프트 수정 제안 ---
        # 마지막 섹션을 위한 추가 지시사항
        final_section_instruction = ""
        if is_final_section:
            final_section_instruction = """
        * IMPORTANT: As this is the final section, write it in a conclusive tone. Summarize the key findings, challenges, and future outlook discussed throughout the report, and present a final, overarching conclusion.
        """

        prompt = f"""You are a professional report writer specializing in media and technology trends. Your current task is to write a single section of a larger report based *only* on the provided reference materials.

            **Overall Report Topic:**
            "{overall_topic}"
    
            ---
    
            **1. Instructions for the Section to Write:**
            * **Section Title:** "{sec_title}"
            * **Role of this Section:** "{sec_plan.get('role')}"
            * **Key Points to Cover (sub-topics):** {sec_plan.get('sub_topics')}
    
            ---
    
            **2. Reference Materials:**
            {source_text}
    
            ---
    
            **3. Output Requirements:**
            * Write the content for the "**{sec_title}**" section ONLY.
            * Start with a headline using the correct section number (e.g., "## {section_number}. {sec_title}").{final_section_instruction}
            * Use clear, professional, and analytical Korean language suitable for a report.
            * Ensure all claims are supported by the provided reference materials and are cited correctly.
            * Format the output in Markdown.
        """
        try:
            resp = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"draft-section-{sec_title.replace(' ', '_')}-{work_id}",
                max_tokens=2048,  # 섹션 하나를 작성하므로 충분한 길이 허용
                temperature=0.2
            )
            drafted_section = resp.get("generated_text", "").strip()
            self.logger.info(f"섹션 '{sec_title}' 작성 완료.", extra=log_extra)
            return drafted_section
        except Exception as e:
            self.logger.error(f"섹션 '{sec_title}' 작성 중 오류 발생: {e}", extra=log_extra)
            return f"## {sec_title}\n\n(오류로 인해 이 섹션을 작성하지 못했습니다.)"

    async def _run_drafting_pipeline(self, workflow_state: OverallWorkflowState):
        """보고서 초안 작성을 위한 메인 파이프라인 (콘텐츠 패킹 적용)"""
        work_id = workflow_state.work_id
        self.logger.info("보고서 초안 작성 파이프라인 시작.", extra={"work_id": work_id})

        structure = workflow_state.report_planning.structure
        source_materials = workflow_state.source_collect.results
        drafted_sections = []
        overall_topic = workflow_state.topic_clarification.draft
        num_sections = len(structure) # 전체 섹션 수 파악

        relationship_analysis = workflow_state.report_planning.query_analysis['relationship']

        for i, (sec_title, sec_content) in enumerate(structure.items()):
            section_number = i + 1  # 1부터 시작하도록 조정
            relevant_sources = source_materials.get(sec_title, [])
            if not relevant_sources:
                drafted_sections.append(f"## {sec_title}\n\n(이 섹션에 대한 참고 자료가 수집되지 않았습니다.)")
                continue

            packed_source_text = self._pack_source_content(relevant_sources, MAX_SOURCE_TOKENS)

            if not packed_source_text:
                drafted_sections.append(f"## {sec_title}\n\n(참고 자료가 있으나 토큰 예산 내에 포함시키지 못했습니다.)")
                continue

            is_final_section = (section_number == num_sections)  # 마지막 섹션인지 확인

            drafted_section = await self._draft_single_section(
                overall_topic, sec_title, sec_content, packed_source_text, work_id, section_number, is_final_section
            )
            drafted_sections.append(drafted_section)

        # 최종 보고서 초안 조립
        final_draft = "\n\n---\n\n".join(drafted_sections)

        # 1. 새로 추가된 메서드를 호출하여 동적으로 제목 생성
        report_title_text = await self._generate_report_title(overall_topic, relationship_analysis, work_id)

        # 2. 생성된 제목과 본문을 합쳐 최종 보고서 완성
        final_draft_body = "\n\n---\n\n".join(drafted_sections)
        final_draft_with_title = f"# {report_title_text}\n\n{final_draft_body}"

        workflow_state.report_draft.draft = final_draft_with_title
        workflow_state.report_draft.is_ready = True
        self.logger.info("보고서 초안 작성을 모두 완료했습니다.", extra={"work_id": work_id})

        # HTML 파일로 저장 (최종 확정 시점)
        try:
            html = render_report_html_from_markdown(work_id, report_title_text, final_draft_with_title)
            save_report_html(work_id, html)
            self.logger.info(f"HTML 보고서가 저장되었습니다. work_id={work_id}", extra={"work_id": work_id})
        except Exception as e:
            self.logger.error(f"HTML 보고서 저장 중 오류: {e}", extra={"work_id": work_id})

        # --- (수정) 보고서에서 category, keywords 추출 (JSON 포맷) ---
        import json as _json
        try:
            extract_prompt = f"""아래는 전문 보고서입니다.\n\n---\n{final_draft_with_title}\n---\n\n아래 두 가지를 반드시 JSON 형식(키: category, keywords)으로 반환하세요.\n- category: POLITICS, IT, FINANCE 중 하나의 문자열\n- keywords: 이 보고서의 핵심 키워드 5개 (중복 없이, 명사 위주, 한글/영어 혼용 가능, 리스트 형태)\n\n[출력 예시]\n{{\n  \"category\": \"IT\",\n  \"keywords\": [\"디지털 네이티브\", \"오프라인 활동\", \"스마트폰 의존\", \"정신건강\", \"커뮤니케이션 패러다임\"]\n}}\n"""
            resp = await self.llm.generate_text(
                messages=[{"role": "user", "content": extract_prompt}],
                request_id=f"extract-cat-keywords-json-{work_id}",
                max_tokens=200,
                temperature=0.2
            )
            result_text = resp.get("generated_text", "").strip()
            # JSON 파싱
            try:
                result_json = _json.loads(result_text)
                category = result_json.get("category", None)
                keywords = result_json.get("keywords", [])
                if not isinstance(keywords, list):
                    keywords = []
            except Exception as e:
                self.logger.error(f"category/keywords JSON 파싱 실패: {e}, 원본: {result_text}", extra={"work_id": work_id})
                category = None
                keywords = []
            if category not in {"POLITICS", "IT", "FINANCE"}:
                self.logger.warning(f"LLM이 반환한 category가 유효하지 않음: {category}", extra={"work_id": work_id})
                category = None
            if not keywords or len(keywords) < 1:
                self.logger.warning(f"LLM이 반환한 keywords가 비어있음: {keywords}", extra={"work_id": work_id})
                keywords = []
            workflow_state.report_draft.category = category
            workflow_state.report_draft.keywords = keywords
            self.logger.info(f"보고서에서 추출된 category: {category}, keywords: {keywords}", extra={"work_id": work_id})
        except Exception as e:
            self.logger.error(f"category/keywords 추출 중 오류: {e}", extra={"work_id": work_id})
            workflow_state.report_draft.category = None
            workflow_state.report_draft.keywords = []

    async def _finalize_and_save_state(self, workflow_state: OverallWorkflowState, log_extra: Dict) -> Dict[str, Any]:
        """최종 상태를 저장하고 반환합니다."""
        updated_state_dict = workflow_state.model_dump(mode='json')
        await self._save_workflow_state_to_redis(workflow_state.work_id, updated_state_dict)
        self.logger.info("N04 노드 처리 완료 및 상태 저장.", extra=log_extra)
        return updated_state_dict

    async def _save_workflow_state_to_redis(self, work_id: str, state_dict: Dict[str, Any]):
        key = f"workflow:{work_id}:full_state"
        try:
            json_compatible_state = json.loads(json.dumps(state_dict, default=str))
            await self.redis.set(key, json_compatible_state, expire=60 * 60 * 6)
        except Exception as e:
            self.logger.error(f"Redis 상태 저장 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})


    async def _generate_report_title(self, topic: str, relationship: str, work_id: str) -> str:
            """LLM을 사용하여 주제와 키워드 관계 분석을 기반으로 보고서 제목을 생성합니다."""
            log_extra = {"work_id": work_id}
            self.logger.info("LLM을 이용한 보고서 제목 생성을 시작합니다.", extra=log_extra)

            prompt = f"""You are an expert copywriter specializing in creating compelling headlines for professional reports on technology and media.
    Your task is to create a single, impactful title for a report based on the provided materials.

    **1. Main Topic of the Report:**
    "{topic}"

    **2. Core Relationship Analysis (from user's query):**
    "The user is interested in how the 'A' factor (e.g., AI-driven summaries, zero-click) is causing a 'B' outcome (e.g., changes in journalism, challenges for media companies). The report should focus on analyzing this cause-and-effect relationship: {relationship}"

    **3. Instructions:**
    - Based on the topic and the detailed relationship analysis, create a single, concise, and professional report title in Korean.
    - The title should be engaging and accurately reflect the core theme.
    - Do NOT add any prefixes like "제목:" or "보고서 제목:".
    - Do NOT enclose the title in quotation marks.
    - Provide ONLY the title text itself.

    **Example Output:**
    AI 요약과 제로클릭 시대, 언론의 위기와 생존 전략

    **Please generate the title now.**
    """
            try:
                resp = await self.llm.generate_text(
                    messages=[{"role": "user", "content": prompt}],
                    request_id=f"generate-title-{work_id}",
                    max_tokens=100,
                    temperature=0.4  # 제목 생성을 위해 약간의 창의성 부여
                )
                generated_title = resp.get("generated_text", "").strip()
                if generated_title:
                    self.logger.info(f"LLM이 생성한 보고서 제목: '{generated_title}'", extra=log_extra)
                    return generated_title
                else:
                    self.logger.warning("LLM이 제목을 생성하지 못했습니다. 기본 주제를 제목으로 사용합니다.", extra=log_extra)
                    return topic  # LLM 응답이 비어있을 경우, 기본 주제로 대체
            except Exception as e:
                self.logger.error(f"보고서 제목 생성 중 오류 발생: {e}", extra=log_extra)
                return topic  # 오류 발생 시, 기본 주제로 대체





