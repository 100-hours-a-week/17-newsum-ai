# service_streamlit/state_view_utils.py

from typing import Any, Dict, List, Optional, Union
from bs4 import BeautifulSoup

# state_v2.py의 모든 모델을 직접 임포트하거나, WorkflowState만 임포트 후 접근
# 여기서는 각 섹션 모델을 직접 임포트했다고 가정 (또는 state.meta 등으로 접근)
from state_v2 import (  # 실제 경로에 맞게 수정
    WorkflowState,
    MetaSection,
    QuerySection,
    SearchSection,
    ReportSection,
    IdeaSection,
    ScenarioSection,
    ImageSection,
    UploadSection,
    ConfigSection,
    RawArticle,
    FinalSatiricalIdea
)

def summarize_html_as_text(html_content: Optional[str]) -> str:
    if not html_content:
        return "N/A"
    soup = BeautifulSoup(html_content, "html.parser")
    text_content = soup.get_text(separator="\n", strip=True)
    return text_content

def _format_optional(value: Any, default: str = "N/A") -> str:
    """None 값을 처리하고 문자열로 변환합니다. 빈 문자열도 default로 처리."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if isinstance(value, list) and not value:  # 빈 리스트
        return default
    if isinstance(value, dict) and not value:  # 빈 딕셔너리
        return default
    return str(value)


def summarize_text(text: Optional[Any]) -> str: # 위치 변경: _format_list_of_dicts 등에서 사용하기 위함
    """None을 처리하고, 텍스트를 주어진 길이로 요약하며 ...를 붙입니다."""
    if text is None:
        return "N/A"
    text_str = str(text)  # 다양한 타입이 올 수 있으므로 문자열 변
    return text_str


def _format_list_of_strings(items: Optional[List[str]], title: str, item_prefix: str = "- ") -> str:
    """문자열 리스트를 마크다운으로 변환합니다."""
    if not items:
        return f"- **{title}:** 없음\n"
    md = f"- **{title} ({len(items)}개):\n"
    for i, item in enumerate(items):
        md += f"  {item_prefix}{_format_optional(item)}\n"
    return md


def _format_list_of_dicts(items: Optional[List[Dict[str, Any]]], title: str, display_keys: Optional[List[str]] = None,
                          item_prefix: str = "  - ") -> str:
    """딕셔너리 리스트를 마크다운으로 변환합니다 (최대 N개 표시, 특정 키만 표시 가능)."""
    if not items:
        return f"- **{title}:** 0개\n"

    md = f"- **{title} ({len(items)}개):\n"
    for i, item_dict in enumerate(items):
        md += f"{item_prefix}{i + 1}. "
        if isinstance(item_dict, dict):
            # RawArticle 특별 처리 (url, title, rank, snippet) - display_keys보다 우선
            if 'title' in item_dict and 'url' in item_dict and 'snippet' in item_dict:  # RawArticle 또는 유사 구조
                md += f"[{_format_optional(item_dict.get('title'))}]({_format_optional(item_dict.get('url'))}) (Rank: {_format_optional(item_dict.get('rank'))}, Snippet: {summarize_text(item_dict.get('snippet'))})"
            # FinalSatiricalIdea 특별 처리 - display_keys보다 우선
            elif all(k in item_dict for k in ['title_concept', 'detailed_content']):  # FinalSatiricalIdea 또는 유사 구조
                md += f"**{_format_optional(item_dict.get('title_concept'))}**: {summarize_text(item_dict.get('detailed_content'))}"
                if item_dict.get('applied_satire_techniques'):
                    techniques = item_dict.get('applied_satire_techniques', [])
                    if techniques:
                        md += f" (기법: {', '.join(techniques)})"
                if item_dict.get('expected_appeal_points'):
                    md += f" (기대 효과: {summarize_text(_format_optional(item_dict.get('expected_appeal_points')))})"
                if item_dict.get('ethical_review_notes'):
                    md += f" (윤리 검토: {summarize_text(_format_optional(item_dict.get('ethical_review_notes')))})"
            elif display_keys:  # 특정 키만 표시
                display_parts = []
                for key in display_keys:
                    if key in item_dict:
                        display_parts.append(f"`{key}`: {_format_optional(item_dict[key])}")
                md += ", ".join(display_parts) if display_parts else "내용 없음"
            else:  # 일반 딕셔너리 (특별 처리나 display_keys 해당 없을 시)
                preview_items = list(item_dict.items()) # 첫 2-3개 키-값 쌍만 간단히 표시
                md += ", ".join([f"`{k}`: {summarize_text(str(v))}" for k, v in preview_items])
        else:
            md += str(item_dict)  # 딕셔너리가 아닌 경우 그냥 문자열 변환
        md += "\n"
    return md


def _format_dict_complex(data: Optional[Dict[str, Any]], title: str, item_prefix: str = "  - ") -> str:
    """복잡한 값(리스트, 딕셔너리)을 포함할 수 있는 딕셔너리를 마크다운으로 변환합니다."""
    if data is None or not data:  # None 또는 빈 딕셔너리
        return f"- **{title}:** 비어있음\n"

    md = f"- **{title}:**\n"
    for key, value in data.items():
        if isinstance(value, list):
            if all(isinstance(i, str) for i in value):  # 문자열 리스트인 경우
                md += f"{item_prefix}**`{key}`** ({len(value)}개):\n"
                for i, val_item in enumerate(value[:3]):  # 최대 3개 표시
                    md += f"{item_prefix}  - {val_item}\n"
                if len(value) > 3: md += f"{item_prefix}  - *(... 외 {len(value) - 3}개)*\n"
            elif all(isinstance(i, dict) for i in value):  # 딕셔너리 리스트인 경우 (간략히)
                md += f"{item_prefix}**`{key}`** ({len(value)}개 항목 리스트):\n"
                for i, dict_item in enumerate(value[:2]):  # 최대 2개 딕셔너리 표시
                    if 'title_concept' in dict_item: # FinalSatiricalIdea 등 특정 키
                        md += f"{item_prefix}  - Item {i + 1}: **{dict_item['title_concept']}** - {summarize_text(dict_item.get('detailed_content', ''))}\n"
                    elif 'title' in dict_item and 'url' in dict_item: # RawArticle 등 특정 키
                        md += f"{item_prefix}  - Item {i + 1}: [{summarize_text(dict_item.get('title',''))}]({dict_item.get('url')})\n"
                    else:
                        md += f"{item_prefix}  - Item {i + 1}: {list(dict_item.keys())[:2]}...\n"  # 키 예시
                if len(value) > 2: md += f"{item_prefix}  - *(... 외 {len(value) - 2}개 항목)*\n"
            else:  # 혼합 리스트 또는 기타
                md += f"{item_prefix}**`{key}`**: (리스트 내용) {summarize_text(str(value))}\n"
        elif isinstance(value, dict):  # 중첩 딕셔너리
            md += f"{item_prefix}**`{key}`** (딕셔너리):\n"
            for sub_key, sub_value in list(value.items())[:3]:  # 최대 3개 키-값 표시
                md += f"{item_prefix}  - `{sub_key}`: {summarize_text(str(sub_value))}\n"
            if len(value) > 3: md += f"{item_prefix}  - *(... 외 {len(value) - 3}개 키)*\n"
        else:  # 단순 값
            md += f"{item_prefix}**`{key}`**: {summarize_text(_format_optional(value))}\n"
    return md


# --- 섹션별 포매팅 함수 (state_v2.py 변경사항 반영) ---

def format_meta_section(meta: MetaSection) -> str:
    md = "## 📊 메타 (Meta)\n"
    md += f"- **Work ID:** `{_format_optional(meta.work_id)}`\n"
    md += f"- **Timestamp:** {_format_optional(meta.timestamp)}\n"
    status_items = []
    if meta.workflow_status:
        for order, status in sorted(meta.workflow_status.items()):
            status_items.append(f"노드 {order}: `{status}`")
    md += f"- **워크플로우 상태:** {', '.join(status_items) if status_items else 'N/A'}\n"
    md += f"- **다음 액션 (라우터용):** `{_format_optional(meta.next_action)}`\n"

    # if meta.llm_think_traces:
    #     md += f"- **LLM Think Traces ({len(meta.llm_think_traces)}개):**\n"
    #     for i, trace in enumerate(meta.llm_think_traces):  # 최근 2개
    #         md += f"  - `{trace.get('node_name', 'UnknownNode')}` (ReqID: `{trace.get('request_id', 'N/A')}`): {summarize_text(trace.get('log_content'))}\n"
    #     md += "- **LLM Think Traces:** 없음\n"
    return md


def format_query_section(query: QuerySection) -> str:
    md = "## ❓ 쿼리 (Query)\n"
    md += f"- **원본 쿼리:** {_format_optional(query.original_query)}\n"
    md += _format_list_of_strings(query.search_target_site_domain, "우선 검색 도메인")
    # initial_context_results는 Dict[str, Any] 리스트이므로, _format_list_of_dicts 사용. RawArticle과 유사할 수 있음.
    md += _format_list_of_dicts(query.initial_context_results, "초기 컨텍스트 검색 결과(N02)")
    md += _format_dict_complex(query.query_context, "쿼리 컨텍스트(N02, N03 분석 결과)")
    md += f"- **N02 LLM 분석 상세(한글):**\n```text\n{summarize_text(query.llm_analysis_details_korean)}\n```\n"
    md += f"- **N03 LLM 검색계획 상세(한글):**\n```text\n{summarize_text(query.llm_analysis_details_searchplan_korean)}\n```\n"
    return md


def format_search_section(search: SearchSection) -> str:
    md = "## 🔍 검색 (Search)\n"
    if search.search_strategy:
        strategy = search.search_strategy
        md += "- **검색 전략 (N03):**\n"
        md += f"  - 작가 컨셉: `{_format_optional(strategy.get('writer_concept'))}`\n"
        md += _format_list_of_strings(strategy.get('selected_tools'), "선택된 도구", item_prefix="    - ")
        md += _format_list_of_strings(strategy.get('queries'), "일반 검색 쿼리", item_prefix="    - ")
        md += f"  - **타겟 시드 도메인 ({strategy.get('seed_domain_count')}개):**\n"
        md += _format_list_of_strings(strategy.get('target_seed_domains'), "타겟 시드 도메인", item_prefix="    - ")
        md += f"  - 파라미터: `{_format_optional(strategy.get('parameters'))}`\n"
    else:
        md += "- **검색 전략 (N03):** 수립되지 않음\n"
    # raw_search_results는 List[RawArticle] 이므로 _format_list_of_dicts가 RawArticle 특별 처리 로직 사용
    md += _format_list_of_dicts(search.raw_search_results, "수집된 원본 검색 결과 (N04)")
    return md


def format_report_section(report: ReportSection) -> str:
    md = "## 📄 리포트 (Report)\n"
    report_summary = summarize_html_as_text(report.report_content)
    md += f"- **HTML 보고서 내용 (N05 통합):** {report_summary}\n"

    # 기존 ReportSection 필드들
    if report.referenced_urls_for_report and (
            report.referenced_urls_for_report.get("used") or report.referenced_urls_for_report.get("not_used")):
        md += "- **보고서 참조 URL (N05 통합):**\n"
        # referenced_urls_for_report의 값은 List[RawArticle] 이므로 _format_list_of_dicts가 RawArticle 특별 처리
        md += _format_list_of_dicts(report.referenced_urls_for_report.get("used"), "사용된 URL",
                                    item_prefix="    - ")
        md += _format_list_of_dicts(report.referenced_urls_for_report.get("not_used"), "미사용 URL",
                                    item_prefix="    - ")
    else:
        md += "- **보고서 참조 URL (N05 통합):** 없음\n"

    md += f"- **문맥적 요약 (N05 통합):**\n```text\n{summarize_text(report.contextual_summary)}\n```\n"
    md += f"- **저장된 보고서 경로 (N05 통합):** `{_format_optional(report.saved_report_path)}`\n"
    md += f"- **LLM 보고서 구조화 상세(한글, N05 통합):**\n```text\n{summarize_text(report.llm_report_structuring_details_korean)}\n```\n"
    return md


def format_idea_section(idea: IdeaSection) -> str:
    md = "## 💡 아이디어 (Idea)\n"
    # N06 시리즈 결과 (IdeaSection 필드 기준)
    md += _format_dict_complex(idea.structured_issue_analysis, "구조화된 이슈 분석 (N06, Idea)")

    if idea.scraped_community_reactions:
        md += "- **수집된 커뮤니티 반응 (N06A, Idea):**\n"
        for platform, reactions in idea.scraped_community_reactions.items():
            md += _format_list_of_strings(reactions, platform, item_prefix=f"    - {platform}: ")
    else:
        md += "- **수집된 커뮤니티 반응 (N06A, Idea):** 없음\n"

    md += _format_dict_complex(idea.community_reaction_analysis, "커뮤니티 반응 분석 결과 (N06B, Idea)")

    # IdeaSection.community_reactions (state_v2.py에 명시된 필드)
    if idea.community_reactions:
        md += "- **커뮤니티 반응 (종합, Idea):**\n" # 필드의 정확한 의미에 따라 제목 변경 가능
        for platform, reactions_list_of_dict in idea.community_reactions.items():
            md += f"  - **{platform.upper()} 반응 ({len(reactions_list_of_dict)}개):**\n"
            for i, reaction_dict in enumerate(reactions_list_of_dict):
                type_val = reaction_dict.get('type', 'N/A')
                content_val = reaction_dict.get('content', reaction_dict.get('description', '내용 없음'))
                md += f"    - {i + 1}. [{type_val}] {summarize_text(content_val)}\n"
    else:
        md += "- **커뮤니티 반응 (종합, Idea):** 없음\n"

    if idea.generated_satirical_reactions: # N06C
        md += "- **생성된 풍자 반응 (N06C, Idea):**\n"
        for platform, reactions_list_of_dict in idea.generated_satirical_reactions.items():
            md += f"  - **{platform.upper()} 반응 ({len(reactions_list_of_dict)}개):**\n"
            for i, reaction_dict in enumerate(reactions_list_of_dict):
                type_val = reaction_dict.get('type', 'N/A')
                content_korean_val = reaction_dict.get('content_korean', '내용 없음')
                md += f"    - {i + 1}. [{type_val}] {summarize_text(content_korean_val)}\n"
    else:
        md += "- **생성된 풍자 반응 (N06C, Idea):** 없음\n"

    # N07 결과 (final_comic_ideas)
    # display_keys를 제거하여 _format_list_of_dicts 내의 FinalSatiricalIdea 특별 처리 로직이 발동하도록 함
    md += _format_list_of_dicts(
        idea.final_comic_ideas, # type: ignore # TypedDict 호환성 (실제로는 List[FinalSatiricalIdea])
        "최종 선정 코믹 아이디어 (N07)"
    )
    # 기존 comic_ideas 필드는 N07의 새로운 출력 구조로 대체되었으므로 주석 처리 또는 제거
    # md += _format_list_of_dicts(idea.comic_ideas, "기존 코믹 아이디어 (사용 중단 가능성)")
    return md


def format_scenario_section(scenario: ScenarioSection) -> str:
    md = "## 📜 시나리오 (Scenario)\n"
    md += f"- **선택된 아이디어 인덱스 (N07 결과 중):** {_format_optional(scenario.selected_comic_idea_for_scenario)}\n"
    md += f"- **시나리오 원본 컨셉 (N07 아이디어 제목):** {_format_optional(scenario.comic_scenario_thumbnail)}\n"

    if scenario.comic_scenarios:
        md += f"- **코믹 시나리오 패널 ({len(scenario.comic_scenarios)}개 - N08):**\n"
        for i, panel_dict in enumerate(scenario.comic_scenarios):
            if isinstance(panel_dict, dict):
                md += f"  - **Panel {panel_dict.get('panel_number', i + 1)} ({panel_dict.get('scene_identifier', 'N/A')}):**\n"
                md += f"    - 설정: {summarize_text(panel_dict.get('setting'))}\n"
                md += f"    - 캐릭터: {summarize_text(panel_dict.get('characters_present'))}\n"
                md += f"    - 주요 액션: {summarize_text(panel_dict.get('key_actions_or_events'))}\n"
                md += f"    - 최종 프롬프트 (N08a 입력): {summarize_text(panel_dict.get('final_image_prompt'))}\n"
            else:
                md += f"  - Panel {i + 1}: (올바르지 않은 형식)\n"
    else:
        md += "- **코믹 시나리오 패널 (N08):** 생성되지 않음\n"
    return md


def format_image_section(image: ImageSection) -> str:
    md = "## 🖼️ 이미지 (Image)\n"
    md += _format_list_of_dicts(
        image.refined_prompts,
        "정제된 이미지 프롬프트 (N08a)",
        display_keys=['scene_identifier', 'model_name', 'prompt_used']
    )
    md += _format_list_of_dicts(
        image.generated_comic_images,
        "생성된 코믹 이미지 URL/경로 (N09)",
        display_keys=['scene_identifier', 'image_url', 'error']
    )
    return md


def format_upload_section(upload: UploadSection) -> str:
    md = "## 📤 업로드 (Upload)\n"
    md += _format_list_of_dicts(upload.uploaded_image_urls, "업로드된 이미지 URL", display_keys=['scene_identifier', 'url'])
    md += f"- **업로드된 리포트 S3 URI:** `{_format_optional(upload.uploaded_report_s3_uri)}`\n"
    return md


def format_config_section(config: ConfigSection) -> str:
    md = "## ⚙️ 설정 (Config)\n"
    md += _format_dict_complex(config.config, "워크플로우 설정값")
    return md


def format_scratchpad_section(scratchpad: Optional[Dict[str, Any]]) -> str:
    md = "## 📝 스크래치패드 (Scratchpad)\n"
    if not scratchpad:
        md += "- 내용 없음\n"
        return md
    md += _format_dict_complex(scratchpad, "임시 데이터")
    return md


def format_workflow_state_to_markdown(state: WorkflowState) -> str:
    """WorkflowState 객체를 전체 마크다운 문자열로 변환합니다."""
    if not isinstance(state, WorkflowState):
        return "오류: 입력값이 WorkflowState 객체가 아닙니다."

    md_parts = [
        format_meta_section(state.meta),
        format_query_section(state.query),
        format_search_section(state.search),
        format_report_section(state.report),
        format_idea_section(state.idea),
        format_scenario_section(state.scenario),
        format_image_section(state.image),
        format_upload_section(state.upload),
        format_config_section(state.config),
        format_scratchpad_section(state.scratchpad),
    ]

    return "\n\n---\n\n".join(md_parts)

# summarize_text 함수는 헬퍼 함수들 위로 이동시킴.
# def summarize_text(text: Optional[Any], max_len: int = 50) -> str:
#     """None을 처리하고, 텍스트를 주어진 길이로 요약하며 ...를 붙입니다."""
#     if text is None:
#         return "N/A"
#     text_str = str(text)
#     if len(text_str) > max_len:
#         return text_str[:max_len - 3] + "..."
#     return text_str