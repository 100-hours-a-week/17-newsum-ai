# state_view_utils.py

from state_v2 import * # 가정: 실제 경로

def _format_optional(value: Any, default: str = "N/A") -> str:
    """None 값을 처리하고 문자열로 변환합니다."""
    return str(value) if value is not None else default

def _format_list(items: Optional[List[Any]], title: str, max_items: int = 3) -> str:
    """리스트를 마크다운으로 변환합니다 (최대 N개 표시)."""
    if not items:
        return f"- **{title}:** 0개\n"

    md = f"- **{title}:** {len(items)}개\n"
    for i, item in enumerate(items[:max_items]):
        # 리스트 항목이 딕셔너리인 경우 간단히 표시 (필요시 커스텀)
        if isinstance(item, dict):
            # RawArticle 특별 처리
            if 'title' in item and 'url' in item:
                 md += f"  - {i+1}. [{item.get('title')}]({item.get('url')}) (Rank: {item.get('rank')})\n"
            else:
                 md += f"  - {i+1}. {str(item)[:80]}...\n" # 딕셔너리를 간단히 문자열로 표시
        else:
            md += f"  - {i+1}. {str(item)}\n"
    if len(items) > max_items:
        md += f"  - *(... 외 {len(items) - max_items}개)*\n"
    return md

def _format_dict(data: Optional[Dict[str, Any]], title: str) -> str:
    """딕셔너리를 마크다운으로 변환합니다."""
    if not data:
        return f"- **{title}:** 비어있음\n"
    md = f"- **{title}:**\n"
    for key, value in data.items():
        md += f"  - `{key}`: {_format_optional(value)}\n"
    return md

# --- 섹션별 포매팅 함수 ---

def format_meta_section(meta: MetaSection) -> str:
    md = "## 📊 메타 (Meta)\n"
    md += f"- **상태:** `{_format_optional(meta.workflow_status)}`\n"
    md += f"- **단계:** `{_format_optional(meta.current_stage)}`\n"
    md += f"- **Trace ID:** `{_format_optional(meta.trace_id)}`\n"
    md += f"- **Comic ID:** `{_format_optional(meta.comic_id)}`\n"
    md += f"- **Timestamp:** {meta.timestamp}\n"
    md += f"- **재시도:** {meta.retry_count}회\n"
    if meta.error_message:
        md += f"- **오류:** <span style='color:red;'>{meta.error_message}</span>\n"
    md += _format_list(meta.error_log, "오류 로그")
    return md

def format_query_section(query: QuerySection) -> str:
    md = "## ❓ 쿼리 (Query)\n"
    md += f"- **원본 쿼리:** {query.original_query or 'N/A'}\n"
    md += _format_dict(query.query_context, "쿼리 컨텍스트")
    md += _format_list(query.initial_context_results, "초기 컨텍스트 결과")
    return md

def format_search_section(search: SearchSection) -> str:
    md = "## 🔍 검색 (Search)\n"
    md += _format_dict(search.search_strategy, "검색 전략")
    md += _format_list(search.raw_search_results, "원시 검색 결과 (기사)")
    return md

def format_report_section(report: ReportSection) -> str:
    md = "## 📄 리포트 (Report)\n"
    if report.report_content:
        md += f"- **리포트 내용:**\n```\n{report.report_content[:150]}...\n```\n"
    else:
        md += "- **리포트 내용:** N/A\n"
    md += _format_list(report.referenced_urls_for_report, "참고 URL")
    md += f"- **HITL 상태:** {_format_optional(report.hitl_status)}\n"
    return md

def format_idea_section(idea: IdeaSection) -> str:
    md = "## 💡 아이디어 (Idea)\n"
    md += _format_list(idea.comic_ideas, "코믹 아이디어")
    return md

def format_scenario_section(scenario: ScenarioSection) -> str:
    md = "## 📜 시나리오 (Scenario)\n"
    md += _format_dict(scenario.selected_comic_idea_for_scenario, "선택된 아이디어")
    md += _format_list(scenario.comic_scenarios, "코믹 시나리오")
    md += f"- **썸네일 프롬프트:** {scenario.thumbnail_image_prompt or 'N/A'}\n"
    return md

def format_image_section(image: ImageSection) -> str:
    md = "## 🖼️ 이미지 (Image)\n"
    md += _format_list(image.generated_comic_images, "생성된 코믹 이미지")
    return md

def format_upload_section(upload: UploadSection) -> str:
    md = "## 📤 업로드 (Upload)\n"
    md += _format_list(upload.uploaded_image_urls, "업로드된 이미지 URL")
    md += f"- **리포트 S3 URI:** `{_format_optional(upload.uploaded_report_s3_uri)}`\n"
    return md

def format_config_section(config: ConfigSection) -> str:
    md = "## ⚙️ 설정 (Config)\n"
    md += _format_dict(config.config, "설정값")
    return md

def format_scratchpad_section(scratchpad: Dict[str, Any]) -> str:
    md = "## 📝 스크래치패드 (Scratchpad)\n"
    md += _format_dict(scratchpad, "임시 데이터")
    return md

# --- 메인 함수 ---

def format_workflow_state_to_markdown(state: WorkflowState) -> str:
    """WorkflowState 객체를 전체 마크다운 문자열로 변환합니다."""

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

    return "\n\n---\n\n".join(md_parts) # 각 섹션을 구분선으로 나눔