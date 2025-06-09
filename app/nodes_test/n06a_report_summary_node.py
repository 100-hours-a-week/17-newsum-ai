import json
from typing import List
from app.workflows.state_v2 import WorkflowState
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json
from app.utils.logger import get_logger

logger = get_logger(__name__)

class N06aReportSummaryNode:
    """
    노드 역할:
      - state.report.report_content(보고서 전문 HTML) 를 입력받아
      - 핵심 인사이트 3~5개 bullet-point 형태로 요약하여
      - state.report.contextual_summary 필드에 JSON list로 저장
      - 이후 n07_simple_scenario 노드로 연결
    사용 필드:
      - 입력: state.report.report_content (str)
      - 출력: state.report.contextual_summary (List[str])
      - 메타 : state.meta.current_stage
    """
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        logger.info("N06aReportSummaryNode initialized.")

    def _build_system_prompt(self) -> str:
        schema_example = {
            "contextual_summary": [
                "<핵심 인사이트 1>",
                "<핵심 인사이트 2>",
                "<핵심 인사이트 3>"
            ]
        }
        json_schema = json.dumps(schema_example, ensure_ascii=False, indent=2)

        return """
You are an expert news summarizer.

Given the full report HTML content, extract 3–5 key insights as bullet points.
Each insight should be concise (20–40 Korean words) and capture the report's essence.
Return only valid JSON matching this schema (no extra text or markdown):

{json_schema}

## Few-Shot Examples

### Example 1
Report Excerpt:
"EU 발표 ... 편향 규제 초안 ... 업계 반발 ... 공청회 ..."
Output:
```json
{
  "contextual_summary": [
    "EU의 AI 편향 규제 초안 발표",
    "기술업계의 실현 가능성 우려 표명",
    "공청회에서 시민들의 윤리적 위험 지적",
    "수정안 통해 균형 있는 접근 모색"
  ]
}
```

### Example 2
Report Excerpt:
"기업, 멀티모달 LLM 시연 ... 텍스트·이미지·음성 통합 ... 관중 반응 ... 개인정보 이슈 Q&A"
Output:
```json
{
  "contextual_summary": [
    "기업의 멀티모달 LLM 시연 성공",
    "텍스트·이미지·음성 실시간 통합 기능 시연",
    "관중의 놀라운 반응과 질문 열기",
    "개인정보 보호 관련 Q&A 진행"
  ]
}
```
""".replace('{json_schema}', json_schema)

    async def run(self, state: WorkflowState) -> dict:
        meta = state.meta
        report_html = state.report.report_content or ""

        system_prompt = self._build_system_prompt()
        user_prompt = report_html[:8000]

        logger.info("N06a: Summarizing report to contextual_summary", extra={"trace_id": meta.trace_id})
        response = await self.llm_service.generate_text(
            system_prompt_content=system_prompt,
            prompt=user_prompt,
            temperature=0.1,
            max_tokens=1200
        )
        parsed = extract_json(response.get("generated_text", ""))
        insights = parsed.get("contextual_summary", [])

        state.report.contextual_summary = insights
        meta.current_stage = "n07_simple_scenario"
        state.meta = meta

        return {
            "report": state.report.model_dump(),
            "meta": state.meta.model_dump()
        }
