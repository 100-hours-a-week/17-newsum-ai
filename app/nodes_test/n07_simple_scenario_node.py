# ai/app/nodes_v2/n07_simple_scenario_node.py

import json
from typing import List, Dict, Any
from app.workflows.state_v2 import WorkflowState
from app.services.llm_service import LLMService
from app.tools.think_parser_tool import extract_json
from app.utils.logger import get_logger

logger = get_logger(__name__)

class N07SimpleScenarioNode:
    """
    n06a에서 만들어진 state.report.contextual_summary (List[str])를 받아,
    Hook→Development→Climax→Resolution 구조의 simple_scenario를 생성합니다.
    출력: state.scenario.simple_scenario = {
        "thumbnail_brief": str,
        "panels": List[str] (4 items)
    }
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        logger.info("N07SimpleScenarioNode initialized.")

    def _build_system_prompt(self) -> str:
        schema_example = {
            "thumbnail_brief": "<string>",
            "panels": ["<string>", "<string>", "<string>", "<string>"]
        }
        json_schema = json.dumps(schema_example, ensure_ascii=False, indent=2)

        return f"""You are a professional webtoon scenario writer.

Given a JSON array of 3–5 core insights (key takeaways from a news report), generate a simple 4-panel scenario in Korean.
Use a 4-beat structure: Hook → Development → Climax → Resolution.
Return _only_ valid JSON matching this schema (no extra text or markdown):

{json_schema}

## Few-Shot Example
Input (insights):
["AI 편향 규제 초안 발표", "기술업계의 우려 표명", "공청회 열띤 토론", "최종 합의안 도출"]

Output:
```json
{{
  "thumbnail_brief": "AI 규제 갈등 현장",
  "panels": [
    "브뤼셀 의사당 앞 정책 발표",
    "업계 대표의 실현 가능성 질문",
    "공청회에서의 열띤 토론",
    "타협안 서명 후 악수 장면"
  ]
}}
```
"""


    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta = state.meta
        report_sec = state.report
        scenario_sec = state.scenario

        # N06a에서 만들어진 리스트
        insights: List[str] = report_sec.contextual_summary or []
        if not insights:
            msg = "No contextual_summary available for simple scenario."
            logger.error(msg, extra={"trace_id": meta.trace_id})
            meta.current_stage = "ERROR"
            meta.error_message = msg
            return {"scenario": scenario_sec.model_dump(), "meta": meta.model_dump()}

        system_prompt = self._build_system_prompt()
        user_input = json.dumps(insights, ensure_ascii=False)

        logger.info("N07: Generating simple scenario from insights.", extra={"trace_id": meta.trace_id})
        response = await self.llm_service.generate_text(
            system_prompt_content=system_prompt,
            prompt=user_input,
            temperature=0.3,
            max_tokens=1200
        )
        raw = response.get("generated_text", "")
        parsed = extract_json(raw)

        # simple_scenario 구조 저장
        scenario_sec.simple_scenario = parsed
        meta.current_stage = "n08_scenario_generation"
        state.scenario = scenario_sec
        state.meta = meta

        return {
            "scenario": scenario_sec.model_dump(),
            "meta": meta.model_dump()
        }
