import json
from typing import List
from app.services.llm_service import LLMService
from app.workflows.state_v2 import WorkflowState
from app.tools.think_parser_tool import extract_json_all
from app.api.v2.schemas.nodes.search_plan import SearchPlanBatch, FrameSearchPlan
from pydantic import ValidationError
from app.nodes_v2.site_domain import CATEGORY_DOMAINS, KOREAN_DOMAINS, ENGLISH_DOMAINS

class SearchPlannerNode:
    def __init__(self, llm: LLMService):
        self.llm = llm

    def build_prompt(
        self,
        refined_intent: str,
        frames: List[dict],
        category: str,
        user_feedback: str | None
    ) -> str:
        domain_hint = ", ".join(CATEGORY_DOMAINS.get(category or "Politics", []))
        frames_dicts = [f.dict() if hasattr(f, "dict") else f for f in frames]
        frames_json = json.dumps(frames_dicts, ensure_ascii=False, indent=2)

        feedback_block = f"\n## User Feedback\n{user_feedback}" if user_feedback else ""

        return f"""You are a Search Planner for a satirical 4-panel comic workflow.

## Refined Intent
{refined_intent}

## Frames
{frames_json}{feedback_block}

## Task
For each frame, return a JSON object:
{{
  "index": <frame index>,
  "title": <same title>,
  "purpose": "explanation|conflict|punchline",
  "queries_ko": ["한글 키워드1", "한글 키워드2", ...],  // 2-4개, 반드시 한글
  "queries_en": ["English keyword1", "English keyword2", ...],  // 2-4개, 반드시 영어
  "tool": "news|site|community|youtube|blog|web",  // 반드시 프레임 목적에 따라 다양하게 선택
  "domains": ["domain1.com", ...],           // choose mainly from: {domain_hint} (ex: community 프레임은 커뮤니티 도메인에서만 선택)
  "max_results": 5
}}
- For each frame, always provide both Korean and English queries (queries_ko, queries_en).
- At least one frame should use "community" as the tool and select domains from community sites (e.g., 9gag.com, reddit.com, imgur.com, knowyourmeme.com, memedroid.com).
Return a JSON list ONLY, NO think, NO explanation, NO comment."""

    @staticmethod
    def is_english_domain(domain: str) -> bool:
        return domain in ENGLISH_DOMAINS
        
    async def run(self, state: WorkflowState) -> WorkflowState:
        prompt = self.build_prompt(
            refined_intent=state.query.refined_intent,
            frames=state.query.frames,
            category=state.query.category,
            user_feedback=state.search.user_feedback,
        )

        raw = ( await self.llm.generate_text(
            system_prompt_content=prompt,
            prompt="Generate search plan.",
            temperature=0.4,
            max_tokens=2500,
        ) ).get("generated_text", "")
        print("RAW")
        print(raw)

        parsed = extract_json_all(raw)
        # 후처리: 도메인 언어에 따라 queries를 선택
        processed_plans = []
        for plan in parsed:
            # domains가 모두 영문이면 영어 쿼리, 모두 한글이면 한글 쿼리, 혼합이면 우선 영어 (개선 필요)
            domains = plan.get("domains", [])
            if domains and all(self.is_english_domain(d) for d in domains):
                queries = plan.get("queries_en", [])
            else:
                queries = plan.get("queries_ko", [])
            processed = {
                "index": plan["index"],
                "title": plan["title"],
                "purpose": plan["purpose"],
                "queries": queries,
                "tool": plan["tool"],
                "domains": plan["domains"],
                "max_results": plan.get("max_results", 5)
            }
            processed_plans.append(processed)
        try:
            batch = SearchPlanBatch(**{"plans": processed_plans})
        except ValidationError as e:
            raise RuntimeError(f"SearchPlan schema validation failed: {e}")

        state.search.search_plan = [p.model_dump() for p in batch.plans]
        return state
