import asyncio
from typing import Dict, Any
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.workflows.state_v2 import WorkflowState
from app.nodes_v2.site_domain import CATEGORY_DOMAINS, ALL_DOMAINS, CATEGORY_HINT, PURPOSE_HINT

class SearchExecutorNode:
    def __init__(self, search_tool: GoogleSearchTool):
        self.search_tool = search_tool

    async def _run_single(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        max_results = plan.get("max_results", 5)
        if isinstance(max_results, list):
            max_results = max_results[0] if max_results else 5
        print(type(max_results))
        docs = await self.search_tool.dispatch(
            plan["tool"],
            plan["queries"],
            plan["domains"],
            max_results=max_results
        )
        return {
            "index": plan["index"],
            "title": plan["title"],
            "purpose": plan.get("purpose", ""),
            "docs": docs
        }

    async def run(self, state: WorkflowState) -> WorkflowState:
        tasks = [self._run_single(p) for p in state.search.search_plan]

        for p in state.search.search_plan:
            print("🔍 queries:", p["queries"], "→", type(p["queries"]))
            print("🔍 domains:", p["domains"], "→", type(p["domains"]))
            print("🔍 max_results:", p["max_results"], "→", type(p["max_results"]))


        state.search.search_results = await asyncio.gather(*tasks)
        return state
