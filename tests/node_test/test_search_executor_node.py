import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.nodes_v2.search_executor_node import SearchExecutorNode
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.workflows.state_v2 import WorkflowState

async def run_test():
    state = WorkflowState()
    state.search.search_plan = [
        {
            "index": 0,
            "title": "보험사의 압박",
            "purpose": "conflict",
            "queries": ["보험 보상 지연"],
            "domains": ["hankyung.com", "etnews.com"],
            "tool": "web"
        },
        {
            "index": 1,
            "title": "밈 반응 보기",
            "purpose": "punchline",
            "queries": ["보험 관련 밈"],
            "domains": ["9gag.com", "reddit.com"],
            "tool": "community"
        },
        {
            "index": 2,
            "title": "기초 뉴스",
            "purpose": "explanation",
            "queries": ["보험 사기 뉴스"],
            "domains": ["reuters.com", "nytimes.com"],
            "tool": "news"
        }
    ]

    search_tool = GoogleSearchTool()
    node = SearchExecutorNode(search_tool=search_tool)
    result_state = await node.run(state)

    print("✅ Search Results:")
    for res in result_state.search.search_results:
        print("-----")
        print(f"Frame {res['index']}: {res['title']}")
        print(f"Docs Returned: {len(res['docs'])}")
        for doc in res['docs']:
            print(f"- {doc.get('title', '')} ({doc.get('site', doc.get('source', ''))})")
            print(f"  ↪ {doc.get('snippet', '')[:80]}...")

if __name__ == "__main__":
    asyncio.run(run_test()) 