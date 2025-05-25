# import asyncio
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# from app.nodes_v2.multi_search_node import MultiSearchNode
# from app.api.v2.schemas.nodes.search_frame import FrameSearchPlan
# from app.workflows.state_v2 import WorkflowState

# async def run_test():
#     state = WorkflowState()
#     state.search.search_plan = [
#         FrameSearchPlan(
#             index=0,
#             title="보험사의 압박",
#             purpose="conflict",
#             queries=["보험 보상 지연"],
#             domains=["hankyung.com", "etnews.com"],
#             tool="web"
#         ),
#         FrameSearchPlan(
#             index=1,
#             title="밈 반응 보기",
#             purpose="punchline",
#             queries=["보험 관련 밈"],
#             domains=["9gag.com", "reddit.com"],
#             tool="community"
#         ),
#         FrameSearchPlan(
#             index=2,
#             title="기초 뉴스",
#             purpose="explanation",
#             queries=["보험 사기 뉴스"],
#             domains=["reuters.com", "nytimes.com"],
#             tool="news"
#         )
#     ]

#     node = MultiSearchNode()
#     result_state = await node.run(state)

#     for res in result_state.search.search_results:
#         print("-----")
#         print(f"Frame {res.frame_index}: {res.title} ({res.purpose})")
#         print(f"Docs Returned: {len(res.result_docs)}")
#         for doc in res.result_docs:
#             print(f"- {doc.title} ({doc.site})")
#             print(f"  ↪ {doc.snippet[:80]}...")



# if __name__ == "__main__":
#     asyncio.run(run_test())
