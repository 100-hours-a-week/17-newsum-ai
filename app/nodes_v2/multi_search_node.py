# from __future__ import annotations

# from typing import List

# from app.workflows.state_v2 import WorkflowState
# from app.tools.search.Google_Search_tool import GoogleSearchTool
# from app.api.v2.schemas.nodes.search_frame import FrameSearchPlan
# from app.api.v2.schemas.nodes.multi_search import SearchDocument, validate_search_result, ValidationError


# class MultiSearchNode:
#     def __init__(self):
#         self.search_tool = GoogleSearchTool()

#     async def search_by_tool(self, plan: FrameSearchPlan) -> List[dict]:
#         """
#         FrameSearchPlan에 따라 적절한 검색 함수 호출
#         """
#         try:
#             params = dict(
#                 keyword=plan.queries[0],
#                 max_results=plan.max_results,
#                 trace_id=f"frame-{plan.index}"
#             )

#             if plan.tool == "news":
#                 return await self.search_tool.search_news_via_cse(**params)

#             elif plan.tool == "community":
#                 return await self.search_tool.search_communities_via_cse(**params)

#             elif plan.tool == "youtube":
#                 return await self.search_tool.search_youtube_videos(**params)

#             elif plan.tool == "site":
#                 return await self.search_tool.search_specific_sites_via_cse(
#                     sites=plan.domains,
#                     **params
#                 )

#             else:  # fallback to web
#                 return await self.search_tool.search_web_via_cse(**params)

#         except Exception as e:
#             print(f"[오류] frame-{plan.index} 검색 실패: {e}")
#             return []

#     async def run(self, state: WorkflowState) -> WorkflowState:
#         results = []

#         for plan in state.search.search_plan:
#             docs_raw = await self.search_by_tool(plan)
#             parsed_docs = []
#             for d in docs_raw:
#                 if "source" in d:
#                     d["site"] = d.pop("source")
#                 try:
#                     parsed_docs.append(SearchDocument(**d))
#                 except ValidationError as e:
#                     print(f"[경고] 문서 파싱 실패 (frame-{plan['index']}):", e)

#             result_dict = {
#                 "frame_index": plan["index"],
#                 "title": plan["title"],
#                 "purpose": plan["purpose"],
#                 "result_docs": parsed_docs
#             }
#             try:
#                 result = validate_search_result(result_dict)
#                 results.append(result)
#             except ValidationError as e:
#                 print(f"[오류] SearchResultSchema 파싱 실패 (frame-{plan['index']}):", e)

#         state.search.search_results = results
#         return state
