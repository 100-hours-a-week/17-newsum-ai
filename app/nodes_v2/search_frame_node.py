# from __future__ import annotations

# from urllib.parse import urlparse
# from typing import List
# from pydantic import BaseModel, Field

# from app.workflows.state_v2 import WorkflowState
# from app.api.v2.schemas.nodes.search_frame import FrameSearchPlan
# from app.api.v2.schemas.nodes.search_frame import validate_search_plan
# from pydantic import ValidationError


# # -------------------------------
# # SearchFrameNode
# # -------------------------------
# class SearchFrameNode:
#     def __init__(self):
#         self.domain_tool_map = {
#             "news": {
#                 "reuters.com", "bbc.com", "nytimes.com", "ft.com", "chosun.com",
#                 "joongang.co.kr", "hani.co.kr", "khan.co.kr", "ohmynews.com",
#                 "cnbc.com", "bloomberg.com", "mk.co.kr", "sedaily.com", "etoday.co.kr"
#             },
#             "web": {
#                 "techcrunch.com", "wired.com", "zdnet.com", "etnews.com",
#                 "arstechnica.com", "theverge.com", "itworld.co.kr", "bloter.net"
#             },
#             "community": {
#                 "9gag.com", "reddit.com", "memedroid.com", "knowyourmeme.com", "imgur.com"
#             },
#             "youtube": {"youtube.com"},
#             "site": set()  # fallback
#         }

#     def extract_base_domain(self, url: str) -> str:
#         """
#         도메인 정규화 (www 제거)
#         """
#         parsed = urlparse(f"https://{url}").netloc.lower()
#         return parsed.replace("www.", "") if parsed else url

#     def decide_tool(self, domains: List[str]) -> str:
#         """
#         domains 리스트에서 가장 많은 도메인 타입에 따라 검색 도구 결정
#         """
#         counts = {k: 0 for k in self.domain_tool_map}
#         for d in domains:
#             base = self.extract_base_domain(d)
#             for tool, domain_set in self.domain_tool_map.items():
#                 if base in domain_set:
#                     counts[tool] += 1
#         return max(counts, key=counts.get) if any(counts.values()) else "web"

#     def expand_terms(self, terms: List[str]) -> List[str]:
#         """
#         단순 중복 제거 및 정제
#         """
#         seen = set()
#         cleaned = []
#         for term in terms:
#             base = term.strip()
#             if base and base not in seen:
#                 seen.add(base)
#                 cleaned.append(base)
#         return cleaned

#     async def run(self, state: WorkflowState) -> WorkflowState:
#         """
#         각 프레임별로 검색 계획을 생성하고, 스키마 검증 후 state에 저장합니다.
#         """
#         plans: List[FrameSearchPlan] = []

#         for idx, frame in enumerate(state.query.frames):
#             queries = self.expand_terms(frame["search_terms"])
#             domains = [self.extract_base_domain(d) for d in frame["preferred_sources"]]
#             tool = self.decide_tool(domains)

#             plan_dict = {
#                 "index": idx,
#                 "title": frame["title"],
#                 "purpose": frame["purpose"],
#                 "queries": queries,
#                 "domains": domains,
#                 "tool": tool,
#                 "lang": "auto",  # 기본값
#                 "max_results": 5,  # 기본값
#             }

#             try:
#                 plan = validate_search_plan(plan_dict)
#             except ValidationError as e:
#                 raise RuntimeError(f"💥 FrameSearchPlan 스키마 검증 실패: {e}")

#             plans.append(plan)

#         state.search.search_plan = plans
#         return state
