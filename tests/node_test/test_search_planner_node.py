import asyncio
import sys
import os

# 'tests/node_test/' 기준으로 루트 디렉토리로 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.nodes_v2.search_planner_node import SearchPlannerNode
from app.services.llm_service import LLMService
from app.workflows.state_v2 import WorkflowState
from app.api.v2.schemas.nodes.query_intent import FrameSchema

async def run_test():
    # 테스트용 초기 상태 설정
    state = WorkflowState()
    state.query.refined_intent = "미국 중서부 폭풍 피해 풍자 만화 기획"
    state.query.category = "Economy"
    state.query.frames = [
        FrameSchema(
            title="보험사의 압박",
            purpose="conflict",
            search_terms=["보험 보상 지연", "피해자와 보험사 갈등"],
            preferred_sources=["hankyung.com", "etnews.com"]
        ).model_dump(),
        FrameSchema(
            title="경제의 파동",
            purpose="punchline",
            search_terms=["지역 경제 회복", "정부 지원 미흡"],
            preferred_sources=["9gag.com", "imgur.com"]
        ).model_dump(),
        FrameSchema(
            title="자연의 분노",
            purpose="explanation",
            search_terms=["미국 중서부 폭풍 피해", "자연 재해 경제 영향"],
            preferred_sources=["reuters.com", "ft.com"]
        ).model_dump(),
    ]

    llm = LLMService()  # 실제 LLM 서버와 연결
    node = SearchPlannerNode(llm=llm)
    result_state = await node.run(state)

    # 결과 출력
    print("✅ Search Plan:")
    for plan in result_state.search.search_plan:
        print("-----")
        print("Frame Index:", plan["index"])
        print("Title:", plan["title"])
        print("Purpose:", plan["purpose"])
        print("Tool:", plan["tool"])
        print("Queries:", plan["queries"])
        print("Domains:", plan["domains"])

if __name__ == "__main__":
    asyncio.run(run_test()) 