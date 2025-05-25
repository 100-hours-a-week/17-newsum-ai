import asyncio
import sys
import os

# 루트 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.nodes_v2.search_frame_node import SearchFrameNode
from app.api.v2.schemas.nodes.query_intent import FrameSchema
from app.workflows.state_v2 import WorkflowState


async def run_test():
    # 1. 테스트용 상태 초기화
    state = WorkflowState()
    state.query.frames = [
        FrameSchema(
            title="보험사의 압박",
            purpose="conflict",
            search_terms=["보험 보상 지연", "피해자와 보험사 갈등"],
            preferred_sources=["hankyung.com", "etnews.com"]
        ),
        FrameSchema(
            title="경제의 파동",
            purpose="punchline",
            search_terms=["지역 경제 회복", "정부 지원 미흡"],
            preferred_sources=["9gag.com", "imgur.com"]
        ),
        FrameSchema(
            title="자연의 분노",
            purpose="explanation",
            search_terms=["미국 중서부 폭풍 피해", "자연 재해 경제 영향"],
            preferred_sources=["reuters.com", "ft.com"]
        )
    ]

    # 2. 노드 실행
    node = SearchFrameNode()
    result_state = await node.run(state)

    # 3. 출력 결과 확인
    print("📦 Generated Search Plans:")
    for plan in result_state.query.search_plan:
        print("-----")
        print("Frame Index:", plan.index)
        print("Title:", plan.title)
        print("Purpose:", plan.purpose)
        print("Tool:", plan.tool)
        print("Queries:", plan.queries)
        print("Domains:", plan.domains)

if __name__ == "__main__":
    asyncio.run(run_test())
