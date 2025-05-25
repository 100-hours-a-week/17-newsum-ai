import asyncio
import sys
import os
# 'tests/node_test/' 기준으로 루트 디렉토리로 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.nodes_v2.query_intent_node import QueryIntentNode
from app.services.llm_service import LLMService
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.workflows.state_v2 import WorkflowState


async def run_test():
    # 테스트용 초기 상태 설정
    state = WorkflowState()
    state.query.original_query = "최근 미국 중서부 지역에서 발생한 폭풍 피해 상황을 풍자 만화로 만들어주세요."
    #"요즘 트럼프가 AI 규제를 푼다던데, 이를 풍자 만화로 만들면 재미있을까요?"

    # LLM과 검색 툴 초기화 (검색 툴은 현재 사용되지 않지만 구조상 필요)
    llm = LLMService()  # 실제 Qwen3 서버 주소에 맞게 수정
    search = GoogleSearchTool()

    node = QueryIntentNode(llm=llm)
    result_state = await node.run(state)

    # 결과 출력
    print("✅ category:", result_state.query.category)
    print("✅ refined_intent:", result_state.query.refined_intent)
    print("✅ frames:")
    for f in result_state.query.frames:
        print(" -", f)

if __name__ == "__main__":
    asyncio.run(run_test())
