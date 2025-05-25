import asyncio
import sys
import os
# 'tests/node_test/' 기준으로 루트 디렉토리로 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.workflows.v2_workflow import full_workflow
from app.services.llm_service import LLMService
from app.services.image_service import ImageService
from app.services.storage_service import StorageService
from app.services.translation_service import TranslationService
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.workflows.state_v2 import WorkflowState


async def run_test():
    # 초기 서비스 객체 생성
    llm = LLMService()
    image_service = ImageService()
    storage_service = StorageService()
    translation_service = TranslationService()
    search_tool = GoogleSearchTool()

    # 초기 상태 설정
    state = WorkflowState()
    state.query.original_query = "최근 미국 중서부 지역에서 발생한 폭풍 피해 상황을 풍자 만화로 만들어주세요."

    # 워크플로우 실행
    graph = await full_workflow(
        llm_service=llm,
        google_search_tool=search_tool,
        image_generation_service=image_service,
        storage_service=storage_service,
        translation_service=translation_service,
    )
    result = await graph.ainvoke(state)

    # 결과 꺼내기
    query = result["query"]

    # 결과 출력
    print("✅ Refined Intent:", query.refined_intent)
    print("✅ Search Plan:")
    for plan in query.search_plan:
        print(f" - [{plan.purpose}] {plan.title} → {plan.tool} → {plan.queries}")

    print("✅ Search Results:")
    for res in query.search_results:
        print("-----")
        print(f"Frame {res.frame_index}: {res.title} ({res.purpose})")
        print(f"Docs Returned: {len(res.result_docs)}")
        for doc in res.result_docs:
            print(f"- {doc.title} ({doc.site})")
            print(f"  ↪ {doc.snippet[:200]}...")

    print("--------------------------------")
    print("🚀 Workflow State:")
    print("  - Query Original Query:", state.query.original_query)
    print("  - Query Refined Intent:", query.refined_intent)
    print("  - Search Plan:")
    for plan in query.search_plan:
        print(f"    - [{plan.purpose}] {plan.title} → {plan.tool} → {plan.queries}")
    print("  - Search Results:")
    for res in query.search_results:
        print(f"    - Frame {res.frame_index} ({res.title}): {len(res.result_docs)} docs")
    
    # 세션 종료
    await search_tool._session.close()


if __name__ == "__main__":
    asyncio.run(run_test())