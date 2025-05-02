# tests/run_imager_test.py
import sys
import os
import asyncio

# 프로젝트 루트를 import path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.workflows.state import ComicState

async def main():
    workflow = build_imager_test_workflow()
    state = ComicState(
        lora_style="Studio Ghibli style",
    )

    final_state = await workflow.ainvoke(state)

    # 시나리오 확인
    scenarios = final_state.get("scenarios", [])
    print("\n✅ 생성된 시나리오 결과:")
    if not scenarios:
        print("⚠️ scenario 항목이 비어있습니다. 에이전트에서 결과가 정상적으로 저장되지 않았을 수 있습니다.")

    for i, scene in enumerate(scenarios):
        print(f"\n📌 컷 {i+1}")
        print(f"Description: {scene.get('description')}")
        print(f"Dialogue   : {scene.get('dialogue')}")
        print(f"Prompt     : {scene.get("prompt")}")
        print()

    # 생성된 이미지 URL 확인
    image_urls = final_state.get("image_urls", [])
    print("\n✅ 생성된 이미지 URL:")
    if not image_urls:
        print("⚠️ image_urls 항목이 비어있습니다. 에이전트에서 결과가 정상적으로 저장되지 않았을 수 있습니다.")
    for i, url in enumerate(image_urls):
        print(f"컷 {i+1}: {url}")
    print("\n\n")

    # print("🧪 최종 상태 전체:", final_state.dict() if hasattr(final_state, "dict") else final_state)





# test workflow
from langgraph.graph import StateGraph, END
from app.workflows.state import ComicState
from app.agents.test_entry import TestEntryAgent
from app.agents.scenariowriter_agent import ScenarioWriterAgent
from app.agents.imager_agent import ImagerAgent

def build_imager_test_workflow() -> StateGraph:
    """
    시나리오 → 이미지 테스트용 최소 LangGraph 워크플로우
    """
    graph = StateGraph(ComicState)

    graph.add_node("test_entry", TestEntryAgent().run)
    graph.add_node("scenario", ScenarioWriterAgent().run)
    graph.add_node("imager", ImagerAgent().run)

    graph.set_entry_point("test_entry")
    graph.add_edge("test_entry", "scenario")
    graph.add_edge("scenario", "imager")
    graph.add_edge("imager", END)

    return graph.compile()



# 이미지 모델 테스트는 colab에 수동으로 테스트 (ngrok 동시 사용 불가로 인함)
if __name__ == "__main__":
    asyncio.run(main())