# app/workflows/main_workflow.py

from langgraph.graph import StateGraph, END
from app.workflows.state import ComicState
from app.agents.collector_agent import collect_news
from app.agents.scraper_agent import ScraperAgent
from app.agents.analyzer_agent import AnalyzerAgent
from app.agents.humorator_agent import HumoratorAgent
from app.agents.scenariowriter_agent import ScenarioWriterAgent
from app.agents.imager_agent import ImagerAgent
from app.agents.postprocessor_agent import PostProcessorAgent
from app.agents.translator_agent import TranslatorAgent
from app.workflows.transitions import determine_next_step

def build_main_workflow() -> StateGraph:
    """
    뉴스 수집부터 만화 완성까지 LangGraph 기반 플로우를 구성합니다.
    """
    graph = StateGraph(ComicState)

    # 노드 등록
    graph.add_node("collect", collect_news)

    # graph.add_node("scrape", ScraperAgent().run)
    # graph.add_node("analyze", AnalyzerAgent().run)
    # graph.add_node("humor", HumoratorAgent().run)
    # graph.add_node("scenario", ScenarioWriterAgent().run)
    # graph.add_node("image", ImagerAgent().run)
    # graph.add_node("postprocess", PostProcessorAgent().run)
    # graph.add_node("translate", TranslatorAgent().run)

    # 흐름 연결
    graph.set_entry_point("collect")
    graph.add_edge("collect", END)
    # graph.add_edge("collect", "scrape")
    # graph.add_edge("scrape", "analyze")
    # graph.add_edge("analyze", "humor")
    # graph.add_edge("humor", "scenario")
    # graph.add_edge("scenario", "image")
    # graph.add_edge("image", "postprocess")
    # graph.add_edge("postprocess", "translate")
    # graph.set_finish_point("translate")

    return graph.compile()
