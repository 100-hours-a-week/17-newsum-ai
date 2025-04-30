# app/workflows/main_workflow.py

import logging
import uuid
from langgraph.graph import StateGraph, END
from app.workflows.state import ComicState
from app.agents.collector_agent import collect_news
from app.agents.scraper_agent import ScraperAgent
from app.agents.individual_summarizer_agent import IndividualSummarizerAgent
from app.agents.synthesis_summarizer_agent import SynthesisSummarizerAgent
from app.agents.content_summarizer_agent import ContentSummarizerAgent  # Changed from SentimentAnalyzerAgent
from app.agents.humorator_agent import HumoratorAgent
from app.agents.scenariowriter_agent import ScenarioWriterAgent
from app.agents.imager_agent import ImagerAgent
from app.agents.postprocessor_agent import PostProcessorAgent
from app.agents.translator_agent import TranslatorAgent
from app.utils.agent_wrapper import save_agent_result, wrap_function_agent

logger = logging.getLogger(__name__)

# 독립 함수형 에이전트 래핑
collect_news = wrap_function_agent('collect_news', 'app.agents.collector_agent')

def build_main_workflow() -> StateGraph:
    """
    뉴스 수집부터 만화 완성까지 LangGraph 기반 플로우를 구성합니다.
    """
    graph = StateGraph(ComicState)

    # 노드 등록 - 에이전트 래퍼 적용
    graph.add_node("collect", collect_news)
    graph.add_node("scrape", save_agent_result(ScraperAgent().run))
    graph.add_node("summarize_individual", save_agent_result(IndividualSummarizerAgent().run))
    graph.add_node("summarize_synthesis", save_agent_result(SynthesisSummarizerAgent().run))
    graph.add_node("analyze_content", save_agent_result(ContentSummarizerAgent().run))
    graph.add_node("humor", save_agent_result(HumoratorAgent().run))
    graph.add_node("scenario", save_agent_result(ScenarioWriterAgent().run))
    graph.add_node("image", save_agent_result(ImagerAgent().run))
    # graph.add_node("postprocess", save_agent_result(PostProcessorAgent().run))  # Not yet implemented
    # graph.add_node("translate", save_agent_result(TranslatorAgent().run))  # Not yet implemented

    # 상태 초기화 - comic_id 추가
    def add_comic_id(state):
        """comic_id가 없는 경우 추가"""
        if hasattr(state, "comic_id") and state.comic_id:
            return state
        
        # 새 comic_id 생성
        comic_id = str(uuid.uuid4())
        state.comic_id = comic_id
        logger.info(f"워크플로우에 comic_id 추가: {comic_id}")
        return state
    
    # 초기 노드로 comic_id 추가 함수 등록
    graph.add_node("initialize", add_comic_id)

    # 조건부 라우팅 설정 (옵션)
    # 에러 상태 확인 및 워크플로우 제어
    def should_continue(state):
        """에러 상태 확인하여 워크플로우 계속 진행 여부 결정"""
        if state.get("error_message"):
            logger.warning(f"워크플로우 중단: {state.get('error_message')}")
            return "end_with_error"
        return "continue"

    # 흐름 연결
    graph.set_entry_point("initialize")
    
    # 기본 워크플로우 경로
    graph.add_edge("initialize", "collect")
    graph.add_edge("collect", "scrape")
    graph.add_edge("scrape", "summarize_individual")
    graph.add_edge("summarize_individual", "summarize_synthesis")
    graph.add_edge("summarize_synthesis", "analyze_content")
    graph.add_edge("analyze_content", "humor")
    graph.add_edge("humor", "scenario")
    graph.add_edge("scenario", "image")
    graph.add_edge("image", END)
    
    # 에러 핸들링을 위한 조건부 라우팅 (선택 사항)
    # 워크플로우를 더 견고하게 만들기 위한 향후 확장
    # graph.add_conditional_edges("collect", should_continue, {
    #     "continue": "scrape",
    #     "end_with_error": END
    # })
    
    # 미구현 단계 - 추후 구현 예정
    # graph.add_edge("image", "postprocess")
    # graph.add_edge("postprocess", "translate")
    # graph.add_edge("translate", END)

    return graph.compile()
