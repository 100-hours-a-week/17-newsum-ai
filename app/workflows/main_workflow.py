# app/workflows/main_workflow.py (Refactored Version)

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

# LangGraph 및 상태/노드 임포트
from langgraph.graph import StateGraph, END
from app.workflows.state import ComicState

# --- 모든 노드 클래스 임포트 (타입 힌트용) ---
from app.nodes.n01_initialize_node import InitializeNode
from app.nodes.n02_topic_analyzer_node import TopicAnalyzerNode
from app.nodes.n03_news_collector_node import NewsCollectorNode
from app.nodes.n04_opinion_collector_node import OpinionCollectorNode
from app.nodes.n05_news_scraper_node import NewsScraperNode
from app.nodes.n06_opinion_scraper_node import OpinionScraperNode
from app.nodes.n07_filter_node import FilterNode
from app.nodes.n08_news_summarizer_node import NewsSummarizerNode
from app.nodes.n09_opinion_summarizer_node import OpinionSummarizerNode
from app.nodes.n10_synthesis_summarizer_node import SynthesisSummarizerNode
from app.nodes.n11_evaluate_summary_node import EvaluateSummaryNode
from app.nodes.n12_trend_analyzer_node import TrendAnalyzerNode
from app.nodes.n13_progress_report_node import ProgressReportNode
from app.nodes.n14_idea_generator_node import IdeaGeneratorNode
from app.nodes.n15_scenario_writer_node import ScenarioWriterNode
from app.nodes.n16_scenario_report_node import ScenarioReportNode
from app.nodes.n17_imager_node import ImagerNode
from app.nodes.n18_translator_node import TranslatorNode
from app.nodes.n19_postprocessor_node import PostprocessorNode


# 로깅 설정
logger = logging.getLogger("MainWorkflowBuilder") # get_logger 대신 기본 로깅 사용 가능

# --- 조건부 엣지 함수 ---
def should_continue_based_on_evaluation(state: ComicState) -> str:
    """
    요약 평가 결과('decision' 상태)에 따라 다음 단계를 결정합니다.
    """
    decision = state.decision
    trace_id = state.trace_id
    log_prefix = f"[{trace_id}]" if trace_id else ""

    logger.info(f"{log_prefix} Evaluating decision: '{decision}'")
    if decision == "research_again":
        logger.warning(f"{log_prefix} Branching: research_again. Workflow likely ending here.")
        return "end_workflow"
    elif decision == "refine_topic":
        logger.warning(f"{log_prefix} Branching: refine_topic. Suggests re-running topic analysis.")
        return "end_workflow_with_refine_suggestion"
    elif decision == "proceed":
        logger.info(f"{log_prefix} Branching: proceed to creative generation.")
        return "generate_ideas"
    else: # 'proceed'가 아니거나 decision이 없는 경우 (기본값 또는 오류)
        logger.warning(f"{log_prefix} Unknown or missing decision ('{decision}'). Proceeding as default.")
        return "generate_ideas"

# --- 워크플로우 빌드 함수 (수정본) ---
def build_main_workflow(
    # --- 모든 노드 인스턴스를 인자로 받음 ---
    node01: InitializeNode,
    node02: TopicAnalyzerNode,
    node03: NewsCollectorNode,
    node04: OpinionCollectorNode,
    node05: NewsScraperNode,
    node06: OpinionScraperNode,
    node07: FilterNode,
    node08: NewsSummarizerNode,
    node09: OpinionSummarizerNode,
    node10: SynthesisSummarizerNode,
    node11: EvaluateSummaryNode,
    node12: TrendAnalyzerNode,
    node13: ProgressReportNode,
    node14: IdeaGeneratorNode,
    node15: ScenarioWriterNode,
    node16: ScenarioReportNode,
    node17: ImagerNode,
    node18: TranslatorNode,
    node19: PostprocessorNode
) -> StateGraph: # 컴파일 전 StateGraph 객체 반환
    """
    주입받은 노드 인스턴스를 사용하여 LangGraph StateGraph를 빌드합니다.
    컴파일은 이 함수 외부 (예: main.py의 lifespan)에서 수행됩니다.
    """
    logger.info("Building the full comic generation workflow graph using provided nodes...")
    workflow = StateGraph(ComicState)

    # --- 서비스/도구/노드 내부 인스턴스화 제거 ---
    # (main.py의 lifespan에서 생성된 객체가 노드에 주입되어 전달됨)

    # --- 그래프 노드 추가 (인자로 받은 노드 사용) ---
    workflow.add_node("initialize", node01.run)
    workflow.add_node("analyze_topic", node02.run)
    workflow.add_node("collect_news", node03.run)
    workflow.add_node("collect_opinions", node04.run)
    workflow.add_node("scrape_news", node05.run)
    workflow.add_node("scrape_opinions", node06.run)
    workflow.add_node("filter_opinions", node07.run)
    workflow.add_node("summarize_news", node08.run)
    workflow.add_node("summarize_opinions", node09.run)
    workflow.add_node("synthesize_summary", node10.run)
    workflow.add_node("evaluate_summary", node11.run)
    workflow.add_node("analyze_trends", node12.run)
    workflow.add_node("report_progress", node13.run)
    workflow.add_node("generate_ideas", node14.run)
    workflow.add_node("write_scenario", node15.run)
    workflow.add_node("report_scenario", node16.run)
    workflow.add_node("generate_images", node17.run)
    workflow.add_node("translate_dialogue", node18.run)
    workflow.add_node("postprocess_comic", node19.run)

    logger.info("All nodes added to the graph.")

    # --- 그래프 진입점 설정 ---
    workflow.set_entry_point("initialize")
    logger.info("Graph entry point set to 'initialize'.")

    # --- 그래프 엣지 정의 ---
    workflow.add_edge("initialize", "analyze_topic")

    # Topic Analysis 후 병렬 실행
    workflow.add_edge("analyze_topic", "collect_news")
    workflow.add_edge("analyze_topic", "collect_opinions")
    workflow.add_edge("analyze_topic", "analyze_trends") # 트렌드 분석도 병렬 시작

    # 데이터 수집 후 스크래핑
    workflow.add_edge("collect_news", "scrape_news")
    workflow.add_edge("collect_opinions", "scrape_opinions")

    # 스크래핑 후 요약
    workflow.add_edge("scrape_news", "summarize_news")
    workflow.add_edge("scrape_opinions", "filter_opinions") # 의견은 필터링 먼저
    workflow.add_edge("filter_opinions", "summarize_opinions")

    # 양쪽 요약 완료 후 통합
    workflow.add_edge("summarize_news", "synthesize_summary")
    workflow.add_edge("summarize_opinions", "synthesize_summary")

    # 통합 요약 후 평가 및 진행 보고
    workflow.add_edge("synthesize_summary", "evaluate_summary")
    # 트렌드 분석 -> 보고서, 평가 -> 보고서 (두 경로 모두 보고서 노드로 이어짐)
    workflow.add_edge("analyze_trends", "report_progress")
    workflow.add_edge("evaluate_summary", "report_progress")

    # 평가 결과에 따른 분기 (Conditional Edge)
    workflow.add_conditional_edges(
        #"evaluate_summary", # 평가 노드 이후 분기 (주의: report_progress 이후에 분기해야 할 수도 있음)
        "report_progress",
        should_continue_based_on_evaluation, # 상태 기반 결정 함수
        {
            "generate_ideas": "generate_ideas", # 아이디어 생성으로 진행
            "end_workflow": END, # 워크플로우 종료 (재탐색 필요 시)
            "end_workflow_with_refine_suggestion": END, # 워크플로우 종료 (토픽 재정의 필요 시)
        }
    )
    # 중요 참고: 위 조건부 엣지는 evaluate_summary 바로 다음에 연결되었습니다.
    # 하지만 report_progress 노드도 evaluate_summary 이후에 실행되도록 연결되어 있습니다.
    # LangGraph 실행 순서에 따라 report_progress가 실행되기 전에 분기될 수 있습니다.
    # 만약 평가 후 *그리고* 보고서 생성 후 분기해야 한다면,
    # 조건부 엣지의 시작 노드를 "report_progress"로 변경해야 합니다.
    # 즉, workflow.add_conditional_edges("report_progress", ...) 형태가 되어야 할 수 있습니다.
    # 이는 워크플로우의 정확한 요구사항에 따라 결정해야 합니다. 현재 코드는 평가 직후 분기합니다.
    logger.info("Conditional edge added after evaluation.")

    # 창작 단계 (generate_ideas 부터 시작)
    workflow.add_edge("generate_ideas", "write_scenario")
    workflow.add_edge("write_scenario", "report_scenario") # 시나리오 보고서 생성
    workflow.add_edge("report_scenario", "generate_images") # 보고 후 이미지 생성
    workflow.add_edge("generate_images", "translate_dialogue") # 이미지 생성 후 번역
    workflow.add_edge("translate_dialogue", "postprocess_comic") # 번역 후 최종 후처리
    workflow.add_edge("postprocess_comic", END) # 최종 완료

    logger.info("All edges defined.")

    # --- 그래프 컴파일 및 체크포인터 설정 제거 ---
    # (이 함수의 책임이 아니며, main.py의 lifespan에서 수행)

    # --- 그래프 시각화 제거 ---
    # (필요 시 main.py에서 컴파일 후 수행)

    return workflow # 컴파일되지 않은 StateGraph 객체 반환

# --- run_workflow 함수 제거 ---
# (초기화 로직 중복 및 복잡성 증가로 인해 제거 권장)
# (필요 시 별도 테스트 스크립트에서 유사 로직 구현)

# --- 스크립트 직접 실행 예시 제거 ---
# (이 파일은 빌더 역할만 수행)