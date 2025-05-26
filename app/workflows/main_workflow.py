# ai/app/workflows/main_workflow.py
from typing import Optional

import aiohttp
from langgraph.graph import StateGraph, END

# 워크플로우 상태 정의
from .state_v2 import WorkflowState

# 유틸리티 및 서비스 임포트
from app.utils.logger import get_logger
from app.services.llm_service import LLMService
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.services.image_service import ImageService # <<< 이미지 서비스 임포트

# --- 노드 클래스 임포트 (N01 ~ N09) ---
from app.nodes_v2.n01_initialize_node import N01InitializeNode
from app.nodes_v2.n02_analyze_query_node import N02AnalyzeQueryNode
from app.nodes_v2.n03_understand_and_plan_node import N03UnderstandAndPlanNode
from app.nodes_v2.n04_execute_search_node import N04ExecuteSearchNode
from app.nodes_v2.n05_report_generation_node import N05ReportGenerationNode
# from app.nodes.n05_hitl_review_node import N05HITLReviewNode  # HITL 노드 추가
from app.nodes_v2.n06_save_report_node import N06SaveReportNode
# from app.nodes.n06a_contextual_summary_node import N06AContextualSummaryNode
# from app.nodes.n07_comic_ideation_node import N07ComicIdeationNode
# from app.nodes.n08_scenario_generation_node import N08ScenarioGenerationNode
# from app.nodes.n09_image_generation_node import N09ImageGenerationNode # <<< N09 임포트
# from app.nodes.n10_finalize_and_notify_node import N10FinalizeAndNotifyNode # <<< N10 임포트

from app.services.storage_service import StorageService # <<< StorageService 임포트
from app.services.translation_service import TranslationService # <<< TranslationService 임포트



logger = get_logger(__name__)

async def compile_workflow(
        llm_service: LLMService,
        Google_Search_tool: GoogleSearchTool,
        image_generation_service: ImageService,
        storage_service: StorageService,  # <<< StorageService 주입
        translation_service: TranslationService,  # <<< TranslationService 주입
        external_api_session: Optional[aiohttp.ClientSession] = None) -> StateGraph:
    """
    (수정됨) Node 1부터 Node 9까지 순차적으로 실행하고 종료되도록 구성된
    워크플로우를 정의하고 컴파일합니다.

    Args:
        llm_service (LLMService): 언어 모델 서비스 인스턴스.
        Google_Search_tool (GoogleSearchTool): 구글 검색 도구 인스턴스.
        image_generation_service (ImageService): 이미지 생성 서비스 인스턴스.

    Returns:
        StateGraph: 컴파일된 LangGraph 워크플로우.
    """
    workflow = StateGraph(WorkflowState)

    # --- 노드 인스턴스 생성 (N01 ~ N09) ---
    n01_initialize = N01InitializeNode()
    n02_analyze_query = N02AnalyzeQueryNode(llm_service=llm_service)
    n03_understand_and_plan = N03UnderstandAndPlanNode(llm_service=llm_service)
    n04_execute_search = N04ExecuteSearchNode(search_tool=Google_Search_tool)
    n05_report_generation = N05ReportGenerationNode(llm_service=llm_service)
    # n05_hitl_review = N05HITLReviewNode(llm_service=llm_service)  # LLM 서비스 주입
    n06_save_report = N06SaveReportNode()
    # n06a_contextual_summary = N06AContextualSummaryNode(llm_service=llm_service)
    # n07_comic_ideation = N07ComicIdeationNode(llm_service=llm_service)
    # n08_scenario_generation = N08ScenarioGenerationNode(llm_service=llm_service)
    # n09_image_generation = N09ImageGenerationNode(image_service=image_generation_service)
    # n10_finalize_and_notify = N10FinalizeAndNotifyNode(
    #     storage_service=storage_service,
    #     http_session=external_api_session  # 공유 세션 전달 또는 None
    # )
    # --- 노드 추가 (Node 1 ~ 9) ---
    workflow.add_node("n01_initialize", n01_initialize.run)
    workflow.add_node("n02_analyze_query", n02_analyze_query.run)
    workflow.add_node("n03_understand_and_plan", n03_understand_and_plan.run)
    workflow.add_node("n04_execute_search", n04_execute_search.run)
    workflow.add_node("n05_report_generation", n05_report_generation.run)
    # workflow.add_node("n05_hitl_review", n05_hitl_review.run)  # HITL 노드 추가
    workflow.add_node("n06_save_report", n06_save_report.run)
    # workflow.add_node("n06a_contextual_summary", n06a_contextual_summary.run)
    # workflow.add_node("n07_comic_ideation", n07_comic_ideation.run)
    # workflow.add_node("n08_scenario_generation", n08_scenario_generation.run)
    # workflow.add_node("n09_image_generation", n09_image_generation.run) # <<< N09 노드 추가
    # workflow.add_node("n10_finalize_and_notify", n10_finalize_and_notify.run)  # <<< N10 노드 추가

    # --- 엣지 정의 (Node 1 -> ... -> 8 -> 9 -> END) ---
    workflow.set_entry_point("n01_initialize")
    workflow.add_edge("n01_initialize", "n02_analyze_query")
    workflow.add_edge("n02_analyze_query", "n03_understand_and_plan")
    workflow.add_edge("n03_understand_and_plan", "n04_execute_search")
    workflow.add_edge("n04_execute_search", "n05_report_generation")
    # workflow.add_edge("n05_report_generation", "n05_hitl_review")

    # # HITL 노드의 결과에 따른 분기 추가 (Section 구조에 맞게 state.meta.workflow_status 사용)
    # def should_continue(state: WorkflowState) -> bool:
    #     """HITL 노드의 결과를 확인하여 워크플로우 계속 진행 여부를 결정 (Section 구조 사용)"""
    #     # state.meta.workflow_status가 'intentionally_terminated'이면 종료, 아니면 계속
    #     return getattr(state.meta, "workflow_status", None) != "intentionally_terminated"

    # # HITL 노드에서 조건부 분기 추가
    # workflow.add_conditional_edges(
    #     "n05_hitl_review",
    #     should_continue,
    #     {
    #         True: "n06_save_report",  # 계속 진행
    #         False: END  # 의도적 종료
    #     }
    # )

    # # HITL 이후 노드들
    workflow.add_edge("n05_report_generation", "n06_save_report")
    workflow.add_edge("n06_save_report", END)
    # workflow.add_edge("n06_save_report", "n06a_contextual_summary")
    # workflow.add_edge("n06a_contextual_summary", "n07_comic_ideation")
    # workflow.add_edge("n07_comic_ideation", "n08_scenario_generation")
    # workflow.add_edge("n08_scenario_generation", "n09_image_generation")
    # workflow.add_edge("n09_image_generation", "n10_finalize_and_notify")
    # workflow.add_edge("n10_finalize_and_notify", END)

    # --- 워크플로우 컴파일 ---
    compiled_app = workflow.compile()

    logger.info("Main workflow compiled successfully with HITL review node.") # 로그 메시지 업데이트
    return compiled_app