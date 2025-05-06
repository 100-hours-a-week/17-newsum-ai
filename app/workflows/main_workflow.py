# ai/app/workflows/main_workflow.py

from langgraph.graph import StateGraph, END

from .state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging
from app.services.llm_service import LLMService
from app.tools.search.Google_Search_tool import GoogleSearchTool

# --- 노드 클래스 임포트 (N01 ~ N06) ---
from app.nodes.n01_initialize_node import N01InitializeNode
from app.nodes.n02_analyze_query_node import N02AnalyzeQueryNode
from app.nodes.n03_understand_and_plan_node import N03UnderstandAndPlanNode
from app.nodes.n04_execute_search_node import N04ExecuteSearchNode
from app.nodes.n05_report_generation_node import N05ReportGenerationNode
from app.nodes.n06_save_report_node import N06SaveReportNode # <<< N06 임포트 추가

logger = get_logger(__name__)

async def compile_workflow(
    llm_service: LLMService,
    Google_Search_tool: GoogleSearchTool,
    # 필요 시 다른 서비스 주입 (예: N06가 설정 파일 필요 시)
) -> StateGraph:
    """
    (수정됨) Node 1부터 Node 6까지 순차적으로 실행하고 종료되도록 구성된
    워크플로우를 정의하고 컴파일합니다.
    """
    workflow = StateGraph(WorkflowState)

    # --- 노드 인스턴스 생성 (N01 ~ N06) ---
    n01_initialize = N01InitializeNode()
    n02_analyze_query = N02AnalyzeQueryNode(llm_service=llm_service, search_tool=Google_Search_tool)
    n03_understand_and_plan = N03UnderstandAndPlanNode(llm_service=llm_service)
    n04_execute_search = N04ExecuteSearchNode(search_tool=Google_Search_tool)
    n05_report_generation = N05ReportGenerationNode(llm_service=llm_service)
    n06_save_report = N06SaveReportNode() # <<< N06 인스턴스 생성

    # --- 노드 추가 (Node 1 ~ 6) ---
    workflow.add_node("n01_initialize", n01_initialize.run)
    workflow.add_node("n02_analyze_query", n02_analyze_query.run)
    workflow.add_node("n03_understand_and_plan", n03_understand_and_plan.run)
    workflow.add_node("n04_execute_search", n04_execute_search.run)
    workflow.add_node("n05_report_generation", n05_report_generation.run)
    workflow.add_node("n06_save_report", n06_save_report.run) # <<< N06 노드 추가

    # --- 엣지 정의 (Node 1 -> ... -> 5 -> 6 -> END) ---
    workflow.set_entry_point("n01_initialize")
    workflow.add_edge("n01_initialize", "n02_analyze_query")
    workflow.add_edge("n02_analyze_query", "n03_understand_and_plan")
    workflow.add_edge("n03_understand_and_plan", "n04_execute_search")
    workflow.add_edge("n04_execute_search", "n05_report_generation")
    # <<< 수정: N05 다음 N06 연결 >>>
    workflow.add_edge("n05_report_generation", "n06_save_report")
    # <<< 수정: N06 실행 후 종료(END)되도록 설정 >>>
    workflow.add_edge("n06_save_report", END)

    # --- 워크플로우 컴파일 ---
    compiled_app = workflow.compile()

    # <<< 수정: 로그 메시지 업데이트 >>>
    logger.info("Main workflow compiled successfully (Nodes 1-6 sequential, ends after n06).")
    return compiled_app