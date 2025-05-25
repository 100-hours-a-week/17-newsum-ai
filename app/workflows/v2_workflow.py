from typing import Optional
import aiohttp

# services
from app.services.llm_service import LLMService
from app.services.image_service import ImageService
from app.services.storage_service import StorageService
from app.services.translation_service import TranslationService

# tools
from app.tools.search.Google_Search_tool import GoogleSearchTool

# nodes
from app.nodes_v2.query_intent_node import QueryIntentNode
from app.nodes_v2.search_frame_node import SearchFrameNode
from app.nodes_v2.multi_search_node import MultiSearchNode

# state & langgraph
from app.workflows.state_v2 import WorkflowState
from langgraph.graph import StateGraph, END

# logger
from app.utils.logger import get_logger
logger = get_logger(__name__)


async def query_intent_workflow(llm_service: LLMService) -> StateGraph:
    workflow = StateGraph(WorkflowState)
    node = QueryIntentNode(llm=llm_service)
    workflow.add_node("query_intent", node.run)
    workflow.set_entry_point("query_intent")
    workflow.add_edge("query_intent", END)
    logger.info("Query intent workflow compiled successfully.")
    return workflow.compile()


async def full_workflow(
    llm_service: LLMService,
    google_search_tool: GoogleSearchTool,
    image_generation_service: ImageService,
    storage_service: StorageService,
    translation_service: TranslationService,
    external_api_session: Optional[aiohttp.ClientSession] = None
) -> StateGraph:
    """
    전체 3단계 워크플로우(QueryIntent → SearchFrame → MultiSearch)를 구성합니다.
    """
    workflow = StateGraph(WorkflowState)

    query_node = QueryIntentNode(llm=llm_service)
    frame_node = SearchFrameNode()
    multi_node = MultiSearchNode()

    workflow.add_node("query_intent", query_node.run)
    workflow.add_node("search_frame", frame_node.run)
    workflow.add_node("multi_search", multi_node.run)

    workflow.set_entry_point("query_intent")
    workflow.add_edge("query_intent", "search_frame")
    workflow.add_edge("search_frame", "multi_search")
    workflow.add_edge("multi_search", END)

    logger.info("Full workflow compiled successfully.")
    return workflow.compile()