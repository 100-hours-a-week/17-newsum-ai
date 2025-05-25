# ai/app/workflows/v2_workflow.py
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

# state & langgraph
from app.workflows.state_v2 import WorkflowState
from langgraph.graph import StateGraph, END

# logger
from app.utils.logger import get_logger
logger = get_logger(__name__)


async def query_intent_workflow(llm_service: LLMService) -> StateGraph:
    """
    QueryIntentNode만을 포함하는 간단한 워크플로우를 컴파일합니다.
    """
    workflow = StateGraph(WorkflowState)
    node = QueryIntentNode(llm=llm_service)
    workflow.add_node("query_intent", node.run)
    workflow.set_entry_point("query_intent")
    workflow.add_edge("query_intent", END)
    logger.info("Query intent workflow compiled successfully.")
    return workflow.compile()


async def main_workflow(
        llm_service: LLMService,
        Google_Search_tool: GoogleSearchTool,
        image_generation_service: ImageService,
        storage_service: StorageService,  # <<< StorageService 주입
        translation_service: TranslationService,  # <<< TranslationService 주입
        external_api_session: Optional[aiohttp.ClientSession] = None) -> StateGraph:
    """
    전체 워크플로우를 컴파일합니다.
    """
    workflow = StateGraph(WorkflowState)
    query_intent_node = QueryIntentNode(llm=llm_service)
    workflow.add_node("query_intent", query_intent_node.run)
    workflow.set_entry_point("query_intent")
    workflow.add_edge("query_intent", END)

    # 워크플로우 컴파일
    logger.info("Main workflow compiled successfully.")
    return workflow.compile()