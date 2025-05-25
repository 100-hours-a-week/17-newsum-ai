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
from app.nodes_v2.search_planner_node import SearchPlannerNode
from app.nodes_v2.search_executor_node import SearchExecutorNode

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

async def compile_full_workflow(
    llm_service: LLMService,
    google_search_tool: GoogleSearchTool,
    image_generation_service: ImageService,
    storage_service: StorageService,
    translation_service: TranslationService,
    external_api_session: Optional[aiohttp.ClientSession] = None
) -> StateGraph:
    wf = StateGraph(WorkflowState)
    wf.add_node("intent", QueryIntentNode(llm=llm_service).run)
    wf.add_node("plan",   SearchPlannerNode(llm=llm_service).run)
    wf.add_node("exec",   SearchExecutorNode(search_tool=google_search_tool).run)

    wf.set_entry_point("intent")
    wf.add_edge("intent", "plan")
    wf.add_edge("plan", "exec")
    wf.add_edge("exec", END)
    return wf.compile()