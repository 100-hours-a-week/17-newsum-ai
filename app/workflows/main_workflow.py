# ai/app/workflows/main_workflow.py
"""
NewSum LangGraph 워크플로우 정의 (v3 상태 및 노드)
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.pregel import Pregel  # 타입 힌트용

from typing import Dict, Any, Optional, TypedDict, Type
from functools import partial  # partial 임포트
import logging

from app.workflows.state_v3 import (
    TopicClarificationPydanticState, ReportPlanningPydanticState,
    SourceCollectPydanticState, ReportDraftingPydanticState,
    PersonaAnalysisState, ImageConceptState, ImagePromptsPydanticState,
    ImageQueueState, OverallWorkflowState
)

# 각 v3 노드 클래스 임포트
from app.nodes_v3.n_01_topic_clarification_node import N01TopicClarificationNode
from app.nodes_v3.n_02_report_search_planning_node import N02ReportSearchPlanningNode
from app.nodes_v3.n_03_search_execution_node import N03SearchExecutionNode
from app.nodes_v3.n_04_report_synthesis_node import N04ReportDraftingNode
from app.nodes_v3.n_05_persona_analysis_node import N05PersonaAnalysisNode
from app.nodes_v3.n_06_opinion_to_image_concept_node import N06OpinionToImageConceptNode
from app.nodes_v3.n_07_image_prompt_generation_node import N07ImagePromptGenerationNode
from app.nodes_v3.n_08_queue_for_image_generation_node import N08QueueForImageGenerationNode

from app.services.database_client import DatabaseClient
from app.services.llm_service import LLMService
from app.services.postgresql_service import PostgreSQLService
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.tools.scraping.article_scraper import ArticleScraperTool
from app.tools.scraping.selenium_scraper import SeleniumScraperTool
from app.config.settings import Settings

# 애플리케이션 로거 사용 권장 (예: from app.utils.logger import get_logger)
# logger = get_logger(__name__)
logger = logging.getLogger(__name__)  # 표준 로깅 사용 예시

# --- V3_TARGET_MAP, NODE_INTERNAL_NAMES_V3, NODE_TO_STATE_FIELD_MAPPING 정의 ---
V3_TARGET_MAP = {
    "TOPIC_CLARIFICATION_N01": "N01_TopicClarification",
    "REPORT_PLANNING_N02": "N02_ReportSearchPlanning",
    "SEARCH_EXECUTION_N03": "N03_SearchExecution",
    "REPORT_SYNTHESIS_N04": "N04_ReportSynthesis",
    "PERSONA_ANALYSIS_N05": "N05_PersonaAnalysis",
    "OPINION_TO_IMAGE_CONCEPT_N06": "N06_OpinionToImageConceptNode",
    "CONCEPT_TO_PROMPT_N07": "N07_ImagePromptGenerationNode",
    "SAVE_IN_QUEUE_N08": "N08_QueueForImageGenerationNode",
}
NODE_INTERNAL_NAMES_V3 = list(V3_TARGET_MAP.values())


def get_node_to_state_field_mapping() -> Dict[str, tuple[str, Type[Any]]]:
    return {
        "N01_TopicClarification": ("topic_clarification", TopicClarificationPydanticState),
        "N02_ReportSearchPlanning": ("report_planning", ReportPlanningPydanticState),
        "N03_SearchExecution": ("source_collect", SourceCollectPydanticState),
        "N04_ReportSynthesis": ("report_draft", ReportDraftingPydanticState),
        "N05_PersonaAnalysis": ("persona_analysis", PersonaAnalysisState),
        "N06_OpinionToImageConceptNode": ("image_concept", ImageConceptState),
        "N07_ImagePromptGenerationNode": ("image_prompts", ImagePromptsPydanticState),
        "N08_QueueForImageGenerationNode": ("insert_image_queue", ImageQueueState),
    }


NODE_TO_STATE_FIELD_MAPPING = get_node_to_state_field_mapping()


# ---------------------------------------------------------------------------

class WorkflowGraphStateV3(TypedDict):
    overall_state: OverallWorkflowState
    user_message_for_feedback: Optional[str]
    user_input_for_n08: Optional[Dict[str, Optional[str]]]
    # target_node_key는 컨트롤러에서 graph_input을 통해 초기 시작 노드를 LangGraph에 전달하는 용도로만 사용.
    # 그래프 상태 내에서는 overall_state.current_node_name으로 현재/이전 노드를 관리.
    # 이 TypedDict에 target_node_key를 포함할 필요는 없음. (컨트롤러의 _prepare_graph_input_from_chat_payload에서 사용)


def get_v3_node_instances(
        redis_client: DatabaseClient,
        llm_service: LLMService,
        pg_service: PostgreSQLService,
        google_search_tool_instance: GoogleSearchTool

) -> Dict[str, Any]:
    article_scraper_instance = ArticleScraperTool()
    selenium_scraper_instance = SeleniumScraperTool()

    return {
        NODE_INTERNAL_NAMES_V3[0]: N01TopicClarificationNode(redis_client, llm_service, pg_service, google_search_tool_instance),
        NODE_INTERNAL_NAMES_V3[1]: N02ReportSearchPlanningNode(redis_client, llm_service),
        NODE_INTERNAL_NAMES_V3[2]: N03SearchExecutionNode(redis_client, llm_service, google_search_tool_instance, article_scraper_instance, selenium_scraper_instance),
        NODE_INTERNAL_NAMES_V3[3]: N04ReportDraftingNode(redis_client, llm_service),  # LLMService 추가 가정
        NODE_INTERNAL_NAMES_V3[4]: N05PersonaAnalysisNode(pg_service, llm_service, redis_client),  # LLMService 추가 가정
        NODE_INTERNAL_NAMES_V3[5]: N06OpinionToImageConceptNode(llm_service, redis_client),  # LLMService 추가 가정
        NODE_INTERNAL_NAMES_V3[6]: N07ImagePromptGenerationNode(llm_service, redis_client),  # LLMService 추가 가정
        NODE_INTERNAL_NAMES_V3[7]: N08QueueForImageGenerationNode(pg_service, redis_client, llm_service),
    }

# 일반화된 노드 실행 래퍼 함수
async def execute_node_wrapper(
        state: WorkflowGraphStateV3,
        node_key: str,  # 현재 실행해야 할 노드의 키 (예: "N01_TopicClarification")
        actual_node_instance: Any  # 미리 바인딩된 실제 노드 인스턴스
) -> Dict[str, Any]:  # 업데이트할 상태 부분만 반환 (주로 overall_state)

    current_overall_state = state['overall_state']
    overall_state_dict = current_overall_state.model_dump(exclude_none=True)
    work_id_for_log = current_overall_state.work_id if current_overall_state else "UNKNOWN_WORK_ID_WRAPPER"

    user_response_for_node: Any = None  # 일반 user_response

    if node_key == NODE_INTERNAL_NAMES_V3[0] or node_key == NODE_INTERNAL_NAMES_V3[1] \
            or node_key == NODE_INTERNAL_NAMES_V3[4] or node_key == NODE_INTERNAL_NAMES_V3[5] or node_key == NODE_INTERNAL_NAMES_V3[6]:
        user_response_for_node = state.get('user_message_for_feedback')

    log_message_content = "None"
    if user_response_for_node is not None:
        log_message_content = f"type: {type(user_response_for_node)}, content: {str(user_response_for_node)[:100]}..."

    logger.info(f"Executing node '{node_key}' with user_response_data: {log_message_content}",
                extra={"work_id": work_id_for_log, "node_key": node_key})


    # 실제 노드 인스턴스의 __call__ 메서드 호출
    try:
        # 노드별로 __call__ 시그니처에 맞춰 호출
        if node_key == NODE_INTERNAL_NAMES_V3[0] or node_key == NODE_INTERNAL_NAMES_V3[1] \
                or node_key == NODE_INTERNAL_NAMES_V3[4] or node_key == NODE_INTERNAL_NAMES_V3[5] or node_key == NODE_INTERNAL_NAMES_V3[6]:
            updated_overall_dict = await actual_node_instance(overall_state_dict, user_response=user_response_for_node)
        # user_response를 받지 않는 노드들 (예: N03, N04 등)
        elif node_key in [NODE_INTERNAL_NAMES_V3[2], NODE_INTERNAL_NAMES_V3[3], NODE_INTERNAL_NAMES_V3[7]]:
            updated_overall_dict = await actual_node_instance(overall_state_dict)  # user_response 없이 호출
        else:
            # 혹시 정의되지 않은 노드 키에 대한 처리 (이론상 발생 안 함)
            logger.error(f"Unknown node_key '{node_key}' in execute_node_wrapper dispatch.",
                         extra={"work_id": work_id_for_log})
            current_overall_state.error_message = f"Unknown node key for execution: {node_key}"
            return {"overall_state": current_overall_state}

    except Exception as e:
        # 에러 발생 시 overall_state에 에러 메시지 기록하고 반환
        logger.error(f"Node '{node_key}' execution error: {e}", exc_info=True, extra={"work_id": work_id_for_log})
        current_overall_state.error_message = f"Error in {node_key}: {str(e)}"
        return {"overall_state": current_overall_state}

    # 사용된 피드백 값은 다음 상태로 넘어갈 때 초기화 (None으로 설정)
    # 이렇게 하면 동일한 피드백이 다음 노드 실행에 의도치 않게 다시 사용되는 것을 방지
    new_overall_state = OverallWorkflowState.model_validate(updated_overall_dict)
    new_overall_state.current_node_name = node_key

    cleaned_user_message_for_feedback = state.get('user_message_for_feedback')

    if node_key in [NODE_INTERNAL_NAMES_V3[0], NODE_INTERNAL_NAMES_V3[1], NODE_INTERNAL_NAMES_V3[4], NODE_INTERNAL_NAMES_V3[5], NODE_INTERNAL_NAMES_V3[6]]:
        cleaned_user_message_for_feedback = None

    return {
        "overall_state": new_overall_state,
        "user_message_for_feedback": cleaned_user_message_for_feedback,
    }


def build_v3_workflow_graph(
        node_instances_param: Dict[str, Any],
        checkpointer: Optional[BaseCheckpointSaver] = None
) -> Pregel:
    workflow = StateGraph(WorkflowGraphStateV3)

    # 각 노드를 그래프에 추가 (partial 사용)
    for node_key, instance in node_instances_param.items():
        if node_key in NODE_INTERNAL_NAMES_V3:  # 유효한 노드 이름인지 확인
            # --- 추가된 디버깅 로그 ---
            logger.info(f"DEBUG BUILD_GRAPH for '{node_key}': Binding 'instance' of type: {type(instance)} to partial.",
                        extra={"work_id": "BUILD_GRAPH_DEBUG"})  # work_id는 이 시점에 없을 수 있음
            if callable(instance):
                logger.info(f"DEBUG BUILD_GRAPH for '{node_key}': 'instance' IS callable.",
                            extra={"work_id": "BUILD_GRAPH_DEBUG"})
            else:
                logger.warning(f"DEBUG BUILD_GRAPH for '{node_key}': 'instance' IS NOT callable.",
                               extra={"work_id": "BUILD_GRAPH_DEBUG"})
            # --- 디버깅 로그 끝 ---

            node_executable = partial(execute_node_wrapper, node_key=node_key, actual_node_instance=instance)
            workflow.add_node(node_key, node_executable)
            logger.info(f"Added node '{node_key}' to graph.")
        else:
            logger.warning(f"Node key '{node_key}' from instances is not in NODE_INTERNAL_NAMES_V3. Skipping.")

    # 라우터 함수
    def workflow_router_v3(state: WorkflowGraphStateV3) -> str:
        overall = state['overall_state']
        work_id_for_log = overall.work_id if overall else "UNKNOWN_WORK_ID_ROUTER"

        last_executed_node_role = overall.current_node_name  # 이전 노드 실행 래퍼에서 설정한 값
        logger.debug(f"Router_v3: WorkID={work_id_for_log}, LastExecutedNodeRole='{last_executed_node_role}'",
                     extra={"work_id": work_id_for_log})

        if overall.error_message:
            logger.error(f"Router_v3: 워크플로우 오류 감지됨 - '{overall.error_message}' (WorkID: {work_id_for_log})",
                         extra={"work_id": work_id_for_log})
            return END

        next_node_to_execute: Optional[str] = None

        if not last_executed_node_role:
            # 이 경우는 엔트리 포인트가 노드가 아닌 라우터로 잘못 설정되었거나,
            # 첫 노드 실행 후 current_node_name이 설정되지 않은 경우.
            # 정상적인 경우라면 set_entry_point로 지정된 노드가 먼저 실행되고,
            # 그 노드가 current_node_name을 설정하므로, 라우터는 항상 이 값을 참조할 수 있어야 함.
            logger.error(
                f"Router_v3: last_executed_node_role이 없습니다! (WorkID: {work_id_for_log}) 그래프 시작 설정을 확인하세요. 안전하게 첫 노드로 라우팅 시도.",
                extra={"work_id": work_id_for_log})
            # 강제로 첫 번째 노드를 다음 노드로 지정하거나, 오류로 간주하고 END로 보낼 수 있음.
            # 여기서는 첫 번째 노드를 지정. (하지만 이 상황은 발생하지 않아야 함)
            next_node_to_execute = NODE_INTERNAL_NAMES_V3[0]

        # --- 각 노드 실행 후 다음 상태 결정 로직 ---

        elif last_executed_node_role == NODE_INTERNAL_NAMES_V3[0]:  # N01_TopicClarification
            if overall.topic_clarification.is_final:
                logger.info(
                    f"Router_v3: {NODE_INTERNAL_NAMES_V3[0]} 최종 확정. 사용자에게 알리기 위해 END. (WorkID: {work_id_for_log})",
                    extra={"work_id": work_id_for_log})
                return END  # 다음 노드로 바로 가지 않고 일단 종료하여 메시지 전달 기회 확보
            elif overall.topic_clarification.question:
                return END  # 사용자 입력 대기
            else:
                next_node_to_execute = NODE_INTERNAL_NAMES_V3[0]

        elif last_executed_node_role == NODE_INTERNAL_NAMES_V3[1]:  # N02_ReportSearchPlanning
            if overall.report_planning.is_ready:
                next_node_to_execute = NODE_INTERNAL_NAMES_V3[2]  # N03_SearchExecution
            elif overall.report_planning.planning_question:
                return END
            else:
                next_node_to_execute = NODE_INTERNAL_NAMES_V3[1]

        elif last_executed_node_role == NODE_INTERNAL_NAMES_V3[2]:  # N03_SearchExecution
            # N03은 source_collect 상태를 업데이트
            if overall.source_collect and overall.source_collect.groups:
                next_node_to_execute = NODE_INTERNAL_NAMES_V3[3]  # N04_ReportSynthesis
            else:
                logger.warning(f"Router_v3: {NODE_INTERNAL_NAMES_V3[2]} 검색 결과 없음. (WorkID: {work_id_for_log})",
                               extra={"work_id": work_id_for_log})
                overall.error_message = f"{NODE_INTERNAL_NAMES_V3[2]}: 검색 결과가 없습니다."
                return END

        elif last_executed_node_role == NODE_INTERNAL_NAMES_V3[3]:  # N04_ReportSynthesis
            if overall.report_draft and overall.report_draft.draft:  # html_path 대신 content 확인 (또는 둘 다)
                next_node_to_execute = NODE_INTERNAL_NAMES_V3[4]  # N05_PersonaAnalysis
            else:
                logger.error(f"Router_v3: {NODE_INTERNAL_NAMES_V3[3]} 보고서 생성 실패. (WorkID: {work_id_for_log})",
                             extra={"work_id": work_id_for_log})
                overall.error_message = f"{NODE_INTERNAL_NAMES_V3[3]}: 보고서 합성에 실패했습니다."
                return END

        elif last_executed_node_role == NODE_INTERNAL_NAMES_V3[4]:  # N05_PersonaAnalysis
            if overall.persona_analysis and overall.persona_analysis.selected_opinion:
                next_node_to_execute = NODE_INTERNAL_NAMES_V3[5]  # OPINION_TO_IMAGE_CONCEPT_N06
            else:
                logger.error(f"Router_v3: {NODE_INTERNAL_NAMES_V3[4]} AI의견 생성 실패. (WorkID: {work_id_for_log})",
                             extra={"work_id": work_id_for_log})
                overall.error_message = f"{NODE_INTERNAL_NAMES_V3[4]}: AI의견 생성에 실패했습니다."
                return END

        elif last_executed_node_role == NODE_INTERNAL_NAMES_V3[5]:
            if overall.image_concept and overall.image_concept.final_concepts:
                next_node_to_execute = NODE_INTERNAL_NAMES_V3[6]
            else:
                logger.error(f"Router_v3: {NODE_INTERNAL_NAMES_V3[5]} 이미지 컨셉 생성 실패. (WorkID: {work_id_for_log})",
                             extra={"work_id": work_id_for_log})
                overall.error_message = f"{NODE_INTERNAL_NAMES_V3[5]}: 이미지 컨셉 생성에 실패했습니다."
                return END

        elif last_executed_node_role == NODE_INTERNAL_NAMES_V3[6]:  # N05_PersonaAnalysis
            if overall.image_prompts and overall.image_prompts.prompt_candidates:
                next_node_to_execute = NODE_INTERNAL_NAMES_V3[7]  # OPINION_TO_IMAGE_CONCEPT_N06
            else:
                logger.error(f"Router_v3: {NODE_INTERNAL_NAMES_V3[6]} 이미지 프롬프트 생성 실패. (WorkID: {work_id_for_log})",
                             extra={"work_id": work_id_for_log})
                overall.error_message = f"{NODE_INTERNAL_NAMES_V3[6]}: 이미지 프롬프트 생성에 실패했습니다."
                return END

        else:  # 라우팅 규칙에 없는 노드 (이론상 발생하면 안됨)
            logger.error(
                f"Router_v3: 알 수 없는 이전 노드 '{last_executed_node_role}'. 워크플로우 강제 종료. (WorkID: {work_id_for_log})",
                extra={"work_id": work_id_for_log})
            overall.error_message = f"알 수 없는 노드 '{last_executed_node_role}'에서 라우팅 실패."
            return END

        if next_node_to_execute and next_node_to_execute in node_instances_param:
            logger.info(f"Router_v3: 다음 실행할 노드 -> '{next_node_to_execute}' (WorkID: {work_id_for_log})",
                        extra={"work_id": work_id_for_log, "next_node": next_node_to_execute})
            return next_node_to_execute  # 다음 노드의 실제 이름 반환

        logger.info(
            f"Router_v3: 다음 실행할 노드 결정 불가 또는 정상적 워크플로우 종료. LastExecuted: '{last_executed_node_role}' (WorkID: {work_id_for_log})",
            extra={"work_id": work_id_for_log})
        return END  # 안전장치 또는 정상 종료

    # 조건부 엣지 설정: 각 실제 노드에서 라우터로, 라우터에서 다음 실제 노드로.
    # 모든 노드는 실행 후 workflow_router_v3를 호출합니다.
    # workflow_router_v3는 다음에 실행될 노드의 '이름'을 반환합니다.
    # 이 이름은 workflow.add_node()에 사용된 키와 일치해야 합니다.
    for node_name_key in node_instances_param.keys():
        if node_name_key in NODE_INTERNAL_NAMES_V3 and node_name_key in workflow.nodes:
            workflow.add_conditional_edges(
                node_name_key,  # 현재 노드에서
                workflow_router_v3,  # 라우터 함수를 통해 다음 노드
                {node_name: node_name for node_name in NODE_INTERNAL_NAMES_V3 + [END]}  # 라우터가 반환하는 노드이름/END를 그대로 사용
            )
        elif node_name_key in NODE_INTERNAL_NAMES_V3 and node_name_key not in workflow.nodes:
            logger.error(f"엣지 설정 오류: 노드 '{node_name_key}'가 node_instances_param에는 있지만 그래프에 add_node되지 않았습니다.")

    # 엔트리 포인트 설정
    workflow.set_entry_point(NODE_INTERNAL_NAMES_V3[0])  # 예: N01_TopicClarification으로 시작

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def build_single_node_graph(node_key: str, node_instance: Any, checkpointer: BaseCheckpointSaver) -> Pregel:
    """지정된 단일 노드에 대한 그래프를 빌드하고 컴파일하는 팩토리 함수"""
    workflow = StateGraph(WorkflowGraphStateV3)

    # 단일 노드 추가
    node_executable = partial(execute_node_wrapper, node_key=node_key, actual_node_instance=node_instance)
    workflow.add_node(node_key, node_executable)

    # 라우팅 없이 바로 종료
    workflow.set_entry_point(node_key)
    workflow.add_edge(node_key, END)

    return workflow.compile(checkpointer=checkpointer)

async def compile_workflow(
        redis_client: DatabaseClient,
        llm_service: LLMService,
        pg_service: PostgreSQLService,
        google_search_tool_instance: GoogleSearchTool,
        settings_obj: Settings,
        checkpointer: Optional[BaseCheckpointSaver] = None
    ) -> Pregel:
        logger.info("Compiling all individual v3 workflows under the original function name...")
        initialized_node_instances = get_v3_node_instances(
            redis_client=redis_client,
            llm_service=llm_service,
            pg_service=pg_service,
            google_search_tool_instance=google_search_tool_instance
        )

        compiled_graphs = {}
        for node_key, instance in initialized_node_instances.items():
            # 내부 로직은 개별 그래프를 생성하도록 변경
            compiled_graphs[node_key] = build_single_node_graph(node_key, instance, checkpointer)

        logger.info("All individual v3 workflows compiled successfully.")
        return compiled_graphs  # 반환값의 구조가 완전히 달라짐
