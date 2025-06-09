# app/workflows/workflow_controller.py
import uuid
import json
from app.workflows.state_v3 import OverallWorkflowState

from app.api.v2.schemas import ChatRequestPayload, ChatResponse

from app.services.postgresql_service import PostgreSQLService
from app.services.backend_client import BackendApiClient
from app.utils.logger import get_logger as get_custom_logger

from langgraph.pregel import Pregel
from langgraph.checkpoint.base import BaseCheckpointSaver
from app.workflows.main_workflow import V3_TARGET_MAP, NODE_INTERNAL_NAMES_V3, NODE_TO_STATE_FIELD_MAPPING
from typing import Dict, Any, Optional, Tuple, Callable, Awaitable
from pydantic_core import ValidationError

NodeHandler = Callable[[ChatRequestPayload, Dict[str, Any]], Awaitable[ChatResponse]]

controller_logger = get_custom_logger("WorkflowControllerV3")


class StateLoadError(Exception):
    """워크플로우 상태 로딩 중 발생하는 오류를 위한 사용자 정의 예외"""
    pass


class WorkflowControllerV3:
    def __init__(
            self,
            pg_service: PostgreSQLService,
            checkpointer: BaseCheckpointSaver,
            backend_client: BackendApiClient,
            compiled_graph: Dict[str, Pregel],
    ):
        self.pg_service = pg_service
        self.checkpointer = checkpointer
        self.backend_client = backend_client
        self.compiled_graph = compiled_graph
        self.node_handlers: Dict[str, NodeHandler] = {
            NODE_INTERNAL_NAMES_V3[0]: self._handle_n01_topic_clarification,
            NODE_INTERNAL_NAMES_V3[1]: self._handle_n02_report_planning,
            NODE_INTERNAL_NAMES_V3[2]: self._handle_n03_search_execution,
            NODE_INTERNAL_NAMES_V3[3]: self._handle_n04_report_synthesis,
            NODE_INTERNAL_NAMES_V3[4]: self._handle_n05_persona_analysisning,
            NODE_INTERNAL_NAMES_V3[5]: self._handle_n06_opinion_to_image_concept,
            NODE_INTERNAL_NAMES_V3[6]: self._handle_n07_image_prompts,
            NODE_INTERNAL_NAMES_V3[7]: self._handle_n08_insert_image_queue,
        }
        controller_logger.info(
            "WorkflowControllerV3 초기화 완료 (PostgreSQLService, BackendClient, CompiledGraph, Checkpointer 수신).")

    async def _load_overall_state_from_external(self, work_id: str, room_id: str) -> Optional[OverallWorkflowState]:
        controller_logger.debug(f"PostgreSQL에서 OverallState 로드 시도. work_id: {work_id}, room_id: {room_id}",
                                extra={"work_id": work_id, "room_id": room_id})
        try:
            work_id_uuid = uuid.UUID(work_id)
            room_id_int = int(room_id)
            state_data_dict = await self.pg_service.get_workflow_state(work_id_uuid, room_id_int)

            if state_data_dict:
                # --- [HOTFIX] DB에 저장된 데이터와 Pydantic 모델 간의 불일치 보정 ---
                if 'report_planning' in state_data_dict and state_data_dict['report_planning'] and state_data_dict[
                    'report_planning'].get('outline_candidates') is None:
                    controller_logger.warning(
                        f"DB의 'outline_candidates'가 None이므로 []로 수정합니다. work_id: {work_id}",
                        extra={"work_id": work_id, "room_id": room_id}
                    )
                    state_data_dict['report_planning']['outline_candidates'] = []
                # --- End of HOTFIX ---

                controller_logger.info(f"PostgreSQL에서 OverallState 로드 성공. work_id: {work_id}, room_id: {room_id}",
                                       extra={"work_id": work_id, "room_id": room_id})
                return OverallWorkflowState.model_validate(state_data_dict)

            controller_logger.warning(f"PostgreSQL에 저장된 OverallState 없음. work_id: {work_id}, room_id: {room_id}",
                                      extra={"work_id": work_id, "room_id": room_id})
            return None
        except ValidationError as e:
            controller_logger.error(f"상태 데이터 유효성 검사 실패: {e}. work_id='{work_id}', room_id='{room_id}'",
                                    exc_info=True, extra={"work_id": work_id, "room_id": room_id})
            raise StateLoadError("저장된 대화 상태를 불러오는 중 오류가 발생했습니다. 데이터 형식이 맞지 않습니다.") from e
        except (ValueError, TypeError) as e:
            controller_logger.error(f"work_id 또는 room_id 형식 오류 로드 중: {e}. work_id='{work_id}', room_id='{room_id}'",
                                    exc_info=True, extra={"work_id": work_id, "room_id": room_id})
            raise StateLoadError("work_id 또는 room_id 형식이 잘못되었습니다.") from e
        except Exception as e:
            controller_logger.error(f"PostgreSQL OverallState 로드 실패: {e}. work_id: {work_id}, room_id: {room_id}",
                                    exc_info=True, extra={"work_id": work_id, "room_id": room_id})
            raise StateLoadError("데이터베이스에서 상태를 불러오는 중 알 수 없는 오류가 발생했습니다.") from e

    async def _save_overall_state_to_external(self, state: OverallWorkflowState, room_id: str):
        if not state.work_id or not room_id:
            controller_logger.error("OverallState 저장 실패: work_id 또는 room_id가 없습니다.",
                                    extra={"work_id": state.work_id or "MISSING", "room_id": room_id or "MISSING"})
            return

        controller_logger.debug(f"PostgreSQL에 OverallState 저장 시도. work_id: {state.work_id}, room_id: {room_id}",
                                extra={"work_id": state.work_id, "room_id": room_id})
        try:
            work_id_uuid = uuid.UUID(state.work_id)
            room_id_int = int(room_id)
            state_dict_for_db = json.loads(state.model_dump_json())

            await self.pg_service.update_workflow_state(
                work_id=work_id_uuid,
                room_id=room_id_int,
                status=state.current_node_name or "UNKNOWN",
                task_details=state_dict_for_db
            )
            controller_logger.info(f"PostgreSQL에 OverallState 저장 완료. work_id: {state.work_id}, room_id: {room_id}",
                                   extra={"work_id": state.work_id, "room_id": room_id})
        except (ValueError, TypeError) as e:
            controller_logger.error(
                f"work_id 또는 room_id 형식 오류 저장 중: {e}. work_id='{state.work_id}', room_id='{room_id}'", exc_info=True,
                extra={"work_id": state.work_id, "room_id": room_id})
        except Exception as e:
            controller_logger.error(f"PostgreSQL OverallState 저장 실패: {e}. work_id: {state.work_id}, room_id: {room_id}",
                                    exc_info=True, extra={"work_id": state.work_id, "room_id": room_id})

    @staticmethod
    def _reset_state_for_target(current_state: OverallWorkflowState, api_target_status: str,
                                is_feedback_loop_to_target: bool = False) -> tuple[OverallWorkflowState, str]:
        work_id = current_state.work_id
        log_extra = {"work_id": work_id, "api_target_status": api_target_status,
                     "is_feedback": is_feedback_loop_to_target}
        controller_logger.info("상태 초기화 및 시작 노드 결정 시작.", extra=log_extra)

        initial_node_for_target = V3_TARGET_MAP.get(api_target_status)
        if not initial_node_for_target or initial_node_for_target not in NODE_INTERNAL_NAMES_V3:
            controller_logger.warning(
                f"API target_status '{api_target_status}'가 V3_TARGET_MAP에 없거나 유효하지 않은 노드. 기본 시작 노드로 폴백.",
                extra=log_extra)
            initial_node_for_target = NODE_INTERNAL_NAMES_V3[0]

        controller_logger.info(f"실제 시작될 내부 노드: {initial_node_for_target}", extra=log_extra)

        do_reset: Dict[str, bool] = {name: False for name in NODE_INTERNAL_NAMES_V3}
        try:
            start_index = NODE_INTERNAL_NAMES_V3.index(initial_node_for_target)
            for i in range(start_index, len(NODE_INTERNAL_NAMES_V3)):
                do_reset[NODE_INTERNAL_NAMES_V3[i]] = True
        except ValueError:
            controller_logger.critical(
                f"CRITICAL: 결정된 시작 노드 '{initial_node_for_target}'이 NODE_INTERNAL_NAMES_V3 목록에 없음. 전체 리셋.",
                extra=log_extra)
            do_reset = {name: True for name in NODE_INTERNAL_NAMES_V3}
            initial_node_for_target = NODE_INTERNAL_NAMES_V3[0]

        if is_feedback_loop_to_target and initial_node_for_target in do_reset:
            do_reset[initial_node_for_target] = False
            controller_logger.info(f"피드백 루프로 식별되어 노드 '{initial_node_for_target}'의 상태는 초기화하지 않습니다.", extra=log_extra)

        for node_internal_name, reset_needed in do_reset.items():
            if reset_needed and node_internal_name in NODE_TO_STATE_FIELD_MAPPING:
                state_field_name, state_model_class = NODE_TO_STATE_FIELD_MAPPING[node_internal_name]
                if hasattr(current_state, state_field_name):
                    setattr(current_state, state_field_name, state_model_class())
                    controller_logger.debug(f"상태 필드 '{state_field_name}' 초기화 완료 (노드: {node_internal_name}).",
                                            extra=log_extra)

        current_state.current_node_name = None
        current_state.error_message = None
        controller_logger.info(f"상태 초기화 완료. 최종 시작될 내부 노드: {initial_node_for_target}", extra=log_extra)
        return current_state, initial_node_for_target

    @staticmethod
    def _prepare_graph_input_from_chat_payload(
            current_overall_state: OverallWorkflowState,
            chat_payload: ChatRequestPayload,
            actual_target_node_key_for_graph: str
    ) -> Dict[str, Any]:

        message = chat_payload.message or ""

        user_input_for_feedback = message

        if user_input_for_feedback:
            controller_logger.debug(
                f"'{actual_target_node_key_for_graph}' 노드에 대한 피드백 메시지 설정: {user_input_for_feedback[:50]}...",
                extra={"work_id": current_overall_state.work_id})

        return {
            "overall_state": current_overall_state,
            "user_message_for_feedback": user_input_for_feedback,
        }

    async def process_chat_interaction(self, payload: ChatRequestPayload) -> ChatResponse:
        error_response, log_extra = self._validate_payload_and_get_log_extra(payload)
        if error_response:
            return error_response

        api_key = payload.target_status
        internal_node_name = V3_TARGET_MAP.get(api_key)

        controller_logger.info(
            f"process_chat_interaction 시작. Target API Key: '{api_key}', Mapped Node: '{internal_node_name}'",
            extra=log_extra
        )

        handler = self.node_handlers.get(internal_node_name)
        if not handler:
            controller_logger.error(f"유효하지 않은 target_status: '{api_key}' (매핑된 노드 없음)", extra=log_extra)
            return ChatResponse(
                message=f"오류: 유효하지 않은 요청입니다. ({api_key})", request_id=payload.request_id,
                work_id=payload.work_id, current_overall_status="ERROR", is_waiting_for_feedback=False
            )

        try:
            return await handler(payload, log_extra)
        except StateLoadError as e:
            controller_logger.error(f"StateLoadError 발생: {e}", extra=log_extra, exc_info=True)
            return ChatResponse(
                message=str(e),
                request_id=payload.request_id,
                work_id=payload.work_id,
                current_overall_status="ERROR_STATE_LOAD",
                is_waiting_for_feedback=False
            )
        except Exception as e:
            controller_logger.critical(
                f"'{internal_node_name}' 핸들러 실행 중 예측하지 못한 오류 발생: {e}", exc_info=True, extra=log_extra
            )
            return ChatResponse(
                message="오류: 요청 처리 중 내부 오류가 발생했습니다.", request_id=payload.request_id,
                work_id=payload.work_id, current_overall_status="ERROR", is_waiting_for_feedback=False
            )

    @staticmethod
    def _validate_payload_and_get_log_extra(
            payload: ChatRequestPayload
    ) -> Tuple[Optional[ChatResponse], Optional[Dict[str, Any]]]:
        if not all([payload.work_id, payload.room_id, payload.target_status]):
            error_msg = "work_id, room_id, target_status는 필수입니다."
            controller_logger.error(f"API 요청 유효성 검사 실패: {error_msg}", extra={"payload": payload.model_dump()})
            return ChatResponse(message=f"오류: {error_msg}", request_id=payload.request_id,
                                work_id=payload.work_id or str(uuid.uuid4()), current_overall_status="ERROR",
                                is_waiting_for_feedback=False), None

        log_extra = {
            "work_id": payload.work_id, "room_id": payload.room_id, "target_status": payload.target_status,
            "request_id": payload.request_id, "user_id": payload.user_id
        }
        return None, log_extra

    async def _base_node_handler(
            self,
            payload: ChatRequestPayload,
            log_extra: Dict[str, Any],
            handler_name: str,
            current_node_check: Callable[[OverallWorkflowState], bool],
            precondition_check: Callable[[OverallWorkflowState], Optional[str]],
            takes_feedback: bool = True
    ) -> ChatResponse:
        controller_logger.info(f"{handler_name} 핸들러 시작.", extra=log_extra)
        state = await self._load_overall_state_from_external(payload.work_id, payload.room_id)
        active_node_key = V3_TARGET_MAP.get(payload.target_status)

        if not state:
            if handler_name != "_handle_n01_topic_clarification":
                return ChatResponse(message="오류: 워크플로우를 찾을 수 없습니다. 주제 구체화부터 시작해주세요.",
                                    request_id=payload.request_id, work_id=payload.work_id,
                                    current_overall_status="ERROR_NOT_FOUND",
                                    is_waiting_for_feedback=False)

            controller_logger.info("새 워크플로우 시작. 초기 상태를 생성합니다.", extra=log_extra)
            state = OverallWorkflowState(
                work_id=payload.work_id,
                user_query=payload.message or "사용자 쿼리 없음"
            )

        else:
            controller_logger.info("기존 워크플로우를 계속합니다.", extra=log_extra)
            precondition_error = precondition_check(state)
            if precondition_error:
                return ChatResponse(message=f"오류: {precondition_error}", request_id=payload.request_id,
                                    work_id=state.work_id,
                                    current_overall_status="ERROR_PRECONDITION_FAILED", is_waiting_for_feedback=False)

            is_node_already_complete = current_node_check(state)
            if is_node_already_complete:
                return ChatResponse(message="이미 완료된 단계입니다.", request_id=payload.request_id, work_id=state.work_id,
                                    current_overall_status=f"{payload.target_status}_ALREADY_COMPLETED",
                                    is_waiting_for_feedback=False)

            state, active_node_key = self._reset_state_for_target(
                current_state=state,
                api_target_status=payload.target_status,
                is_feedback_loop_to_target=True
            )

        graph_input = self._prepare_graph_input_from_chat_payload(state, payload, active_node_key)
        final_state = await self._execute_graph(graph_input, state, active_node_key, log_extra)
        await self._save_overall_state_to_external(final_state, payload.room_id)
        return self._build_final_response(payload.request_id, final_state)

    async def _handle_n01_topic_clarification(self, payload: ChatRequestPayload,
                                              log_extra: Dict[str, Any]) -> ChatResponse:
        return await self._base_node_handler(
            payload, log_extra, "_handle_n01_topic_clarification",
            current_node_check=lambda s: s.topic_clarification.is_final,
            precondition_check=lambda s: None,
            takes_feedback=True
        )

    async def _handle_n02_report_planning(self, payload: ChatRequestPayload, log_extra: Dict[str, Any]) -> ChatResponse:
        return await self._base_node_handler(
            payload, log_extra, "_handle_n02_report_planning",
            current_node_check=lambda s: s.report_planning.is_ready,
            precondition_check=lambda s: "주제 구체화 단계를 먼저 완료해야 합니다." if not s.topic_clarification.is_final else None,
            takes_feedback=True
        )

    async def _handle_n03_search_execution(self, payload: ChatRequestPayload,
                                           log_extra: Dict[str, Any]) -> ChatResponse:
        return await self._base_node_handler(
            payload, log_extra, "_handle_n03_search_execution",
            current_node_check=lambda s: s.source_collect.is_ready,
            precondition_check=lambda s: "보고서 계획 단계를 먼저 완료해야 합니다." if not s.report_planning.is_ready else None,
            takes_feedback=False
        )

    async def _handle_n04_report_synthesis(self, payload: ChatRequestPayload,
                                           log_extra: Dict[str, Any]) -> ChatResponse:
        return await self._base_node_handler(
            payload, log_extra, "_handle_n04_report_synthesis",
            current_node_check=lambda s: s.report_draft.is_ready,
            precondition_check=lambda s: "자료 수집 단계를 먼저 완료해야 합니다." if not s.source_collect.is_ready else None,
            takes_feedback=False
        )

    async def _handle_n05_persona_analysisning(self, payload: ChatRequestPayload,
                                             log_extra: Dict[str, Any]) -> ChatResponse:
        return await self._base_node_handler(
            payload, log_extra, "_handle_n05_persona_analysisning",
            current_node_check=lambda s: s.persona_analysis.is_ready,
            precondition_check=lambda s: "보고서 생성 단계를 먼저 완료해야 합니다." if not s.report_draft.is_ready else None,
            takes_feedback=False
        )

    async def _handle_n06_opinion_to_image_concept(self, payload: ChatRequestPayload,
                                             log_extra: Dict[str, Any]) -> ChatResponse:
        return await self._base_node_handler(
            payload, log_extra, "_handle_n06_opinion_to_image_concept",
            current_node_check=lambda s: s.image_concept.is_ready,
            precondition_check=lambda s: "페르소나에 따른 의견 생성 단계를 먼저 완료해야 합니다." if not s.persona_analysis.is_ready else None,
            takes_feedback=False
        )
    async def _handle_n07_image_prompts(self, payload: ChatRequestPayload,
                                             log_extra: Dict[str, Any]) -> ChatResponse:
        return await self._base_node_handler(
            payload, log_extra, "_handle_n07_image_prompts",
            current_node_check=lambda s: s.image_prompts.is_ready,
            precondition_check=lambda s: "의견에 따른 이미지 컨셉 생성 단계를 먼저 완료해야 합니다." if not s.image_concept.is_ready else None,
            takes_feedback=False
        )
    async def _handle_n08_insert_image_queue(self, payload: ChatRequestPayload,
                                             log_extra: Dict[str, Any]) -> ChatResponse:
        return await self._base_node_handler(
            payload, log_extra, "_handle_n08_insert_image_queue",
            current_node_check=lambda s: s.insert_image_queue.is_ready,
            precondition_check=lambda s: "이미지 컨셉 의 프롬프트 변환 단계를 먼저 완료해야 합니다." if not s.image_prompts.is_ready else None,
            takes_feedback=False
        )

    async def _execute_graph(self, graph_input: Dict, initial_state: OverallWorkflowState, active_node_key: str,
                             log_extra: Dict) -> OverallWorkflowState:
        graph_to_run = self.compiled_graph.get(active_node_key)
        if not graph_to_run:
            error_msg = f"Graph execution error: No compiled graph found for node key '{active_node_key}'."
            controller_logger.error(error_msg, extra=log_extra)
            initial_state.error_message = error_msg
            return initial_state

        config = {"configurable": {"thread_id": initial_state.work_id}}
        final_state = initial_state

        try:
            controller_logger.info(f"개별 그래프 실행 시작. Target Node: {active_node_key}", extra=log_extra)
            final_output = await graph_to_run.ainvoke(graph_input, config=config)

            if final_output and "overall_state" in final_output:
                final_state = OverallWorkflowState.model_validate(final_output["overall_state"])
                controller_logger.info(f"개별 그래프 실행 완료. 최종 노드: {final_state.current_node_name}", extra=log_extra)
            else:
                raise ValueError("그래프 실행 후 유효한 'overall_state'를 얻지 못함.")
        except Exception as e:
            error_msg = f"Graph execution error in '{active_node_key}': {str(e)[:200]}"
            controller_logger.error(f"그래프 실행 중 예외 발생: {e}", exc_info=True, extra=log_extra)
            final_state.error_message = (final_state.error_message or "") + f" | {error_msg}"

        return final_state

    @staticmethod
    def _build_final_response(request_id: str, state: OverallWorkflowState) -> ChatResponse:
        if state.error_message:
            return ChatResponse(message=f"오류: {state.error_message}", request_id=request_id, work_id=state.work_id,
                                current_overall_status="ERROR", is_waiting_for_feedback=False)

        ai_question, next_target_api_key = WorkflowControllerV3._determine_next_question(state)

        if ai_question:
            return ChatResponse(
                message=ai_question, request_id=request_id, work_id=state.work_id,
                current_overall_status="WAITING_USER_INPUT", next_question=ai_question,
                is_waiting_for_feedback=True, next_expected_target_status=next_target_api_key
            )

        # --- [FIX] `panel_detail` -> `panel_details` 오타 수정 ---
        status_map = [
            (lambda s: s.insert_image_queue.is_ready, 7, "이미지 생성 큐 저장이 완료되었습니다. \n 이후 배치 일정에 따라 이미지가 생성되어 게시글로 올라가게 됩니다."),
            (lambda s: s.image_prompts.is_ready, 6, "이미지 프롬프트 변환이 완료되었습니다. \n 이제 해당 내용을 큐로 만들어 이미지화룰 준비해 주세요."),

            (lambda s: s.image_concept.is_ready, 5, "이미지 컨셉이 정해졌습니다. \n 이제 컨셉 내용을 이미지 모델에 맞게 프롬프트로 전환해 주세요."),
            (lambda s: s.persona_analysis.is_ready, 4, "페르소나에 따른 의견생성 완료. \n 의견 내용을 기반으로 4컨만화 이미지의 컨셉 및 삽입 텍스트를 생성해 주세요."),
            (lambda s: s.report_draft.is_ready, 3, "보고서 생성이 완료. \n 이제 램덤한 AI 페르소나에 의한 의견을 생성하세요"),
            (lambda s: s.source_collect.is_ready, 2, f"자료 수집 및 분석이 완료되었습니다. \n \"{len(state.source_collect.results)}\" 개의 목차"),
            (lambda s: s.report_planning.is_ready, 1,
             f"보고서 계획이 확정되었습니다. 확정된 계획은 다음과 같습니다:\n\n"
             f"```json\n"
             f"{json.dumps(state.report_planning.structure, indent=2, ensure_ascii=False)}\n"
             f"```"),
            (lambda s: s.topic_clarification.is_final, 0, f"주제가 확정되었습니다: \"{state.topic_clarification.draft}\""),
        ]

        for check_func, index, msg in status_map:
            if check_func(state):
                api_key = list(V3_TARGET_MAP.keys())[index]
                if index < len(NODE_INTERNAL_NAMES_V3) - 1:
                    next_api_key = list(V3_TARGET_MAP.keys())[index + 1]
                    msg += f" 다음 단계를 진행할 수 있습니다." # ({NODE_INTERNAL_NAMES_V3[index + 1]})
                    next_target_api_key = next_api_key
                else:
                    msg += " 모든 워크플로우가 완료되었습니다."
                    next_target_api_key = None

                return ChatResponse(
                    message=msg, request_id=request_id, work_id=state.work_id,
                    current_overall_status=f"{api_key}_COMPLETED",
                    is_waiting_for_feedback=False,
                    next_expected_target_status=next_target_api_key,
                )

        return ChatResponse(
            message="워크플로우가 시작되었습니다. 주제 구체화를 진행합니다.", request_id=request_id, work_id=state.work_id,
            current_overall_status="STARTED", is_waiting_for_feedback=False
        )

    @staticmethod
    def _determine_next_question(state: OverallWorkflowState) -> Tuple[Optional[str], Optional[str]]:
        if state.topic_clarification.question and not state.topic_clarification.is_final:
            return state.topic_clarification.question, list(V3_TARGET_MAP.keys())[0]

        if state.report_planning.planning_question and not state.report_planning.is_ready:
            return state.report_planning.planning_question, list(V3_TARGET_MAP.keys())[1]

        if state.persona_analysis.question and not state.persona_analysis.is_ready:
            return state.persona_analysis.question, list(V3_TARGET_MAP.keys())[4]

        if state.image_concept.question and not state.image_concept.is_ready:
            return state.image_concept.question, list(V3_TARGET_MAP.keys())[5]

        if state.image_prompts.question and not state.image_prompts.is_ready:
            return state.image_prompts.question, list(V3_TARGET_MAP.keys())[6]

        return None, None
