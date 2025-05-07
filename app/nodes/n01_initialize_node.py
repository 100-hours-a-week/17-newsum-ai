# ai/app/nodes/n01_initialize_node.py
import uuid
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging

logger = get_logger(__name__)

# N01 설정값 예시 (실제로는 외부 설정 파일 등에서 관리 가능)
MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 512


class N01InitializeNode:
    """(업그레이드됨) 워크플로우 시작 시 상태를 초기화하고 입력 유효성을 검사하는 노드."""

    async def run(self, state: WorkflowState) -> Dict[str, Any]: # 입력으로 WorkflowState 객체 전체를 받음
        node_name = self.__class__.__name__

        # API 요청에서 전달된 original_query와 config를 사용
        original_query = state.original_query or ""
        initial_config = state.config or {}


        # --- ID 일관성 확보 ---
        # background_tasks에서 comic_id와 trace_id를 전달했다면 해당 값을 사용
        # state 객체를 통해 이 값들이 이미 설정되어 있을 것임.
        # LangGraph는 노드 실행 전 상태 객체(state)를 해당 노드의 입력으로 전달함.
        # 따라서, background_tasks.py에서 initial_workflow_input에 comic_id와 trace_id를 넣었다면,
        # N01InitializeNode의 state 인자를 통해 이 값들에 접근 가능해야 함.

        # 만약 state.comic_id나 state.trace_id가 None이나 빈 문자열로 들어온다면,
        # 이는 background_tasks.py에서 initial_workflow_input에 해당 키를 포함시키지 않았거나,
        # LangGraph의 상태 전달 방식에 대한 이해가 더 필요함을 의미.
        # 여기서는 state 객체에 이미 올바른 ID가 설정되어 있다고 가정하고 진행.
        # 또는, 명시적으로 initial_input에서 직접 ID를 가져와서 사용하는 방식도 고려 가능.

        # WorkflowState 모델에 따라 state.comic_id와 state.trace_id가 존재.
        # 이 값들은 LangGraph가 이전 단계(여기서는 background_task의 ainvoke 호출 시 입력)로부터 전달.
        comic_id_to_use = state.comic_id
        trace_id_to_use = state.trace_id
        # 만약 전달된 ID가 없다면 (예: 워크플로우가 다른 방식으로 시작된 경우) 새로 생성 (Fallback)
        if not comic_id_to_use:
            comic_id_to_use = str(uuid.uuid4())
            logger.warning(f"N01: comic_id was not provided, generating a new one: {comic_id_to_use}")
        if not trace_id_to_use:
            # trace_id는 comic_id와 동일하게 사용하거나, 별도로 관리할 수 있음
            # 여기서는 comic_id와 동일하게 설정
            trace_id_to_use = comic_id_to_use
            if comic_id_to_use != state.comic_id:  # comic_id는 있었는데 trace_id만 없었던 경우
                logger.warning(f"N01: trace_id was not provided, using comic_id as trace_id: {trace_id_to_use}")

        timestamp = datetime.now(timezone.utc).isoformat()
        writer_id = initial_config.get('writer_id', 'default_writer')

        extra_log_data = {
            'trace_id': trace_id_to_use,  # 일관된 ID 사용
            'comic_id': comic_id_to_use,  # 일관된 ID 사용
            'writer_id': writer_id,
            'node_name': node_name,
            'retry_count': 0  # 초기화 시 재시도 횟수는 0
        }

        logger.info(
            f"Entering node. Initial Query: '{original_query}', "
            f"Config Summary: {summarize_for_logging(initial_config, max_len=200)}",
            extra=extra_log_data
        )

        # --- 입력 쿼리 유효성 검사 ---
        error_messages = []
        if not original_query:
            error_messages.append("Initial query is empty or missing.")
        elif len(original_query) < MIN_QUERY_LENGTH:
            error_messages.append(
                f"Initial query is too short (min {MIN_QUERY_LENGTH} chars). Query: '{original_query}'")
        elif len(original_query) > MAX_QUERY_LENGTH:
            error_messages.append(
                f"Initial query is too long (max {MAX_QUERY_LENGTH} chars). Query length: {len(original_query)}")

        if error_messages:
            full_error_msg = " ".join(error_messages)
            logger.error(f"Input validation failed: {full_error_msg}", extra=extra_log_data)
            return {
                "trace_id": trace_id_to_use,
                "comic_id": comic_id_to_use,
                "timestamp": timestamp,
                "config": initial_config,  # 원본 config 유지
                "original_query": original_query,  # 원본 query 유지 (오류났어도)
                "current_stage": "ERROR",  # 명확한 에러 스테이지
                "error_message": f"N01 Validation Error: {full_error_msg}",  # 최상위 에러 메시지
                "error_log": [{"stage": node_name, "error": full_error_msg, "timestamp": timestamp}]
            }

        # 상태 업데이트 준비
        try:
            update_dict = {
                "trace_id": trace_id_to_use,  # 명시적으로 다시 설정하여 확인
                "comic_id": comic_id_to_use,  # 명시적으로 다시 설정하여 확인
                "timestamp": timestamp,  # N01에서 생성하는 타임스탬프
                "query_context": {},  # 초기화
                "initial_context_results": [],  # 초기화
                "search_strategy": None,  # 초기화
                "raw_search_results": None,  # 초기화 (state.py 정의에 따름)
                "report_content": None,  # 초기화 (state.py 정의에 따름)
                "saved_report_path": None,  # 초기화 (state.py 정의에 따름)
                "retry_count": 0,  # 초기화
                "error_log": [],  # 초기화
                "error_message": None,  # 초기화
                "current_stage": node_name  # 현재 노드 이름 설정
                # original_query와 config는 변경하지 않으므로 update_dict에 포함 X (state에 이미 존재)
            }
            logger.info(f"Exiting node. State Initialized. Summary: {summarize_for_logging(update_dict)}",
                        extra=extra_log_data)
            return update_dict

        except Exception as e:
            error_msg = f"Error during state initialization: {e}"
            logger.exception(error_msg, extra=extra_log_data)
            return {
                "trace_id": trace_id_to_use, # 명시적으로 다시 설정하여 확인
                "comic_id": comic_id_to_use, # 명시적으로 다시 설정하여 확인
                "timestamp": timestamp,
                "current_stage": "ERROR",
                "error_message": f"N01 Initialization Exception: {error_msg}",
                "error_log": [{"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                               "timestamp": datetime.now(timezone.utc).isoformat()}]
            }