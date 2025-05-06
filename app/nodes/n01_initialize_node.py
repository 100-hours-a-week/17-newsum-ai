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

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        original_query = state.original_query or ""
        initial_config = state.config or {}  # 다양한 설정을 받을 수 있음

        comic_id = str(uuid.uuid4())  # 새 워크플로우에 대한 고유 ID 생성
        trace_id = comic_id  # trace_id와 comic_id 동일하게 사용
        timestamp = datetime.now(timezone.utc).isoformat()

        # writer_id 외 추가 설정 예시 (config에서 가져옴)
        writer_id = initial_config.get('writer_id', 'default_writer')
        target_audience = initial_config.get('target_audience', 'general_public')  # 예시 추가 설정

        extra_log_data = {
            'trace_id': trace_id, 'comic_id': comic_id,
            'writer_id': writer_id, 'node_name': node_name, 'retry_count': 0
        }

        logger.info(
            f"Entering node. Initial Query: '{original_query}', "
            f"Config Summary: {summarize_for_logging(initial_config, max_len=200)}",
            extra=extra_log_data
        )

        # --- 입력 쿼리 유효성 검사 강화 ---
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
                "trace_id": trace_id,
                "comic_id": comic_id,
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
                "trace_id": trace_id,
                "comic_id": comic_id,
                "timestamp": timestamp,
                # original_query와 config는 state에 이미 존재하며 유효성 검사 통과했으므로 명시적 업데이트 불필요
                # 단, LangGraph의 작동 방식에 따라 명시적으로 다시 전달해야 할 수도 있음 (현재는 생략)
                "query_context": {},
                "initial_context_results": [],
                "search_strategy": None,
                # ... (이하 N01에서 초기화하는 다른 필드들) ...
                "retry_count": 0,
                "error_log": [],  # 성공 시 오류 로그 초기화
                "current_stage": node_name
            }
            logger.info(f"Exiting node. State Initialized. Summary: {summarize_for_logging(update_dict)}",
                        extra=extra_log_data)
            return update_dict

        except Exception as e:
            error_msg = f"Error during state initialization: {e}"
            logger.exception(error_msg, extra=extra_log_data)
            return {
                "trace_id": trace_id,
                "comic_id": comic_id,
                "timestamp": timestamp,
                "current_stage": "ERROR",
                "error_message": f"N01 Initialization Exception: {error_msg}",
                "error_log": [{"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                               "timestamp": datetime.now(timezone.utc).isoformat()}]
            }