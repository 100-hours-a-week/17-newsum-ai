# app/nodes/01_initialize_node.py (Refactored)

import uuid
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# 프로젝트 구성 요소 임포트
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState # 워크플로우 상태 모델

# 로거 설정
logger = get_logger(__name__)

class InitializeNode:
    """
    워크플로우 실행을 초기화하고 초기 상태의 핵심 필드를 설정합니다.
    [... existing docstring ...]
    """

    inputs: List[str] = ["initial_query", "trace_id", "config"]
    outputs: List[str] = [
        "comic_id", "trace_id", "timestamp", "initial_query",
        "node1_processing_stats", "used_links", "error_message",
    ]

    def __init__(self):
        logger.info("InitializeNode initialized.")

    def _normalize_query(self, query: Optional[str]) -> str:
        """쿼리 문자열 정규화: 앞뒤 공백 제거 및 연속 공백 단일화"""
        if not query:
            return ""
        normalized = re.sub(r'\s+', ' ', query).strip()
        return normalized

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """워크플로우 초기 상태의 핵심 필드를 계산하고 설정합니다."""
        start_time = datetime.now(timezone.utc)
        # 상태에 trace_id가 있으면 사용, 없으면 새로 생성
        trace_id = getattr(state, 'trace_id', None) or str(uuid.uuid4())
        # comic_id는 여기서 새로 생성
        comic_id = str(uuid.uuid4())
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # ADDED: Standard log data

        node_class_name = self.__class__.__name__
        # --- ADDED: Start Logging ---
        logger.info(f"--- Executing {node_class_name} ---", extra=extra_log_data)
        logger.debug(f"Entering state:\n{summarize_for_logging(state, is_state_object=True)}", extra=extra_log_data)
        # --------------------------

        timestamp = start_time.isoformat()

        # 입력 쿼리 정규화 (getattr로 안전하게 접근)
        initial_query = getattr(state, 'initial_query', None)
        normalized_query = self._normalize_query(initial_query)
        if initial_query is None:
             logger.warning("initial_query was not provided in the input state.", extra=extra_log_data)
        else:
             logger.info(f"Original query: '{initial_query}', Normalized query: '{normalized_query}'", extra=extra_log_data)

        # 처리 시간 계산
        end_time = datetime.now(timezone.utc)
        node1_processing_stats = (end_time - start_time).total_seconds()

        logger.info(f"Initialization complete. comic_id={comic_id}", extra=extra_log_data)

        # --- 상태 업데이트 데이터 준비 ---
        update_data: Dict[str, Any] = {
            "comic_id": comic_id,          # 새로 생성된 comic_id
            "trace_id": trace_id,          # 생성 또는 전달받은 trace_id 설정
            "timestamp": timestamp,        # 워크플로우 시작 타임스탬프
            "initial_query": normalized_query, # 정규화된 쿼리 전달
            "node1_processing_stats": node1_processing_stats,
            "used_links": [],              # 사용된 링크 목록 초기화
            "error_message": None,         # 에러 메시지 초기화
            # config는 여기서 업데이트하지 않음 (입력에서 전달만 됨)
        }

        # --- ADDED: End Logging ---
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node1_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        # ComicState 모델에 정의된 필드만 반환
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}