# app/nodes/01_initialize_node.py (Reorganized Version)

import uuid
import re
from datetime import datetime, timezone
from typing import Dict, Any
from app.utils.logger import get_logger
from app.workflows.state import ComicState # 워크플로우 상태 모델

# --- 로거 설정 ---
logger = get_logger("InitializeNode")

class InitializeNode:
    """
    (Reorganized) 워크플로우 실행을 초기화하고 초기 상태의 핵심 필드를 설정합니다.

    주요 역할:
    - 고유 ID (trace_id, comic_id) 및 타임스탬프 생성/설정.
    - 입력된 초기 쿼리(initial_query) 정규화.
    - 처리 시간 통계(processing_stats) 초기화 및 기록.

    참고:
    - 애플리케이션 설정 값(LLM 모델, 온도, 타임아웃 등)은 상태 객체(state.config)에
      복사하지 않습니다. 대신, 각 노드나 도구가 필요할 때마다 중앙 설정 객체
      (`from app.config.settings import settings`)를 직접 참조하여 사용합니다.
      이는 상태 객체를 가볍게 유지하고 설정 관리의 일관성을 높입니다.
    """

    def __init__(self):
        """노드 초기화 (현재는 특별한 로직 없음)"""
        pass

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """
        워크플로우 초기 상태의 핵심 필드를 계산하고 설정합니다.

        Args:
            state: 입력 ComicState ('initial_query', 'trace_id' 포함 가능).

        Returns:
            ComicState 업데이트를 위한 딕셔너리 (이 노드에서 설정/계산된 핵심 값만 포함).
            LangGraph가 이 딕셔너리를 기존 상태와 병합합니다.
        """
        start_time = datetime.now(timezone.utc)
        # 상태에 trace_id가 있으면 사용, 없으면 새로 생성하여 일관성 유지
        trace_id = state.trace_id or str(uuid.uuid4())
        log_prefix = f"[{trace_id}]" # 로그 추적을 위한 접두사
        logger.info(f"{log_prefix} Executing InitializeNode...")

        # --- 고유 ID 및 타임스탬프 생성 ---
        comic_id = str(uuid.uuid4())
        timestamp = start_time.isoformat()

        # --- 입력 쿼리 정규화 ---
        initial_query = state.initial_query or ""
        normalized_query = self._normalize_query(initial_query)
        logger.info(f"{log_prefix} Original query: '{initial_query}', Normalized query: '{normalized_query}'")

        # --- 처리 통계 업데이트 ---
        # 기존 통계가 있을 수 있으므로 복사 후 업데이트
        processing_stats: Dict[str, float] = state.processing_stats.copy() if state.processing_stats else {}
        end_time = datetime.now(timezone.utc)
        processing_stats['initialize_node_time'] = (end_time - start_time).total_seconds()

        logger.info(f"{log_prefix} Initialization complete: comic_id={comic_id}")

        # --- ComicState 업데이트를 위한 결과 반환 ---
        # 이 노드에서 계산하거나 명시적으로 초기화해야 하는 값만 반환합니다.
        # 'config' 필드는 더 이상 여기서 채우지 않습니다.
        update_data: Dict[str, Any] = {
            "comic_id": comic_id,
            "trace_id": trace_id,          # 생성 또는 전달받은 trace_id 설정
            "timestamp": timestamp,        # 워크플로우 시작 타임스탬프
            "initial_query": normalized_query, # 정규화된 쿼리 전달
            "processing_stats": processing_stats, # 업데이트된 처리 시간 통계
            "used_links": [],              # 사용된 링크 목록 초기화
            "error_message": None,         # 이전 에러 메시지 초기화 (선택적)
            # 나머지 ComicState 필드들은 모델의 기본값 (빈 리스트/딕셔너리, None 등)으로
            # LangGraph에 의해 자동으로 초기화됩니다.
        }

        # --- 참고: settings.py 와의 일관성 확인 필요 ---
        # 원본 코드에서 참조했던 아래 설정들이 settings.py 에 정의되어 있어야 합니다.
        # 각 노드/도구는 필요 시 'settings.설정명' 형태로 직접 참조합니다.
        # 예: settings.MAX_ARTICLE_TEXT_LEN, settings.MAX_OPINION_TEXT_LEN,
        #     settings.MAX_CONTEXT_LEN_SCENARIO, settings.ENABLE_SCENARIO_EVALUATION,
        #     settings.HTTP_TIMEOUT 등 (이전 코드에서 참조된 항목들)
        # 만약 누락되었다면, 재구성된 settings.py 파일에 추가해야 합니다.

        # 안전장치: 반환하기 전에 ComicState 모델에 정의된 필드만 포함하는지 확인
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}

    def _normalize_query(self, query: str) -> str:
        """쿼리 문자열 정규화: 앞뒤 공백 제거 및 연속 공백 단일화"""
        if not query:
            return ""
        # 정규 표현식을 사용하여 하나 이상의 공백 문자를 단일 공백으로 치환
        normalized = re.sub(r'\s+', ' ', query).strip()
        return normalized