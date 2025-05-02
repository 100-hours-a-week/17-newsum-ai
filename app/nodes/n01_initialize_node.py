# app/nodes/01_initialize_node.py (Improved Version)

import uuid
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 기본값 참조 또는 설정 확인용
from app.utils.logger import get_logger
from app.workflows.state import ComicState # 워크플로우 상태 모델

# 로거 설정
logger = get_logger(__name__) # __name__ 사용 권장

class InitializeNode:
    """
    워크플로우 실행을 초기화하고 초기 상태의 핵심 필드를 설정합니다.

    주요 역할:
    - 고유 ID (trace_id, comic_id) 및 타임스탬프 생성/설정.
    - 입력된 초기 쿼리(initial_query) 정규화.
    - 처리 시간 통계(processing_stats) 초기화 및 기록.

    참고:
    - 대부분의 애플리케이션 설정 값(LLM 모델, 온도, 타임아웃 등)은 상태 객체(state.config)에
      복사하지 않습니다. 대신, 각 노드나 도구가 필요할 때마다 중앙 설정 객체
      (`app.config.settings` 또는 주입된 설정 객체)를 직접 참조합니다.
    - 워크플로우 인스턴스별로 달라질 수 있는 설정(예: 사용자 지정 온도)은
      그래프 실행 시점에 외부에서 `state.config`에 주입되어야 합니다.
      이 노드는 `state.config`를 *설정하지 않으며*, 후속 노드는 필요한 설정이
      `state.config`에 존재한다고 가정하거나 `settings`의 기본값을 사용해야 합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["initial_query", "trace_id", "config"] # config도 받을 수 있음 (하지만 여기서 설정 안 함)
    outputs: List[str] = [
        "comic_id", "trace_id", "timestamp", "initial_query",
        "processing_stats", "used_links", "error_message",
        # config는 여기서 업데이트하지 않음
    ]

    def __init__(self):
        """노드 초기화"""
        logger.info("InitializeNode initialized.")

    def _normalize_query(self, query: Optional[str]) -> str:
        """쿼리 문자열 정규화: 앞뒤 공백 제거 및 연속 공백 단일화"""
        if not query:
            return ""
        # 정규 표현식을 사용하여 하나 이상의 공백 문자를 단일 공백으로 치환
        normalized = re.sub(r'\s+', ' ', query).strip()
        return normalized

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """
        워크플로우 초기 상태의 핵심 필드를 계산하고 설정합니다.

        Args:
            state: 입력 ComicState. `initial_query`, `trace_id`, `config`가 포함될 수 있습니다.
                   `config`는 이 노드에서 읽거나 쓰지 않지만, 후속 노드를 위해 전달될 수 있습니다.

        Returns:
            ComicState 업데이트를 위한 딕셔너리. 이 노드에서 설정/계산된 필드만 포함합니다.
        """
        start_time = datetime.now(timezone.utc)
        # 상태에 trace_id가 있으면 사용, 없으면 새로 생성하여 일관성 유지
        # `getattr` 사용으로 state에 trace_id가 없는 초기 상태에서도 안전하게 처리
        trace_id = getattr(state, 'trace_id', None) or str(uuid.uuid4())
        log_prefix = f"[{trace_id}]" # 로그 추적을 위한 접두사
        logger.info(f"{log_prefix} Executing InitializeNode...")

        # --- 고유 ID 및 타임스탬프 생성 ---
        comic_id = str(uuid.uuid4())
        timestamp = start_time.isoformat()

        # --- 입력 쿼리 정규화 ---
        # `getattr` 사용으로 state에 initial_query가 없는 초기 상태에서도 안전하게 처리
        initial_query = getattr(state, 'initial_query', None)
        normalized_query = self._normalize_query(initial_query)
        logger.info(f"{log_prefix} Original query: '{initial_query}', Normalized query: '{normalized_query}'")

        # --- 처리 통계 업데이트 ---
        # 이전 상태의 processing_stats를 안전하게 가져와 업데이트
        processing_stats: Dict[str, float] = getattr(state, 'processing_stats', None) or {}
        end_time = datetime.now(timezone.utc)
        processing_stats['initialize_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일

        logger.info(f"{log_prefix} Initialization complete: comic_id={comic_id}")

        # --- ComicState 업데이트를 위한 결과 반환 ---
        # 이 노드에서 계산하거나 명시적으로 초기화해야 하는 값만 반환합니다.
        # 'config' 필드는 여기서 수정하지 않습니다.
        # Pydantic 모델의 기본값으로 초기화될 필드(예: 빈 리스트/딕셔너리)는 명시적으로 반환할 필요 없음
        update_data: Dict[str, Any] = {
            "comic_id": comic_id,
            "trace_id": trace_id,          # 생성 또는 전달받은 trace_id 설정
            "timestamp": timestamp,        # 워크플로우 시작 타임스탬프
            "initial_query": normalized_query, # 정규화된 쿼리 전달
            "processing_stats": processing_stats, # 업데이트된 처리 시간 통계
            "used_links": [],              # 사용된 링크 목록 초기화 (명시적 초기화)
            "error_message": None,         # 이전 에러 메시지 초기화
        }

        # 안전장치: 반환하기 전에 ComicState 모델에 정의된 필드만 포함하는지 확인
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}