# ai/app/nodes_v2/n01_initialize_node.py
"""
N01InitializeNode: 섹션 기반 WorkflowState를 위한 노드입니다.
- `meta`, `query`, `search`, `report` 등의 하위 섹션을 채웁니다.
- LangGraph가 메인 상태에 병합할 부분 딕셔너리(partial-dict)를 반환합니다.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict  # Optional 제거

from app.workflows.state_v2 import (  # MetaSection 등 직접 임포트 제거
    WorkflowState,
    QuerySection as QuerySectionModel,  # 모델 클래스 이름 변경하여 충돌 방지
    SearchSection as SearchSectionModel,
    ReportSection as ReportSectionModel,
    IdeaSection as IdeaSectionModel,
    ScenarioSection as ScenarioSectionModel,
    ImageSection as ImageSectionModel,
    UploadSection as UploadSectionModel,
    ConfigSection as ConfigSectionModel,
    MetaSection as MetaSectionModel
)
from app.utils.logger import get_logger, summarize_for_logging

# 실제 프로젝트에서는 아래 주석을 해제하고 image_style_config에서 함수를 임포트해야 합니다.
# from app.config.image_style_config import get_image_mode_for_writer

logger = get_logger(__name__)

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 512


# 테스트 또는 임시 사용을 위한 더미 함수입니다.
# 실제 환경에서는 app.config.image_style_config에서 해당 함수를 임포트해야 합니다.
def get_image_mode_for_writer(writer_id: str) -> str:
    """작성자 ID에 따라 이미지 모드를 결정하는 더미 함수입니다."""
    logger.debug(f"더미 get_image_mode_for_writer 호출됨 (writer_id: {writer_id})")
    if writer_id == "special_writer":
        return "flux_custom"
    return "default_image_mode"


class N01InitializeNode:
    """섹션 기반 WorkflowState를 초기화하고 입력된 쿼리를 검증합니다."""

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        current_node_order = 1  # 이 노드는 첫 번째 단계이므로 1로 설정합니다.

        # ------------------------------------------------------------------
        # 0. 입력 값 수집
        # ------------------------------------------------------------------
        original_query: str = state.query.original_query or ""
        initial_config: Dict[str, Any] = state.config.config or {}

        writer_id = str(initial_config.get("writer_id", "1"))
        initial_config["image_mode"] = get_image_mode_for_writer(writer_id)

        print()
        print(f"writer_id: {writer_id} | image_mode: {initial_config['image_mode']}")
        print()

        # work_id 결정 로직: state.meta.work_id가 있으면 사용, 없으면 새로 생성
        work_id: str = state.meta.work_id or str(uuid.uuid4())

        timestamp = datetime.now(timezone.utc).isoformat()

        extra_log = {
            "work_id": work_id,
            "writer_id": writer_id,
            "node_name": node_name,
            "node_order": current_node_order
        }

        logger.info(
            f"[{node_name}] 노드 진입. 쿼리: '{original_query}' | 설정: {summarize_for_logging(initial_config, 200)}",
            extra=extra_log,
        )

        # ------------------------------------------------------------------
        # 1. 입력 쿼리 검증
        # ------------------------------------------------------------------
        errors = []
        if not original_query:
            errors.append("초기 쿼리가 비어있거나 누락되었습니다.")
        elif len(original_query) < MIN_QUERY_LENGTH:
            errors.append(f"쿼리가 너무 짧습니다 (<{MIN_QUERY_LENGTH}자).")
        elif len(original_query) > MAX_QUERY_LENGTH:
            errors.append(f"쿼리가 너무 깁니다 (>{MAX_QUERY_LENGTH}자).")

        if errors:
            msg = " ".join(errors)
            logger.error(f"[{node_name}] 유효성 검증 실패: {msg}", extra=extra_log)

            initial_workflow_status = {current_node_order: "ERROR"}

            # 오류 발생 시 Pydantic 모델을 사용하여 섹션 구성 후 model_dump() 호출
            return {
                "meta": MetaSectionModel(
                    work_id=work_id,
                    timestamp=timestamp,
                    workflow_status=initial_workflow_status,
                ).model_dump(),
                "query": QuerySectionModel(
                    original_query=original_query
                ).model_dump(),
                "config": ConfigSectionModel(config=initial_config).model_dump(),
            }

        # ------------------------------------------------------------------
        # 2. 성공적인 초기화를 위한 업데이트 딕셔너리 구성
        # ------------------------------------------------------------------
        current_workflow_status = {current_node_order: "COMPLETED"}

        update_dict: Dict[str, Any] = {
            "meta": MetaSectionModel(
                work_id=work_id,
                timestamp=timestamp,
                workflow_status=current_workflow_status,
            ).model_dump(),
            "query": QuerySectionModel(
                original_query=original_query,
                query_context={},
                initial_context_results=[],
                search_target_site_domain=[],
            ).model_dump(),
            "search": SearchSectionModel(
                search_strategy=None,
                raw_search_results=None,
            ).model_dump(),
            "report": ReportSectionModel(
                report_content=None,
                referenced_urls_for_report={"used": [], "not_used": []},
                contextual_summary=None,
                saved_report_path=None,
            ).model_dump(),
            "idea": IdeaSectionModel(
                comic_ideas=[]
            ).model_dump(),
            "scenario": ScenarioSectionModel(
                selected_comic_idea_for_scenario=None,
                comic_scenarios=[],
                comic_scenario_thumbnail=None,
            ).model_dump(),
            "image": ImageSectionModel(
                refined_prompts=[],
                generated_comic_images=[]
            ).model_dump(),
            "upload": UploadSectionModel(
                uploaded_image_urls=[],
                uploaded_report_s3_uri=None,
            ).model_dump(),
            "config": ConfigSectionModel(config=initial_config).model_dump(),
            "scratchpad": {},
        }

        logger.info(
            f"[{node_name}] 초기화 완료.",
            extra=extra_log,
        )
        return update_dict