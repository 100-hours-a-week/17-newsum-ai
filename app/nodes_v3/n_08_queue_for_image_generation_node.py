# app/nodes_v3/n_08_queue_for_image_generation_node.py

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

# --- 애플리케이션 구성 요소 임포트 ---
from app.services.postgresql_service import PostgreSQLService
from app.services.database_client import DatabaseClient  # RedisClient 용도
from app.utils.logger import get_logger
from app.workflows.state_v3 import (
    OverallWorkflowState,
    ImagePromptItemPydantic,
    ImageQueueState
)

# --- 로거 설정 ---
logger = get_logger("n_08_QueueForImageGenerationNode")
FIXED_TIME = datetime(2025, 6, 10, 15, 0, tzinfo=timezone.utc)


class N08QueueForImageGenerationNode:
    """
    LangGraph 비동기 노드 – 이미지 생성용 데이터를 DB 큐에 저장합니다. (n_08)
    """

    def __init__(self, postgre_db_client: PostgreSQLService, redis_client: DatabaseClient):
        """
        노드 초기화. 필요한 서비스 클라이언트들을 주입받습니다.
        :param postgre_db_client: PostgreSQL DB 상호작용을 위한 클라이언트
        :param redis_client: 상태 저장을 위한 Redis 클라이언트
        """
        self.db = postgre_db_client
        self.redis = redis_client
        self.logger = logger

    async def __call__(self, current_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """LangGraph 노드의 메인 진입점 함수. 이 노드는 사용자와 상호작용하지 않습니다."""
        work_id = current_state_dict.get("work_id", "UNKNOWN_WORK_ID_N08")
        log_extra = {"work_id": work_id}
        self.logger.info("N08_QueueForImageGenerationNode 시작.", extra=log_extra)

        try:
            workflow_state = OverallWorkflowState(**current_state_dict)
            node_state = workflow_state.insert_image_queue
        except ValidationError as e:
            self.logger.error(f"N08 State 유효성 검사 실패: {e}", extra=log_extra)
            current_state_dict.setdefault('insert_image_queue', {})['error_message'] = str(e)
            return current_state_dict

        # --- 이전 노드들의 최종 결과물이 준비되었는지 확인 ---
        persona_analysis_state = workflow_state.persona_analysis
        report_draft_state = workflow_state.report_draft
        image_prompts_state = workflow_state.image_prompts

        if not (persona_analysis_state.is_ready and report_draft_state.is_ready and image_prompts_state.is_ready):
            error_msg = "이전 노드(페르소나 분석, 보고서 작성, 프롬프트 생성) 중 하나 이상이 완료되지 않았습니다."
            self.logger.error(error_msg, extra=log_extra)
            node_state.error_message = error_msg
            return await self._finalize_and_save_state(workflow_state, log_extra)

        # --- DB에 저장할 데이터 추출 ---
        try:
            # 썸네일(0번) + 4컷(1~4번) 프롬프트 합치기
            all_prompts = []
            thumbnail = image_prompts_state.thumbnail_prompt
            panels = image_prompts_state.panels
            if thumbnail:
                all_prompts.append(thumbnail)
            all_prompts.extend(panels)
            # image_concept_state 정의 추가
            image_concept_state = workflow_state.image_concept
            # title, panel_descriptions 추출
            title = image_concept_state.final_thumbnail.caption if image_concept_state.final_thumbnail else None
            panel_descriptions = [c.caption for c in image_concept_state.final_concepts] if image_concept_state.final_concepts else []
            job_data = {
                "work_id": work_id,
                "persona_id": persona_analysis_state.selected_opinion.persona_id,
                "report_text": report_draft_state.draft,
                "title": title,
                "panel_descriptions": panel_descriptions,
                "image_prompts": [p.model_dump() for p in all_prompts],
                "scheduled_at": FIXED_TIME
            }
        except AttributeError as e:
            error_msg = f"DB에 저장할 데이터를 추출하는 중 필요한 값이 없습니다: {e}"
            self.logger.error(error_msg, extra=log_extra)
            node_state.error_message = error_msg
            return await self._finalize_and_save_state(workflow_state, log_extra)

        # --- DB에 작업 큐잉 ---
        job_id = await self._queue_image_generation_job(**job_data)

        if job_id:
            node_state.is_ready = True
            node_state.job_id = job_id
            self.logger.info(f"이미지 생성 작업(Job ID: {job_id})이 큐에 성공적으로 저장되었습니다.", extra=log_extra)
        else:
            node_state.error_message = "이미지 생성 작업을 큐에 저장하는 데 실패했습니다."
            self.logger.error(node_state.error_message, extra=log_extra)

        workflow_state.insert_image_queue = node_state
        return await self._finalize_and_save_state(workflow_state, log_extra)

    async def _queue_image_generation_job(self, work_id: str, persona_id: str, report_text: str, title: str, panel_descriptions: list, image_prompts: list, scheduled_at: str) -> Optional[int]:
        """
        이미지 생성에 필요한 모든 데이터를 DB 테이블에 'pending' 상태로 삽입합니다.
        """
        # RETURNING 절은 삽입된 행의 특정 컬럼 값을 반환하도록 지시합니다.
        query = """
            INSERT INTO ai_test_image_generation_queue (work_id, persona_id, report_text, title, panel_descriptions, image_prompts, scheduled_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING job_id;
        """
        # image_prompts 리스트를 JSON 문자열로 변환
        image_prompts_json = json.dumps(image_prompts, ensure_ascii=False)
        panel_descriptions_json = json.dumps(panel_descriptions, ensure_ascii=False)
        try:
            record = await self.db.fetch_one(query, work_id, persona_id, report_text, title, panel_descriptions_json, image_prompts_json, scheduled_at)
            if record and 'job_id' in record:
                job_id = record['job_id']
                self.logger.info(f"DB에서 반환된 Job ID: {job_id}", extra={"work_id": work_id})
                return job_id
            else:
                self.logger.error("DB INSERT 후 job_id를 반환받지 못했습니다.", extra={"work_id": work_id})
                return None
        except Exception as e:
            self.logger.error(f"DB 작업 큐 삽입 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})
            return None

    async def _finalize_and_save_state(self, workflow_state: OverallWorkflowState, log_extra: Dict) -> Dict[str, Any]:
        """최종 상태를 저장하고 반환합니다."""
        updated_state_dict = workflow_state.model_dump(mode='json')
        await self._save_workflow_state_to_redis(workflow_state.work_id, updated_state_dict)
        self.logger.info("N08 노드 처리 완료 및 상태 저장.", extra=log_extra)
        return updated_state_dict

    async def _save_workflow_state_to_redis(self, work_id: str, state_dict: Dict[str, Any]):
        """워크플로우 상태를 Redis에 저장합니다."""
        key = f"workflow:{work_id}:full_state"
        try:
            json_compatible_state = json.loads(json.dumps(state_dict, default=str))
            await self.redis.set(key, json_compatible_state, expire=60 * 60 * 6)
        except Exception as e:
            self.logger.error(f"Redis 상태 저장 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})