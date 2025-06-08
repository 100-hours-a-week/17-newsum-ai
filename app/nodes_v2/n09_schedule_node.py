# ai/app/nodes_v2/n09_schedule_node.py

import uuid  # comic_id (work_id)를 UUID로 변환하기 위함
from typing import Dict, Any, List  # List 타입 힌트 추가

from app.workflows.state_v2 import WorkflowState  # ImageSection 모델 임포트
from app.utils.logger import get_logger  # summarize_for_logging 사용
from app.services.postgresql_service import PostgreSQLService  # PostgreSQL 서비스 임포트

logger = get_logger(__name__)

# 이 노드의 워크플로우 내 순서 정의
NODE_ORDER = 13  # 또는 N08a 이후의 적절한 순서 (예: 8.2 또는 9)


class N09ScheduleNode:
    """
    (최종 수정) ImageSection의 refined_prompts를 PostgreSQL DB의
    ai_test_image_generation_queue 테이블에 저장하여 이미지 생성을 예약하는 노드입니다.
    state_v2.py 호환성 및 명확한 로깅을 적용합니다.
    """

    def __init__(self, postgresql_service: PostgreSQLService):
        """
        노드 초기화.

        Args:
            postgresql_service (PostgreSQLService): PostgreSQL 서비스 인스턴스.
        """
        self.pg_service = postgresql_service
        logger.info(f"{self.__class__.__name__} 초기화 완료.")

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        """
        이미지 생성 작업을 DB에 예약합니다.
        """
        meta_sec = state.meta
        image_sec = state.image  # ImageSection에서 정제된 프롬프트 읽기
        # config_sec_dict = state.config.config or {} # 필요시 설정값 사용

        node_name = self.__class__.__name__
        work_id_str = meta_sec.work_id  # MetaSection의 work_id 사용
        # writer_id = config_sec_dict.get("writer_id", "default_writer") # 로깅 등에 필요시 사용

        extra_log_base = {'work_id': work_id_str, 'node_name': node_name, 'node_order': NODE_ORDER}

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"  # 노드 처리 시작 상태
        logger.info(f"노드 진입: 이미지 생성 작업 DB 예약 시작.", extra=extra_log_base)

        if not work_id_str:
            error_msg = "작업 ID (work_id)가 누락되어 이미지 생성을 예약할 수 없습니다."
            logger.error(error_msg, extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            # image_sec은 변경 없으므로 반환 불필요, meta만 반환
            return {"meta": meta_sec.model_dump()}

        try:
            # work_id (이전 comic_id)를 UUID 객체로 변환
            work_id_uuid = uuid.UUID(work_id_str)
        except ValueError:
            error_msg = f"유효하지 않은 작업 ID (work_id) 형식입니다: {work_id_str}"
            logger.error(error_msg, extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            return {"meta": meta_sec.model_dump()}

        # N08a에서 정제된 프롬프트 목록 가져오기
        prompts_to_schedule: List[Dict[str, Any]] = image_sec.refined_prompts or []

        if not prompts_to_schedule:
            logger.warning("DB에 예약할 정제된 프롬프트가 없습니다. N08a 노드 결과를 확인하세요.", extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"  # 작업할 내용이 없으므로 완료 (또는 SKIPPED)
            # image_sec.generated_comic_images는 이 노드에서 채우지 않음
            return {"meta": meta_sec.model_dump()}

        try:
            # PostgreSQLService의 schedule_image_generation 메서드 호출
            # 이 메서드는 내부적으로 bulk insert를 수행해야 함
            # (comic_id, scene_identifier, prompt_used, model_name 등을 인자로 받음)
            await self.pg_service.schedule_image_generation(work_id_uuid, prompts_to_schedule)  #

            num_scheduled = len(prompts_to_schedule)
            logger.info(f"총 {num_scheduled}개의 이미지 생성 작업이 DB에 성공적으로 예약되었습니다 (작업 ID: {work_id_str}).",
                        extra=extra_log_base)

            meta_sec.workflow_status[NODE_ORDER] = "SCHEDULED"  # '예약됨' 상태로 명시적 변경
            # meta_sec.next_action 등은 LangGraph의 라우팅 로직에서 처리 (여기서는 직접 설정 안 함)

            # 이 노드는 image_sec.generated_comic_images를 직접 업데이트하지 않으므로,
            # image 섹션을 반환할 필요는 없음 (또는 변경 없음을 명시).
            # LangGraph는 반환된 딕셔너리를 상태에 병합함.
            return {"meta": meta_sec.model_dump()}

        except Exception as e:
            error_msg = f"이미지 생성 작업 DB 예약 중 예상치 못한 오류 발생: {e}"
            # traceback.format_exc() 대신 exc_info=True 사용
            logger.error(error_msg, extra=extra_log_base, exc_info=True)

            meta_sec.workflow_status[NODE_ORDER] = "ERROR"
            # state_v2.MetaSection에 error_log 필드가 없으므로, 별도 저장 로직 불필요.
            # meta_sec.error_message도 사용 안 함.
            return {"meta": meta_sec.model_dump()}