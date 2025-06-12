# app/nodes_v3/n_08_queue_for_image_generation_node.py

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import os

from pydantic import ValidationError
import aiohttp  # 외부 API 호출용

# --- 애플리케이션 구성 요소 임포트 ---
from app.services.postgresql_service import PostgreSQLService
from app.services.database_client import DatabaseClient  # RedisClient 용도
from app.utils.logger import get_logger
from app.workflows.state_v3 import (
    OverallWorkflowState,
    ImagePromptItemPydantic,
    ImageQueueState
)
from app.config.settings import settings  # settings 임포트 추가
from app.services.storage_service import StorageService

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

        # --- API 전송용 데이터 추출 및 dummy 값 보완 ---
        try:
            # 반드시 필요한 값 체크
            thumbnail = image_prompts_state.thumbnail_prompt
            panels = image_prompts_state.panels
            image_concept_state = workflow_state.image_concept
            persona_id = persona_analysis_state.selected_opinion.persona_id if persona_analysis_state.selected_opinion else None
            report_draft = report_draft_state.draft

            # 필수 값 체크
            if not thumbnail:
                raise ValueError("썸네일 프롬프트(thumbnail_prompt)가 존재하지 않습니다.")
            if not panels or len(panels) == 0:
                raise ValueError("패널 프롬프트(panels)가 존재하지 않습니다.")
            if not report_draft_state.category:
                raise ValueError(f"category 값이 없음: category={report_draft_state.category}")
            if not report_draft_state.keywords:
                raise ValueError(f"keywords 값이 없음: keywords={report_draft_state.keywords}")
            if not image_concept_state.final_thumbnail:
                raise ValueError("최종 썸네일 콘셉트(final_thumbnail)가 존재하지 않습니다.")
            if not image_concept_state.final_concepts or len(image_concept_state.final_concepts) < 4:
                raise ValueError("최종 패널 콘셉트(final_concepts)가 4개 미만입니다.")
            if not persona_id:
                raise ValueError("선택된 페르소나(persona_analysis.selected_opinion)가 존재하지 않습니다.")
            if not report_draft:
                raise ValueError("보고서 초안(report_draft.draft)이 존재하지 않습니다.")

            all_prompts = [thumbnail.model_dump()]
            all_prompts.extend([p.model_dump() for p in panels])
            title = image_concept_state.final_thumbnail.caption
            panel_descriptions = [c.caption for c in image_concept_state.final_concepts]
            category = report_draft_state.category
            keyword = ", ".join(report_draft_state.keywords)
            content = report_draft[:200] if report_draft else "dummy_content"  # 요약 구현 필요

            # 1. HTML 파일 경로 결정 (settings에서 읽음)
            html_file_path = os.path.join(settings.REPORT_HTML_OUTPUT_DIR, f"{work_id}.html")
            storage_service = StorageService()
            report_url = "https://dummy.com/report.html"  # 기본값

            # 2. 파일이 존재하면 S3 업로드 및 CloudFront URL 획득
            if os.path.exists(html_file_path):
                upload_result = await storage_service.upload_file_with_cloudfront_url(
                    file_path=html_file_path,
                    object_key=f"reports/{work_id}.html",
                    content_type="text/html"
                )
                if upload_result.get("cloudfront_url"):
                    report_url = upload_result["cloudfront_url"]
                else:
                    raise ValueError(f"CloudFront URL 생성 실패: {upload_result}")
            else:
                raise ValueError(f"HTML 파일이 존재하지 않음: {html_file_path}")

            payload = {
                "work_id": work_id,
                "ai_author_id": persona_id,
                "keyword": keyword,     # 쉼표로 join된 문자열
                "category": category,   # POLITICS, IT, FINANCE 중 하나
                "title": title,
                "reportUrl": report_url,
                "content": content,     # TODO: 요약 구현 필요
                "description1": panel_descriptions[0],
                "description2": panel_descriptions[1],
                "description3": panel_descriptions[2],
                "description4": panel_descriptions[3],
                "imagePrompts": all_prompts
            }
        except Exception as e:
            error_msg = f"API 전송용 데이터 추출 중 오류: {e}"
            self.logger.error(error_msg, extra=log_extra)
            node_state.error_message = error_msg
            return await self._finalize_and_save_state(workflow_state, log_extra)

        # --- 외부 API 호출 ---
        api_url = settings.EXTERNAL_NOTIFICATION_API_URL
        if not api_url:
            node_state.error_message = "EXTERNAL_NOTIFICATION_API_URL이 설정되어 있지 않습니다."
            self.logger.error(node_state.error_message, extra=log_extra)
            workflow_state.insert_image_queue = node_state
            return await self._finalize_and_save_state(workflow_state, log_extra)
        api_result = await self._send_image_prompt_to_api(payload, api_url, log_extra)
        if api_result.get("status") == "success":
            node_state.is_ready = True
            node_state.job_id = None  # 외부 API에서 반환하는 값이 있으면 할당
            self.logger.info(f"이미지 생성 API 호출 성공.", extra=log_extra)
        else:
            node_state.error_message = f"API 호출 실패: {api_result.get('error')}"
            self.logger.error(node_state.error_message, extra=log_extra)

        workflow_state.insert_image_queue = node_state
        return await self._finalize_and_save_state(workflow_state, log_extra)

    async def _send_image_prompt_to_api(self, payload: dict, api_url: str, log_extra: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(api_url, headers=headers, json=payload) as resp:
                    resp_text = await resp.text()
                    status = resp.status
                    if 200 <= status < 300:
                        self.logger.info(f"N08: API 호출 성공 (Status: {status})", extra=log_extra)
                        return {"status": "success", "response": resp_text}
                    else:
                        self.logger.error(f"N08: API 호출 실패 (Status: {status}): {resp_text}", extra=log_extra)
                        return {"status": "failed", "error": resp_text}
            except Exception as e:
                self.logger.error(f"N08: API 호출 중 예외 발생: {e}", extra=log_extra)
                return {"status": "failed", "error": str(e)}

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