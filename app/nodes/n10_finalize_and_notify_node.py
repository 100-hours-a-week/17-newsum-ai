# ai/app/nodes/n10_finalize_and_notify_node.py
import traceback
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
import asyncio
import aiohttp # 외부 API 호출용
import json # 외부 API 페이로드용
from pathlib import Path
import mimetypes # 로컬 파일 ContentType 추측용

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging, PROJECT_ROOT
from app.config.settings import Settings
# 필요한 서비스 임포트
from app.services.storage_service import StorageService
from app.services.translation_service import TranslationService

logger = get_logger(__name__)
settings = Settings()

# 로컬 이미지 저장 기본 경로 (N09에서 사용된 경로와 일치해야 함)
LOCAL_IMAGE_BASE_DIR_N10 = PROJECT_ROOT / "results"
# 동시 S3 업로드 수 제한 (설정 또는 기본값)
MAX_PARALLEL_S3_UPLOADS = settings.S3_MAX_PARALLEL_UPLOADS or 5

class N10FinalizeAndNotifyNode:
    """
    워크플로우 최종 단계: 생성된 이미지 S3 업로드, 보고서 번역, 참조 링크 추출,
    그리고 최종 결과 데이터를 외부 API로 전송합니다.
    """

    def __init__(
        self,
        storage_service: StorageService,
        translation_service: TranslationService,
        # 외부 API 호출을 위한 aiohttp 세션 (lifespan에서 관리/주입 권장)
        http_session: Optional[aiohttp.ClientSession] = None
    ):
        """
        노드 초기화.

        Args:
            storage_service (StorageService): S3 상호작용 서비스.
            translation_service (TranslationService): 번역 서비스.
            http_session (Optional[aiohttp.ClientSession]): 외부 API 호출용 HTTP 세션.
                                                            None이면 내부적으로 생성/관리.
        """
        self.storage_service = storage_service
        self.translation_service = translation_service

        # 외부 API 호출용 세션 관리
        self._session = http_session
        self._created_session = False
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=settings.EXTERNAL_API_TIMEOUT_SECONDS or 30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._created_session = True
            logger.info("N10: Created internal aiohttp session for external API calls.")

    async def _upload_image_to_s3(
        self,
        image_info: Dict[str, Any], # N09 결과 항목
        comic_id: str,
        s3_prefix: str,
        acl: Optional[str],
        trace_id: str,
        extra_log_data: dict
    ) -> Dict[str, Optional[str]]:
        """단일 이미지를 로컬 경로에서 S3로 업로드하고 S3 URI/URL 반환"""
        result = {
            "scene_identifier": image_info.get("scene_identifier", "Unknown Scene"),
            "s3_url": None, # S3 접근 가능 URL 또는 URI
            "error": image_info.get("error") # N09 오류 우선 전달
        }
        if result["error"]:
            logger.warning(f"Skipping S3 upload for {result['scene_identifier']} due to previous error: {result['error']}", extra=extra_log_data)
            return result

        local_image_path = image_info.get("image_path")
        if not local_image_path:
            # 로컬 경로 없이 URL만 있는 경우 (처리 방식 결정 필요)
            if image_info.get("image_url"):
                 result["error"] = "Cannot upload to S3: Local path missing, only URL available."
                 # result["s3_url"] = image_info["image_url"] # 또는 URL 사용?
            else:
                 result["error"] = "Local image path is missing and no URL found."
            logger.warning(result["error"], extra=extra_log_data)
            return result

        local_file = Path(local_image_path)
        if not local_file.is_file():
            result["error"] = f"Local image file not found at path: {local_image_path}"
            logger.warning(result["error"], extra=extra_log_data)
            return result

        # S3 객체 키 생성 (경로 구분자 통일 등 고려)
        filename = local_file.name
        # Posix 경로 형태로 변환하여 S3 키 생성
        object_key = Path(s3_prefix.strip('/')) / comic_id / "images" / filename
        object_key_str = object_key.as_posix() # S3 키는 '/' 구분자 사용

        content_type, _ = mimetypes.guess_type(local_image_path)
        content_type = content_type or 'image/png'

        try:
            # StorageService 사용하여 업로드
            upload_result = await self.storage_service.upload_file(
                file_path=str(local_file),
                object_key=object_key_str,
                content_type=content_type,
                acl=acl
            )

            if "error" not in upload_result:
                # StorageService가 s3_uri를 반환하면 사용, 아니면 퍼블릭 URL 구성
                result["s3_url"] = upload_result.get("s3_uri")
                # 필요 시 퍼블릭 URL 구성 로직 추가
                if not result["s3_url"] and acl == 'public-read' and self.storage_service.s3_client:
                     bucket = self.storage_service.bucket_name
                     # region = self.storage_service.region_name or self.storage_service.s3_client.meta.region_name
                     # Boto3 최신 버전에서는 region_name 직접 접근이 다를 수 있음, config에서 가져오는 것이 안전
                     region = self.storage_service.region_name or settings.AWS_REGION or 'ap-northeast-2'
                     result["s3_url"] = f"https://{bucket}.s3.{region}.amazonaws.com/{object_key_str}"

                logger.info(f"Successfully uploaded {result['scene_identifier']} image to S3: {result['s3_url']}", extra=extra_log_data)
            else:
                result["error"] = f"S3 upload failed via StorageService: {upload_result['error']}"
                logger.warning(result["error"], extra=extra_log_data)

        except Exception as e:
            error_msg = f"Unexpected error uploading image {local_image_path} to S3: {e}"
            logger.exception(error_msg, extra=extra_log_data)
            result["error"] = error_msg

        return result

    async def _translate_report(
        self,
        report_html: str,
        target_lang: str,
        trace_id: str,
        extra_log_data: dict
    ) -> Optional[str]:
        """보고서 HTML 내용을 번역합니다."""
        if not self.translation_service or not self.translation_service.is_enabled:
            logger.warning("Translation service is disabled or not available, skipping report translation.", extra=extra_log_data)
            return None
        if not report_html:
            logger.warning("Report content is empty, skipping translation.", extra=extra_log_data)
            return None

        source_lang = "ko" # 원본 보고서 언어 가정
        logger.info(f"Translating report content from '{source_lang}' to '{target_lang}'...", extra=extra_log_data)

        try:
            # TranslationService의 translate 메소드 호출 (내부에서 aiohttp 세션 처리 가정)
            # 제공된 translation_service.py는 외부 세션을 받으므로 전달 필요
            if not self._session or self._session.closed:
                 logger.error("Cannot translate report: N10 http session is closed or unavailable.", extra=extra_log_data)
                 return None

            translated_html = await self.translation_service.translate(
                text=report_html,
                source_lang=source_lang,
                target_lang=target_lang,
                session=self._session, # N10이 관리하는 세션 전달
                trace_id=trace_id
            )
            if translated_html:
                logger.info(f"Report content translated successfully to '{target_lang}'.", extra=extra_log_data)
                return translated_html
            else:
                logger.warning("Report translation failed or returned empty.", extra=extra_log_data)
                return None
        except Exception as e:
            logger.exception(f"Unexpected error during report translation: {e}", extra=extra_log_data)
            return None

    def _extract_referenced_urls(self, state: WorkflowState) -> List[str]:
        """보고서 생성에 사용된 참조 URL 목록을 추출합니다."""
        urls: Set[str] = set()
        # N04 검색 결과에서 추출
        if state.raw_search_results and isinstance(state.raw_search_results, list):
            for item in state.raw_search_results:
                if isinstance(item, dict) and item.get("url") and isinstance(item.get("url"), str):
                    # file:// 스키마 제외 (로컬 경로)
                    if not item["url"].startswith("file://"):
                         urls.add(item["url"])
        # N05 sources에서 추출 (만약 state에 저장했다면)
        # ...

        logger.info(f"Extracted {len(urls)} unique referenced URLs.")
        return sorted(list(urls))

    async def _send_to_external_api(
        self,
        payload: Dict[str, Any],
        api_url: str,
        api_token: Optional[str],
        trace_id: str,
        extra_log_data: dict
    ) -> Dict[str, Any]:
        """구성된 데이터를 외부 API로 전송합니다."""
        if not api_url:
            logger.warning("External notification API URL not configured. Skipping notification.", extra=extra_log_data)
            return {"status": "skipped", "reason": "API URL not configured"}

        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        logger.info(f"Sending final data to external API: {api_url}", extra=extra_log_data)
        logger.debug(f"External API Payload Summary: {summarize_for_logging(payload, max_len=500)}", extra=extra_log_data)

        try:
            if not self._session or self._session.closed:
                 logger.error("Cannot send to external API: N10 http session is closed or unavailable.", extra=extra_log_data)
                 return {"status": "failed", "error": "HTTP session not available"}

            async with self._session.post(api_url, headers=headers, json=payload) as response:
                response_text = await response.text()
                status_code = response.status
                if 200 <= status_code < 300:
                    logger.info(f"Successfully sent data to external API (Status: {status_code}).")
                    try:
                         return {"status": "success", "response_status": status_code, "response_body": json.loads(response_text)}
                    except json.JSONDecodeError:
                         return {"status": "success", "response_status": status_code, "response_body": response_text}
                else:
                    logger.error(f"Failed to send data to external API (Status: {status_code}): {response_text[:500]}", extra=extra_log_data)
                    return {"status": "failed", "response_status": status_code, "error": response_text}
        except aiohttp.ClientConnectorError as e:
             logger.error(f"Connection error sending data to external API: {e}", extra=extra_log_data)
             return {"status": "failed", "error": f"Connection error: {e}"}
        except asyncio.TimeoutError:
            logger.error(f"Timeout sending data to external API: {api_url}", extra=extra_log_data)
            return {"status": "failed", "error": "Request timed out"}
        except Exception as e:
            logger.exception(f"Unexpected error sending data to external API: {e}", extra=extra_log_data)
            return {"status": "failed", "error": f"Unexpected error: {e}"}

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')
        error_log = list(state.error_log or [])

        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        logger.info(f"Entering node {node_name}. Finalizing workflow results and notifying external system.", extra=extra)

        # 결과 변수 초기화
        s3_upload_results: List[Dict[str, Optional[str]]] = []
        translated_report: Optional[str] = None
        referenced_urls_list: List[str] = []
        external_api_call_result: Optional[Dict[str, Any]] = None

        if not comic_id:
            logger.error("Critical: comic_id is missing. Cannot perform finalization tasks.", extra=extra)
            error_log.append({"stage": node_name, "error": "comic_id missing", "timestamp": datetime.now(timezone.utc).isoformat()})
            return { "current_stage": "ERROR", "error_log": error_log, "error_message": f"{node_name} Error: comic_id missing."}

        try:
            # --- 1. 이미지 S3 업로드 (병렬 처리) ---
            s3_prefix = settings.S3_IMAGE_PREFIX or "generated-comics"
            acl_setting = settings.S3_IMAGE_ACL # 설정값 (None일 수 있음)
            upload_tasks = []
            if state.generated_comic_images and isinstance(state.generated_comic_images, list):
                for image_info in state.generated_comic_images:
                    if isinstance(image_info, dict):
                        upload_tasks.append(
                            self._upload_image_to_s3(
                                image_info, comic_id, s3_prefix, acl_setting, trace_id, extra
                            )
                        )

            if upload_tasks:
                logger.info(f"Starting {len(upload_tasks)} S3 upload tasks (max parallel: {MAX_PARALLEL_S3_UPLOADS})...", extra=extra)
                semaphore = asyncio.Semaphore(MAX_PARALLEL_S3_UPLOADS)
                async def run_upload_with_semaphore(task_coro):
                    async with semaphore:
                        return await task_coro
                s3_results = await asyncio.gather(*(run_upload_with_semaphore(task) for task in upload_tasks), return_exceptions=True)
                # 결과를 s3_upload_results에 채우고 오류 로깅
                for result in s3_results:
                    if isinstance(result, Exception):
                        err_msg = f"An S3 upload task failed: {result}"
                        logger.error(err_msg, extra=extra)
                        error_log.append({"stage": f"{node_name}.s3_upload_exception", "error": str(result), "timestamp": datetime.now(timezone.utc).isoformat()})
                        s3_upload_results.append({"scene_identifier": "Unknown (upload exception)", "s3_url": None, "error": str(result)})
                    elif isinstance(result, dict):
                        s3_upload_results.append(result)
                        if result.get("error"):
                             error_log.append({"stage": f"{node_name}.s3_upload_error", "scene": result.get("scene_identifier"), "error": result["error"], "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                logger.info("No images found or processed in N09 to upload to S3.", extra=extra)


            # --- 2. 보고서 번역 ---
            if state.report_content:
                target_lang = settings.REPORT_TRANSLATION_TARGET_LANGUAGE or "en"
                translated_report = await self._translate_report(
                    state.report_content, target_lang, trace_id, extra
                )
                if not translated_report:
                     error_log.append({"stage": f"{node_name}._translate_report", "error": "Report translation failed.", "timestamp": datetime.now(timezone.utc).isoformat()})


            # --- 3. 참조 링크 목록 생성 ---
            referenced_urls_list = self._extract_referenced_urls(state)


            # --- 4. 외부 API 전송 ---
            external_api_url = settings.EXTERNAL_NOTIFICATION_API_URL
            external_api_token = settings.EXTERNAL_NOTIFICATION_API_TOKEN

            if external_api_url:
                # 외부 API 페이로드 구성
                api_payload = {
                    "comicId": comic_id, "traceId": trace_id,
                    "status": "completed" if not error_log else "completed_with_errors", # 전체 오류 로그 확인
                    "originalQuery": state.original_query, "writerId": writer_id,
                    "reportTranslatedHtml": translated_report,
                    "referencedUrls": referenced_urls_list,
                    "generatedImages": s3_upload_results, # S3 URL 또는 오류 정보 포함
                    "scenarioText": state.comic_scenarios[0].get("scenario_text") if state.comic_scenarios else None, # 시나리오 텍스트만 전달
                    "ideaTitle": state.selected_comic_idea_for_scenario.get("title") if state.selected_comic_idea_for_scenario else None,
                    "finalTimestamp": datetime.now(timezone.utc).isoformat()
                }
                external_api_call_result = await self._send_to_external_api(
                    api_payload, external_api_url, external_api_token, trace_id, extra
                )
                if external_api_call_result.get("status") == "failed":
                    error_log.append({
                        "stage": f"{node_name}._send_to_external_api",
                        "error": f"External API notification failed: {external_api_call_result.get('error', 'Unknown reason')}",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            else:
                logger.info("External notification API URL not configured. Skipping notification.", extra=extra)
                external_api_call_result = {"status": "skipped", "reason": "API URL not configured"}


            # --- 최종 상태 업데이트 ---
            # N10에서 발생한 오류가 있는지 확인하여 최종 상태 결정
            node10_errors = [log for log in error_log if log.get("stage", "").startswith(node_name)]
            final_stage = "DONE" if not node10_errors else "DONE_WITH_ERRORS"

            update_dict = {
                "uploaded_image_urls": s3_upload_results,
                "translated_report_content": translated_report,
                "referenced_urls": referenced_urls_list,
                "external_api_response": external_api_call_result,
                "current_stage": final_stage, # 최종 완료 또는 오류 상태
                "error_log": error_log # 모든 오류 누적
            }

            # 로깅 요약
            log_summary_update = {
                "current_stage": final_stage,
                "uploaded_image_count": len(s3_upload_results),
                "images_upload_errors": sum(1 for img in s3_upload_results if img.get("error")),
                "translation_successful": bool(translated_report),
                "referenced_urls_count": len(referenced_urls_list),
                "external_api_status": external_api_call_result.get("status") if external_api_call_result else "skipped"
            }

            logger.info(
                 f"Exiting node {node_name}. Finalization complete. Output Update Summary: {summarize_for_logging(log_summary_update)}",
                extra=extra
            )
            return update_dict

        except Exception as e:
            error_msg = f"Unexpected error in {node_name} execution: {e}"
            logger.exception(error_msg, extra=extra)
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})

            # 예외 발생 시에도 외부 API 알림 시도 (오류 상태로)
            if external_api_url and not external_api_call_result:
                error_payload = {"comicId": comic_id, "traceId": trace_id, "status": "failed", "error_message": error_msg}
                await self._send_to_external_api(error_payload, external_api_url, external_api_token, trace_id, extra)

            return {
                "uploaded_image_urls": s3_upload_results, # 부분 결과 포함 가능
                "translated_report_content": translated_report,
                "referenced_urls": referenced_urls_list,
                "external_api_response": external_api_call_result or {"status": "failed", "error": "Workflow failed before/during API call"},
                "error_log": error_log,
                "current_stage": "ERROR",
                "error_message": f"{node_name} Exception: {error_msg}"
            }
        finally:
            # 노드 종료 시 내부 생성된 aiohttp 세션 닫기
             if hasattr(self, '_session') and self._session and self._created_session and not self._session.closed:
                 await self._session.close()
                 logger.info(f"{node_name}: Closed internal aiohttp session.")