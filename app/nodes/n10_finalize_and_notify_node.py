# ai/app/nodes/n10_finalize_and_notify_node.py
import traceback
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
import asyncio
import aiohttp  # 외부 API 호출용
import json  # 외부 API 페이로드용
from pathlib import Path
import mimetypes  # 로컬 파일 ContentType 추측용
import os  # main 테스트에서 경로 작업 및 파일 생성/삭제를 위해 추가
import shutil  # main 테스트에서 디렉토리 삭제를 위해 추가
import uuid  # main 테스트에서 고유 ID 생성을 위해 추가

# --- 실제 애플리케이션 환경에 맞게 설정되어야 하는 임포트 ---
from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging, PROJECT_ROOT
from app.config.settings import Settings
from app.services.storage_service import StorageService
# Google API 키 방식의 번역 서비스로 변경
from app.services.google_translation_service import GoogleRestTranslationService as TranslationService

# --- 실제 애플리케이션 환경에 맞게 설정되어야 하는 임포트 끝 ---

logger = get_logger(__name__)
settings = Settings()  # 전역 settings 객체 사용

# 동시 S3 업로드 수 제한 (설정 또는 기본값)
MAX_PARALLEL_S3_UPLOADS = settings.S3_MAX_PARALLEL_UPLOADS or 5


class N10FinalizeAndNotifyNode:
    """
    워크플로우 최종 단계: 생성된 이미지 S3 업로드, 보고서 번역, 참조 링크 추출,
    그리고 최종 결과 데이터를 외부 API로 전송합니다.
    (ACL 미사용, GoogleRestTranslationService 사용, 썸네일 처리 반영)
    """

    def __init__(
            self,
            storage_service: StorageService,
            translation_service: TranslationService,  # GoogleRestTranslationService 인스턴스
            http_session: Optional[aiohttp.ClientSession] = None  # 외부 API 알림용 세션
    ):
        self.storage_service = storage_service
        self.translation_service = translation_service
        self._external_api_session = http_session  # 외부 알림 API 전용 세션
        self._created_external_api_session = False  # N10이 직접 세션을 만들었는지 여부

        # GoogleRestTranslationService는 자체 httpx 클라이언트를 관리하므로,
        # N10의 aiohttp 세션은 외부 알림 API에만 사용됩니다.
        if self._external_api_session is None and settings.EXTERNAL_NOTIFICATION_API_URL:
            timeout_seconds = settings.EXTERNAL_API_TIMEOUT_SECONDS if hasattr(settings,
                                                                               'EXTERNAL_API_TIMEOUT_SECONDS') else 30
            timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
            self._external_api_session = aiohttp.ClientSession(timeout=timeout)
            self._created_external_api_session = True
            logger.info(
                f"N10: Created internal aiohttp session for EXTERNAL notification API (timeout: {timeout_seconds}s).")

    async def _upload_image_to_s3(
            self,
            image_info: Dict[str, Any],
            comic_id: str,
            s3_base_prefix: str,  # 예: "comics"
            trace_id: str,
            extra_log_data: dict
    ) -> Dict[str, Optional[str]]:
        # N09에서 전달된 is_thumbnail 플래그 확인
        is_thumbnail = image_info.get("is_thumbnail", False)
        scene_identifier = image_info.get("scene_identifier", "thumbnail" if is_thumbnail else "UnknownScene")

        result = {
            "scene_identifier": scene_identifier,
            "s3_url": None,
            "is_thumbnail": is_thumbnail,  # 결과에 썸네일 여부 포함
            "error": image_info.get("error")  # N09의 오류 우선 전달
        }

        if result["error"]:
            log_target = "thumbnail" if is_thumbnail else f"scene '{scene_identifier}'"
            logger.warning(
                f"N10: Skipping S3 upload for {log_target} due to previous error from N09: {result['error']}",
                extra=extra_log_data)
            return result

        local_image_path_str = image_info.get("image_path")
        remote_image_url = image_info.get("image_url")  # ImageService가 URL을 반환한 경우

        if local_image_path_str:
            local_file = Path(local_image_path_str)
            if not local_file.is_file():
                result["error"] = f"Local image file not found at path: {local_image_path_str}"
                logger.warning(result["error"], extra=extra_log_data)
                return result

            filename = local_file.name
            # 썸네일인 경우 S3 경로에 "thumbnail" 디렉토리 사용, 아니면 "images" 사용
            if is_thumbnail:
                object_key_path = Path(s3_base_prefix.strip('/')) / comic_id / "thumbnail" / filename
            else:
                object_key_path = Path(s3_base_prefix.strip('/')) / comic_id / "images" / filename
            object_key_str = object_key_path.as_posix()  # S3는 '/' 구분자 사용

            content_type, _ = mimetypes.guess_type(local_image_path_str)
            # 일반적인 이미지 타입으로 기본값 설정
            content_type = content_type or (
                'image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png')

            try:
                # StorageService 사용하여 업로드 (acl=None 으로 전달하여 ACL 미사용 명시)
                upload_result = await self.storage_service.upload_file(
                    file_path=str(local_file),
                    object_key=object_key_str,
                    content_type=content_type,
                    acl=None  # ACL 명시적으로 사용 안함
                )

                # StorageService의 upload_file 메소드가 ACL 미지원 시 warning을 포함하여 반환할 수 있음
                if not upload_result.get("error"):
                    result["s3_url"] = upload_result.get("s3_uri")  # StorageService가 반환하는 s3_uri 사용
                    log_msg_prefix = "Thumbnail" if is_thumbnail else f"Scene '{scene_identifier}'"

                    if upload_result.get("warning"):  # ACL 미지원 등으로 경고가 있을 수 있음
                        logger.warning(
                            f"N10: S3 Upload for {log_msg_prefix} completed with warning: {upload_result['warning']}. URL: {result['s3_url']}",
                            extra=extra_log_data)
                    else:
                        logger.info(f"N10: {log_msg_prefix} image uploaded to S3: {result['s3_url']}",
                                    extra=extra_log_data)
                else:
                    err_detail = upload_result['error']
                    result["error"] = f"S3 upload failed via StorageService: {err_detail}"
                    logger.warning(result["error"], extra=extra_log_data)

            except Exception as e:
                error_msg = f"N10: Unexpected error uploading image {local_image_path_str} to S3: {type(e).__name__} - {e}"
                logger.exception(error_msg, extra=extra_log_data)  # 스택 트레이스 포함 로깅
                result["error"] = error_msg

        elif remote_image_url:
            # 로컬 경로가 없고 ImageService가 직접 URL을 반환한 경우
            log_target = "thumbnail" if is_thumbnail else f"scene '{scene_identifier}'"
            logger.info(
                f"N10: Using remote image URL directly for {log_target} (no S3 upload from N10): {remote_image_url}",
                extra=extra_log_data)
            result["s3_url"] = remote_image_url  # 이 URL을 S3 URL로 간주 (또는 최종 URL)
        else:
            result["error"] = "Neither local image path nor remote image URL is available for N10 processing."
            logger.warning(f"N10: {result['error']} for {'thumbnail' if is_thumbnail else scene_identifier}",
                           extra=extra_log_data)

        return result

    async def _translate_report(
            self,
            report_html: str,
            target_lang: str,
            trace_id: str,
            extra_log_data: dict
    ) -> Optional[str]:
        if not self.translation_service or not self.translation_service.is_enabled:
            logger.warning(
                "Google Translation service (REST API Key) is disabled or not available, skipping report translation.",
                extra=extra_log_data)
            return None
        if not report_html:
            logger.warning("Report content is empty, skipping translation.", extra=extra_log_data)
            return None

        # Google Translation은 source_lang을 자동으로 감지할 수 있으므로 "ko" 대신 None 전달도 가능
        source_lang_for_request = "ko"
        logger.info(
            f"Translating report content from '{source_lang_for_request}' to '{target_lang}' using Google Translate (REST API Key)...",
            extra=extra_log_data)

        try:
            # GoogleRestTranslationService의 translate 메소드 호출
            # HTML 번역을 위해 text 인자에 HTML 문자열 전달, 서비스 내에서 format_='html' 사용 가정
            translated_html = await self.translation_service.translate(
                text=report_html,
                source_lang=source_lang_for_request,  # 또는 None
                target_lang=target_lang,
                trace_id=trace_id
            )
            if translated_html:
                logger.info(
                    f"Report content translated successfully to '{target_lang}' using Google Translate (REST API Key).",
                    extra=extra_log_data)
            else:
                # translate 메소드가 None을 반환하면 (오류 포함하여) 여기서 로깅
                logger.warning("Report translation using Google Translate (REST API Key) failed or returned empty.",
                               extra=extra_log_data)
            return translated_html
        except Exception as e:  # Tenacity 재시도 후 최종 실패 시 예외가 다시 발생할 수 있음
            logger.exception(f"Google REST report translation failed after retries in N10: {type(e).__name__} - {e}",
                             extra=extra_log_data)
            return None  # 최종 실패 시 None 반환

    def _extract_referenced_urls(self, state: WorkflowState) -> List[str]:
        urls: Set[str] = set()
        if state.raw_search_results and isinstance(state.raw_search_results, list):
            for item in state.raw_search_results:
                if isinstance(item, dict) and item.get("url") and isinstance(item.get("url"), str):
                    if not item["url"].startswith("file://"):  # 로컬 파일 경로 스키마 제외
                        urls.add(item["url"])
        logger.info(f"Extracted {len(urls)} unique referenced URLs from raw_search_results.")
        return sorted(list(urls))

    async def _send_to_external_api(
            self,
            payload: Dict[str, Any],
            api_url: str,
            api_token: Optional[str],
            trace_id: str,
            extra_log_data: dict
    ) -> Dict[str, Any]:
        if not api_url:  # settings.EXTERNAL_NOTIFICATION_API_URL이 None이거나 비어있을 경우
            logger.warning("External notification API URL not configured. Skipping notification.", extra=extra_log_data)
            return {"status": "skipped", "reason": "API URL not configured"}

        headers = {"Content-Type": "application/json"}
        if api_token:  # settings.EXTERNAL_NOTIFICATION_API_TOKEN
            headers["Authorization"] = f"Bearer {api_token}"

        logger.info(f"Sending final data to external API: {api_url}", extra=extra_log_data)
        logger.debug(f"External API Payload Summary: {summarize_for_logging(payload, max_len=1000)}",
                     extra=extra_log_data)

        session_to_use = self._external_api_session
        internal_session_created_for_this_call = False  # 이 호출을 위해 내부 세션이 생성되었는지 여부
        if not session_to_use or session_to_use.closed:
            # 주입된 세션이 없거나 닫혔으면, 이 호출만을 위해 임시 세션 생성
            logger.warning(
                "N10 _send_to_external_api: Provided http_session is closed or None. Creating a temporary one.",
                extra=extra_log_data)
            timeout_seconds = settings.EXTERNAL_API_TIMEOUT_SECONDS if hasattr(settings,
                                                                               'EXTERNAL_API_TIMEOUT_SECONDS') else 30
            timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
            session_to_use = aiohttp.ClientSession(timeout=timeout)
            internal_session_created_for_this_call = True

        if not session_to_use:  # 그래도 세션이 없다면 (거의 발생 안 함)
            logger.error("N10 _send_to_external_api: Failed to obtain a valid HTTP session for external API call.",
                         extra=extra_log_data)
            return {"status": "failed", "error": "HTTP session unavailable for external API"}

        try:
            async with session_to_use.post(api_url, headers=headers, json=payload) as response:
                response_text = await response.text()
                status_code = response.status
                log_response_summary = response_text[:500]
                if 200 <= status_code < 300:
                    logger.info(
                        f"Successfully sent data to external API (Status: {status_code}). Response: {log_response_summary}")
                    try:
                        return {"status": "success", "response_status": status_code,
                                "response_body": json.loads(response_text)}
                    except json.JSONDecodeError:
                        return {"status": "success", "response_status": status_code, "response_body": response_text}
                else:
                    logger.error(f"Failed to send data to external API (Status: {status_code}): {log_response_summary}",
                                 extra=extra_log_data)
                    return {"status": "failed", "response_status": status_code, "error": response_text}
        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection error sending data to external API '{api_url}': {e}", extra=extra_log_data)
            return {"status": "failed", "error": f"Connection error: {e}"}
        except asyncio.TimeoutError:  # aiohttp.ClientTimeout 관련 예외는 보통 asyncio.TimeoutError로 잡힘
            logger.error(f"Timeout sending data to external API: {api_url}", extra=extra_log_data)
            return {"status": "failed", "error": "Request timed out"}
        except Exception as e:
            logger.exception(f"Unexpected error sending data to external API: {e}", extra=extra_log_data)
            return {"status": "failed", "error": f"Unexpected error: {type(e).__name__} - {e}"}
        finally:
            # 이 함수 내에서 임시로 세션을 만들었다면 닫아줌
            if internal_session_created_for_this_call and session_to_use and not session_to_use.closed:
                await session_to_use.close()
                logger.info("N10 _send_to_external_api: Closed temporary aiohttp session created for this call.",
                            extra=extra_log_data)

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id = config.get('writer_id', 'default_writer')
        error_log = list(state.error_log or [])  # 이전 노드들의 오류 로그를 가져와 누적

        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id, 'node_name': node_name}
        logger.info(f"Entering node {node_name}. Finalizing workflow and notifying.", extra=extra)

        s3_upload_results: List[Dict[str, Optional[str]]] = []
        translated_report_content_final: Optional[str] = None
        referenced_urls_list_final: List[str] = []
        external_api_call_result_final: Optional[Dict[str, Any]] = None

        if not comic_id:
            error_msg = "Critical: comic_id is missing. Cannot perform finalization tasks."
            logger.error(error_msg, extra=extra)
            error_log.append(
                {"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            return {"current_stage": "ERROR", "error_log": error_log, "error_message": error_msg}

        try:
            # --- 1. 이미지 S3 업로드 (컷 + 썸네일) ---
            s3_base_prefix_for_images = settings.S3_IMAGE_PREFIX if hasattr(settings, 'S3_IMAGE_PREFIX') else "comics"

            upload_tasks = []
            if state.generated_comic_images and isinstance(state.generated_comic_images, list):
                for image_info_item in state.generated_comic_images:
                    if isinstance(image_info_item, dict):  # 각 이미지 정보가 딕셔너리인지 확인
                        upload_tasks.append(
                            self._upload_image_to_s3(
                                image_info=image_info_item,
                                comic_id=comic_id,
                                s3_base_prefix=s3_base_prefix_for_images,
                                trace_id=trace_id,  # type: ignore
                                extra_log_data=extra
                            )
                        )

            if upload_tasks:
                logger.info(
                    f"N10: Starting {len(upload_tasks)} S3 upload tasks (컷 및 썸네일 포함, max parallel: {MAX_PARALLEL_S3_UPLOADS})...",
                    extra=extra)
                semaphore = asyncio.Semaphore(MAX_PARALLEL_S3_UPLOADS)

                async def run_upload_with_semaphore(task_coro):
                    async with semaphore: return await task_coro

                s3_task_results = await asyncio.gather(*(run_upload_with_semaphore(task) for task in upload_tasks),
                                                       return_exceptions=True)

                for i, res_item in enumerate(s3_task_results):
                    # 원본 image_info를 참조하여 scene_identifier 등을 가져오기 위한 노력 (결과가 예외일 경우 대비)
                    original_image_info = state.generated_comic_images[i] if i < len(
                        state.generated_comic_images) else {}  # type: ignore
                    scene_id_from_source = original_image_info.get("scene_identifier", f"UnknownItem_task_{i}")
                    is_thumb_from_source = original_image_info.get("is_thumbnail", False)

                    if isinstance(res_item, Exception):
                        err_msg = f"N10: S3 upload task for '{scene_id_from_source}' (thumbnail: {is_thumb_from_source}) failed: {type(res_item).__name__} - {res_item}"
                        logger.error(err_msg, extra=extra, exc_info=res_item)  # 예외 정보 포함 로깅
                        error_log.append({"stage": f"{node_name}.s3_upload_exception", "scene": scene_id_from_source,
                                          "is_thumbnail": is_thumb_from_source, "error": err_msg,
                                          "timestamp": datetime.now(timezone.utc).isoformat()})
                        s3_upload_results.append({"scene_identifier": scene_id_from_source, "s3_url": None,
                                                  "is_thumbnail": is_thumb_from_source, "error": err_msg})
                    elif isinstance(res_item, dict):
                        s3_upload_results.append(
                            res_item)  # _upload_image_to_s3 반환값 (scene_identifier, is_thumbnail 포함)
                        if res_item.get("error"):
                            error_log.append(
                                {"stage": f"{node_name}.s3_upload_error", "scene": res_item.get("scene_identifier"),
                                 "is_thumbnail": res_item.get("is_thumbnail"), "error": res_item["error"],
                                 "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                logger.info("N10: No images found in state.generated_comic_images to upload to S3.", extra=extra)

            # --- 2. 보고서 번역 ---
            if state.report_content:
                target_lang = settings.REPORT_TRANSLATION_TARGET_LANGUAGE if hasattr(settings,
                                                                                     'REPORT_TRANSLATION_TARGET_LANGUAGE') else "en"
                translated_report_content_final = await self._translate_report(state.report_content, target_lang,
                                                                               trace_id, extra)  # type: ignore
                if not translated_report_content_final and state.report_content:  # 번역 시도했으나 결과가 없는 경우
                    error_log.append({"stage": f"{node_name}._translate_report",
                                      "error": "Report translation returned empty or failed (check service logs).",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                logger.info("N10: No report content found in state. Skipping translation.", extra=extra)

            # --- 3. 참조 링크 목록 생성 ---
            referenced_urls_list_final = self._extract_referenced_urls(state)

            # --- 4. 외부 API 전송 ---
            ext_api_url = settings.EXTERNAL_NOTIFICATION_API_URL if hasattr(settings,
                                                                            'EXTERNAL_NOTIFICATION_API_URL') else None
            ext_api_token = settings.EXTERNAL_NOTIFICATION_API_TOKEN if hasattr(settings,
                                                                                'EXTERNAL_NOTIFICATION_API_TOKEN') else None

            if ext_api_url:
                # 페이로드 구성 시 썸네일 URL 분리 또는 generatedImages 내 플래그 활용
                thumbnail_s3_info = next(
                    (img for img in s3_upload_results if img.get("is_thumbnail") and img.get("s3_url")), None)
                scene_s3_images = [img for img in s3_upload_results if not img.get("is_thumbnail")]

                current_overall_status = "completed"
                if error_log or state.error_message:  # 현재까지의 모든 오류 확인
                    current_overall_status = "completed_with_errors"

                api_payload = {
                    "comicId": comic_id, "traceId": trace_id, "status": current_overall_status,
                    "originalQuery": state.original_query, "writerId": writer_id,
                    "reportTranslatedHtml": translated_report_content_final,
                    "referencedUrls": referenced_urls_list_final,
                    "thumbnailImage": thumbnail_s3_info,  # 썸네일 정보 (URL, 오류 등 포함 가능)
                    "sceneImages": scene_s3_images,  # 컷 이미지 정보 리스트
                    # "generatedImages": s3_upload_results, # 또는 전체 리스트와 is_thumbnail 플래그 전달
                    "scenarioText": state.comic_scenarios[0].get("scenario_text") if state.comic_scenarios and
                                                                                     state.comic_scenarios[0] else None,
                    # type: ignore
                    "ideaTitle": state.selected_comic_idea_for_scenario.get(
                        "title") if state.selected_comic_idea_for_scenario else None,  # type: ignore
                    "finalTimestamp": datetime.now(timezone.utc).isoformat(),
                    "errorsOccurred": bool(error_log or state.error_message)
                }
                if state.error_message: api_payload["mainWorkflowError"] = state.error_message

                external_api_call_result_final = await self._send_to_external_api(
                    api_payload, ext_api_url, ext_api_token, trace_id, extra
                )
                if external_api_call_result_final.get("status") == "failed":
                    error_log.append({"stage": f"{node_name}._send_to_external_api",
                                      "error": f"External API notification failed: {external_api_call_result_final.get('error', 'Unknown reason')}",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                logger.info("N10: External notification API URL not configured. Skipping notification.", extra=extra)
                external_api_call_result_final = {"status": "skipped", "reason": "API URL not configured in settings"}

            # --- 최종 상태 업데이트 ---
            final_workflow_stage = "DONE"
            if error_log or state.error_message:
                final_workflow_stage = "DONE_WITH_ERRORS"

            update_dict = {
                "uploaded_image_urls": s3_upload_results,  # 썸네일 포함된 전체 결과
                "translated_report_content": translated_report_content_final,
                "referenced_urls": referenced_urls_list_final,
                "external_api_response": external_api_call_result_final,
                "current_stage": final_workflow_stage,
                "error_log": error_log
            }
            # 워크플로우 레벨의 주 오류 메시지가 전체 오류 로그에 아직 없다면 추가 (중복 방지)
            if state.error_message and not any(
                    entry.get("error") == state.error_message and entry.get("stage", "").lower().startswith("workflow")
                    for entry in error_log):
                error_log.append({"stage": "WorkflowTopLevelError", "error": state.error_message,
                                  "timestamp": datetime.now(timezone.utc).isoformat()})

            log_summary_update = {
                "final_workflow_stage": final_workflow_stage,
                "uploaded_image_count": len(s3_upload_results),  # 썸네일 포함
                "images_with_upload_errors": sum(1 for img in s3_upload_results if img.get("error")),
                "translation_successful": bool(translated_report_content_final),
                "external_api_status": external_api_call_result_final.get(
                    "status") if external_api_call_result_final else "skipped"
            }
            logger.info(
                f"Exiting node {node_name}. Finalization complete. Summary: {summarize_for_logging(log_summary_update)}",
                extra=extra)
            return update_dict

        except Exception as e:  # N10 노드 실행 중 예기치 않은 최상위 예외
            error_msg = f"N10: Unexpected critical error in {node_name} execution: {type(e).__name__} - {e}"
            logger.exception(error_msg, extra=extra)
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})

            # 외부 API 알림 시도 (실패 알림)
            ext_api_url_on_exc = settings.EXTERNAL_NOTIFICATION_API_URL if hasattr(settings,
                                                                                   'EXTERNAL_NOTIFICATION_API_URL') else None
            ext_api_token_on_exc = settings.EXTERNAL_NOTIFICATION_API_TOKEN if hasattr(settings,
                                                                                       'EXTERNAL_NOTIFICATION_API_TOKEN') else None
            if ext_api_url_on_exc and not external_api_call_result_final:  # 아직 알림 안 보냈고 URL 있으면
                logger.info(f"N10: Attempting failure notification due to critical error in {node_name}.", extra=extra)
                critical_error_payload = {
                    "comicId": comic_id, "traceId": trace_id, "status": "failed",
                    "error_message": error_msg,
                    "finalTimestamp": datetime.now(timezone.utc).isoformat(),
                    "errorsOccurred": True
                }
                # 이 알림 호출의 결과는 최종 결과에 반영하지 않거나, 별도 처리. 일단 호출만 시도.
                await self._send_to_external_api(critical_error_payload, ext_api_url_on_exc, ext_api_token_on_exc,
                                                 trace_id, extra)

            return {
                "uploaded_image_urls": s3_upload_results,
                "translated_report_content": translated_report_content_final,
                "referenced_urls": referenced_urls_list_final,
                "external_api_response": external_api_call_result_final or {
                    "status": "not_attempted_due_to_early_failure"},
                "error_log": error_log,
                "current_stage": "ERROR",
                "error_message": error_msg
            }
        finally:
            # N10 노드가 직접 생성한 aiohttp 세션인 경우에만 닫기
            if hasattr(self, '_external_api_session') and self._external_api_session and \
                    hasattr(self, '_created_external_api_session') and self._created_external_api_session and \
                    not self._external_api_session.closed:
                await self._external_api_session.close()
                logger.info(f"N10: Closed internally created aiohttp session for external API notifications.")


async def main_test_n10():
    """
    N10FinalizeAndNotifyNode를 테스트하기 위한 main 함수입니다. (썸네일 처리 및 Google 번역 반영)
    실제 Settings와 서비스를 사용하므로, 실행 환경에 관련 설정이 필요합니다.
    """
    print("--- N10FinalizeAndNotifyNode Test (Upgraded - Thumbnail & GoogleRestTranslation) ---")
    logger.info("N10 Test (Thumbnail, GoogleRest): 시작")

    # 전역 settings 객체 사용
    if not settings.S3_BUCKET_NAME or not settings.AWS_REGION:
        logger.error("N10 Test: S3_BUCKET_NAME 또는 AWS_REGION 설정이 없습니다.")
        print("[오류] S3 버킷/리전 설정이 필요합니다. .env 파일을 확인하세요.")
        return

    # GoogleRestTranslationService는 GOOGLE_API_KEY를 settings에서 사용
    if not settings.GOOGLE_API_KEY:
        logger.warning("N10 Test: GOOGLE_API_KEY가 설정에 없습니다. Google 번역이 실패할 수 있습니다.")
        # print("[경고] Google 번역(API 키 방식)을 위해서는 GOOGLE_API_KEY 설정이 .env 파일에 필요합니다.")

    storage_service = StorageService()
    translation_service = TranslationService()  # GoogleRestTranslationService 사용 (import에서 별칭)

    external_api_session = None
    if settings.EXTERNAL_NOTIFICATION_API_URL:  # 외부 알림 API URL이 설정된 경우에만 세션 생성
        timeout_seconds = settings.EXTERNAL_API_TIMEOUT_SECONDS if hasattr(settings,
                                                                           'EXTERNAL_API_TIMEOUT_SECONDS') else 30
        timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
        external_api_session = aiohttp.ClientSession(timeout=timeout)
        logger.info("N10 Test (Thumbnail, GoogleRest): main에서 EXTERNAL_NOTIFICATION_API_URL 용 aiohttp 세션 생성")

    node = N10FinalizeAndNotifyNode(
        storage_service=storage_service,
        translation_service=translation_service,
        http_session=external_api_session  # 생성한 세션 주입 또는 None (노드가 내부 생성)
    )

    trace_id = f"test-trace-n10-final-{uuid.uuid4().hex[:8]}"
    comic_id = f"test-comic-n10-final-{uuid.uuid4().hex[:8]}"

    # N09가 이미지를 settings.IMAGE_STORAGE_PATH 아래에 저장했다고 가정하고 더미 파일 생성
    # ImageService의 storage_base_path가 settings.IMAGE_STORAGE_PATH를 사용
    # 파일명은 ImageService가 자체적으로 생성하므로, 예측하기 어려움.
    # N10 테스트에서는 N09의 출력인 image_path를 직접 시뮬레이션.
    test_image_output_root = Path(
        settings.IMAGE_STORAGE_PATH) if settings.IMAGE_STORAGE_PATH else PROJECT_ROOT / "temp_n10_test_images"
    # comic_id 별로 디렉토리를 만들지 않고, IMAGE_STORAGE_PATH에 직접 저장하는 것으로 가정
    # (ImageService가 comic_id를 경로에 포함하지 않는다고 가정)
    test_image_output_root.mkdir(parents=True, exist_ok=True)

    generated_images_for_state = []

    # 더미 컷 이미지 1개 생성
    cut_image_filename = f"test_cut_{comic_id}_01.png"
    cut_image_path = test_image_output_root / cut_image_filename
    try:
        with open(cut_image_path, "w", encoding="utf-8") as f:
            f.write(f"Dummy cut image for {comic_id}")
        logger.info(f"N10 Test: Created dummy cut image: {cut_image_path}")
        generated_images_for_state.append({
            "scene_identifier": "FinalCut_01", "image_path": str(cut_image_path),
            "prompt_used": "A final test cut image prompt", "is_thumbnail": False, "error": None
        })
    except Exception as e:
        logger.error(f"N10 Test: Failed to create dummy cut image {cut_image_path}: {e}")
        generated_images_for_state.append(
            {"scene_identifier": "FinalCut_01", "is_thumbnail": False, "error": f"Dummy file creation failed: {e}"})

    # 더미 썸네일 이미지 1개 생성
    thumb_image_filename = f"test_thumbnail_{comic_id}.jpg"
    thumb_image_path = test_image_output_root / thumb_image_filename
    try:
        with open(thumb_image_path, "w", encoding="utf-8") as f:
            f.write(f"Dummy thumbnail for {comic_id}")
        logger.info(f"N10 Test: Created dummy thumbnail: {thumb_image_path}")
        generated_images_for_state.append({
            "scene_identifier": "thumbnail_final_01", "image_path": str(thumb_image_path),
            "prompt_used": "A final test thumbnail prompt", "is_thumbnail": True, "error": None
        })
    except Exception as e:
        logger.error(f"N10 Test: Failed to create dummy thumbnail {thumb_image_path}: {e}")
        generated_images_for_state.append({"scene_identifier": "thumbnail_final_01", "is_thumbnail": True,
                                           "error": f"Dummy file creation failed: {e}"})

    # URL만 있는 이미지 (N09에서 ImageService가 URL을 반환한 경우)
    generated_images_for_state.append({
        "scene_identifier": "FinalURLScene", "image_url": "https://example.com/final_image.png",
        "prompt_used": "Image from URL prompt", "is_thumbnail": False, "error": None
    })

    state = WorkflowState(
        trace_id=trace_id, comic_id=comic_id,
        original_query="N10 최종 통합 테스트 (썸네일, Google 번역)",
        config={"writer_id": "n10_integration_tester"},
        generated_comic_images=generated_images_for_state,
        report_content="<h1>N10 Integration Test Report</h1><p>This report will be translated by Google. Hello and welcome to the final test!</p>",
        raw_search_results=[{"url": "https://www.example.com/final_article"}],
        comic_scenarios=[{"scenario_text": "A brief scenario text..."}],
        selected_comic_idea_for_scenario={"title": "The Grand Finale Idea"},
        current_stage="N09_IMAGE_GENERATION_COMPLETED", error_log=[]
    )

    logger.info(
        f"N10 Test: WorkflowState prepared. Comic ID: {comic_id}, Total images for N10: {len(generated_images_for_state)}")

    result_update = None
    try:
        result_update = await node.run(state)
        logger.info(
            f"N10 Test: node.run() completed. Result summary: {summarize_for_logging(result_update, max_len=1000)}")

        print(f"\n[INFO] N10 Node Run Complete. Final Stage: {result_update.get('current_stage')}")
        print(f"  Uploaded Images ({len(result_update.get('uploaded_image_urls', []))} items):")
        for item in result_update.get('uploaded_image_urls', []):
            print(f"    - Scene/ID: {item.get('scene_identifier')}, Thumbnail: {item.get('is_thumbnail')}, "
                  f"URL: {item.get('s3_url')}, Error: {item.get('error')}")

        print(
            f"  Translated Report (first 100 chars): {summarize_for_logging(result_update.get('translated_report_content'), max_len=100)}")
        print(f"  Referenced URLs: {result_update.get('referenced_urls')}")
        print(f"  External API Response: {result_update.get('external_api_response')}")
        if result_update.get('error_log'):
            print(f"  Error Log ({len(result_update['error_log'])} entries):")  # type: ignore
            for err in result_update['error_log']:  # type: ignore
                print(f"    - Stage: {err.get('stage')}, Error: {err.get('error')}")

    except Exception as e:
        logger.error(f"N10 Test: Exception during node.run(): {e}", exc_info=True)
        print(f"[ERROR] Exception in N10 test main: {e}")
    finally:
        # 외부 API용 세션 닫기 (main에서 생성한 경우)
        if external_api_session and not external_api_session.closed:
            await external_api_session.close()
            logger.info("N10 Test: Closed aiohttp session for external API (if created by main).")

        # TranslationService가 자체 클라이언트를 관리하고 close 메서드가 있다면 호출
        if hasattr(translation_service, 'close') and asyncio.iscoroutinefunction(
                translation_service.close):  # type: ignore
            await translation_service.close()  # type: ignore
            logger.info("N10 Test: TranslationService resources closed.")

        # 테스트용으로 생성한 전체 이미지 디렉토리 정리
        if test_image_output_root.exists():
            try:
                shutil.rmtree(test_image_output_root)  # 해당 루트 디렉토리 전체 삭제
                logger.info(f"N10 Test: Cleaned up base test image directory: {test_image_output_root}")
            except Exception as e:
                logger.warning(f"N10 Test: Failed to clean up base test image directory {test_image_output_root}: {e}")

    logger.info("N10 Test: 완료")
    print("--- N10FinalizeAndNotifyNode Test (Upgraded - Thumbnail & GoogleRestTranslation) End ---")


if __name__ == "__main__":
    import logging

    # 로깅 레벨을 DEBUG로 설정하면 더 많은 정보 확인 가능 (예: HTTP 요청 상세)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] [%(node_name)s] %(message)s')
    # ContextFilter를 사용하려면 로깅 설정 함수 호출 필요
    # from app.utils.logger import setup_logging
    # setup_logging() # YAML 설정 파일 기반 로깅 (선택 사항)

    asyncio.run(main_test_n10())