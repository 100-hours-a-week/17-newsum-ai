# ai/app/nodes/n10_finalize_and_notify_node.py
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import asyncio
import aiohttp
import json
from pathlib import Path
import mimetypes

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging, PROJECT_ROOT
from app.config.settings import Settings
from app.services.storage_service import StorageService

# TranslationService는 N10에서 직접 보고서 번역에 사용하지 않으므로,
# 다른 텍스트 필드(예: 아이디어 요약 등) 번역이 필요할 경우에만 주입받도록 __init__에서 Optional 처리 가능.
# 현재는 다른 필드 번역 요구사항이 없으므로, __init__에서 주석 처리하거나 제거 가능.
# from app.services.google_translation_service import GoogleRestTranslationService as TranslationService

logger = get_logger(__name__)
settings = Settings()

MAX_PARALLEL_S3_UPLOADS = getattr(settings, 'S3_MAX_PARALLEL_UPLOADS', 5)
# N06에서 생성된 보고서의 언어 코드를 참조하기 위한 설정 (N10의 파일명 규칙 및 로그에 사용)
DEFAULT_ORIGINAL_REPORT_LANG_FROM_N06 = getattr(settings, 'ORIGINAL_REPORT_LANGUAGE', 'en')
DEFAULT_TRANSLATED_REPORT_LANG_FROM_N06 = getattr(settings, 'N06_REPORT_TRANSLATION_TARGET_LANG', 'ko')


def extract_text_from_html(html_string: Optional[str]) -> str:
    if not html_string: return ""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html_string, "html.parser")
        text = ' '.join(soup.stripped_strings)
        return text.strip()
    except ImportError:
        logger.warning(
            "N10: BeautifulSoup4 not installed. HTML content will be used as is for summary, which might include tags.")
        import re
        text = re.sub(r'<[^>]+>', ' ', html_string)
        text = ' '.join(text.split()).strip()
        return text
    except Exception as e:
        logger.error(f"N10: Error extracting text from HTML: {e}")
        return ""


class N10FinalizeAndNotifyNode:
    def __init__(
            self,
            storage_service: StorageService,
            http_session: Optional[aiohttp.ClientSession] = None
    ):
        self.storage_service = storage_service
        self._external_api_session = http_session
        self._created_external_api_session = False

        if self._external_api_session is None and settings.EXTERNAL_NOTIFICATION_API_URL:
            timeout_seconds = getattr(settings, 'EXTERNAL_API_TIMEOUT_SECONDS', 60)
            timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
            self._external_api_session = aiohttp.ClientSession(timeout=timeout)
            self._created_external_api_session = True
            logger.info(f"N10: Created internal aiohttp.ClientSession for External API (timeout: {timeout_seconds}s).")

    def s3_uri_to_https_url(self, s3_uri: Optional[str]) -> Optional[str]:
        if not s3_uri or not s3_uri.startswith("s3://"):
            return s3_uri
        try:
            bucket_and_path = s3_uri[5:]
            bucket, *path_parts = bucket_and_path.split("/")
            path = "/".join(path_parts)
            region = getattr(settings, "AWS_REGION", "ap-northeast-2")  # 설정에서 리전 가져오기
            # 일반적인 S3 가상 호스팅 스타일 URL
            return f"https://{bucket}.s3.{region}.amazonaws.com/{path}"
        except Exception as e:
            logger.error(f"N10: Failed to convert S3 URI '{s3_uri}' to HTTPS URL: {e}")
            return s3_uri  # 변환 실패 시 원본 반환

    async def _upload_local_file_to_s3(
            self, local_file_path_str: str, comic_id: str, s3_base_prefix: str,
            target_s3_filename: str, trace_id: str, extra_log_data: dict
    ) -> Optional[str]:
        """
        로컬 파일 시스템의 파일을 읽어 S3에 업로드하고 S3 URI를 반환합니다.
        주로 N06에서 저장한 보고서 파일을 S3에 올릴 때 사용됩니다.
        """
        local_file = Path(local_file_path_str)
        if not local_file.is_file():
            logger.warning(f"N10: Local file not found for S3 upload: {local_file_path_str}", extra=extra_log_data)
            return None

        try:
            # HTML 파일이라고 가정하고 content_type 고정 (필요시 파일 확장자로 추론)
            content_type = "text/html; charset=utf-8"

            object_key_path = Path(s3_base_prefix.strip('/')) / comic_id / "report" / target_s3_filename
            object_key_str = object_key_path.as_posix()

            logger.info(
                f"N10: Uploading local file '{local_file.name}' (as S3 object '{target_s3_filename}') to S3 key: '{object_key_str}'",
                extra=extra_log_data)
            upload_result = await self.storage_service.upload_file(
                file_path=str(local_file),  # 로컬 파일 경로 직접 전달
                object_key=object_key_str,
                content_type=content_type,
                acl=None  # ACL은 버킷 정책을 따르도록 None 또는 생략
            )
            if not upload_result.get("error"):
                s3_uri = upload_result.get("s3_uri")
                logger.info(
                    f"N10: File '{target_s3_filename}' successfully uploaded to S3 from '{local_file_path_str}'. S3 URI: {s3_uri}",
                    extra=extra_log_data)
                return s3_uri
            else:
                logger.error(
                    f"N10: Failed to upload '{target_s3_filename}' from '{local_file_path_str}' to S3: {upload_result.get('error')}",
                    extra=extra_log_data)
                return None
        except Exception as e:
            logger.exception(f"N10: Unexpected error during S3 upload of local file '{local_file_path_str}': {e}",
                             extra=extra_log_data)
            return None

    async def _upload_image_to_s3(
            self, image_info: Dict[str, Any], comic_id: str, s3_base_prefix: str,
            trace_id: str, extra_log_data: dict
    ) -> Dict[str, Optional[str]]:
        is_thumbnail = image_info.get("is_thumbnail", False)
        scene_identifier = image_info.get("scene_identifier", "thumbnail" if is_thumbnail else "UnknownScene")
        # 반환할 결과 딕셔너리 초기화
        result = {
            "scene_identifier": scene_identifier,
            "s3_url": None,  # HTTPS URL이 저장될 필드
            "is_thumbnail": is_thumbnail,
            "error": image_info.get("error")  # 이전 노드에서 발생한 오류 승계
        }

        if result["error"]:
            logger.warning(
                f"N10: Skipping S3 upload for image '{result['scene_identifier']}' due to prior error: {result['error']}",
                extra=extra_log_data)
            return result

        local_image_path_str = image_info.get("image_path")
        remote_image_url = image_info.get("image_url")  # DALL-E 등에서 직접 URL 반환 시

        if local_image_path_str:  # 로컬 파일 경로가 있는 경우
            local_file = Path(local_image_path_str)
            if not local_file.is_file():
                result["error"] = f"Local image file not found: {local_image_path_str}"
                logger.warning(result["error"], extra=extra_log_data)
                return result

            filename = local_file.name
            # S3 객체 키 경로 구성 (썸네일과 일반 이미지 경로 분리)
            s3_image_folder = "thumbnail" if is_thumbnail else "images"
            object_key_path = Path(s3_base_prefix.strip('/')) / comic_id / s3_image_folder / filename
            object_key_str = object_key_path.as_posix()

            # Content-Type 추론
            content_type, _ = mimetypes.guess_type(local_image_path_str)
            content_type = content_type or (
                'image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png')  # 기본값

            try:
                upload_s3_result = await self.storage_service.upload_file(
                    file_path=str(local_file),
                    object_key=object_key_str,
                    content_type=content_type,
                    acl=None
                )
                if not upload_s3_result.get("error"):
                    # S3 URI를 HTTPS URL로 변환하여 저장
                    result["s3_url"] = self.s3_uri_to_https_url(upload_s3_result.get("s3_uri"))
                    log_msg_prefix = "Thumbnail" if is_thumbnail else f"Scene image '{scene_identifier}'"
                    logger.info(f"N10: {log_msg_prefix} uploaded to S3: {result['s3_url']}", extra=extra_log_data)
                else:
                    result["error"] = f"S3 upload failed for '{filename}': {upload_s3_result['error']}"
                    logger.warning(result["error"], extra=extra_log_data)
            except Exception as e:
                error_msg = f"N10: Unexpected S3 upload error for local image '{local_image_path_str}': {type(e).__name__} - {e}"
                logger.exception(error_msg, extra=extra_log_data)
                result["error"] = error_msg
        elif remote_image_url:  # 원격 URL만 있는 경우 (예: DALL-E 직접 URL)
            # 이 URL이 이미 공개적으로 접근 가능한 HTTPS URL이라고 가정
            logger.info(
                f"N10: Using provided remote image URL for {'thumbnail' if is_thumbnail else scene_identifier}: {remote_image_url}",
                extra=extra_log_data)
            result["s3_url"] = remote_image_url
        else:  # 이미지 경로도, URL도 없는 경우
            result[
                "error"] = f"Image source (path or URL) missing for {'thumbnail' if is_thumbnail else scene_identifier}."
            logger.warning(result["error"], extra=extra_log_data)
        return result

    async def _send_to_external_api(
            self, payload: Dict[str, Any], api_url: str, trace_id: str, extra_log_data: dict
    ) -> Dict[str, Any]:
        # ... (이전 답변의 _send_to_external_api 구현과 동일)
        if not api_url:
            logger.warning("External API URL not configured. Skipping.", extra=extra_log_data)
            return {"status": "skipped", "reason": "API URL not configured",
                    "request_payload_summary": summarize_for_logging(payload)}

        headers = {"Content-Type": "application/json"}
        logger.info(f"N10: Sending data to external API: {api_url}", extra=extra_log_data)
        logger.debug(
            f"N10: External API Payload for comic_id '{extra_log_data.get('comic_id', 'N/A')}': {summarize_for_logging(payload, max_len=2000)}",
            extra=extra_log_data)

        session_to_use = self._external_api_session
        internal_session_created = False
        if not session_to_use or session_to_use.closed:
            logger.warning("N10: External API session not available or closed. Creating temporary session.",
                           extra=extra_log_data)
            timeout_s = getattr(settings, 'EXTERNAL_API_TIMEOUT_SECONDS', 60)
            session_to_use = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(timeout_s)))
            internal_session_created = True

        if not session_to_use:  # 세션 생성 실패 시
            return {"status": "failed", "error": "HTTP session unavailable",
                    "request_payload_summary": summarize_for_logging(payload)}

        try:
            async with session_to_use.post(api_url, headers=headers, json=payload) as response:
                resp_text = await response.text()
                status_code = response.status
                log_resp_summary = summarize_for_logging(resp_text, max_len=500)

                response_data = {"status": "failed", "response_status": status_code, "response_body": resp_text,
                                 "request_payload_summary": summarize_for_logging(payload)}
                if 200 <= status_code < 300:
                    response_data["status"] = "success"
                    logger.info(
                        f"N10: External API call successful (Status: {status_code}). Response: {log_resp_summary}",
                        extra=extra_log_data)
                    try:
                        response_data["response_body"] = json.loads(resp_text)
                    except json.JSONDecodeError:
                        pass  # 텍스트로 유지
                else:
                    logger.error(f"N10: External API call failed (Status: {status_code}): {log_resp_summary}",
                                 extra=extra_log_data)
                    response_data["error"] = resp_text  # 상세 오류는 error 필드에
                return response_data
        except Exception as e:
            logger.exception(f"N10: Error during external API call: {e}", extra=extra_log_data)
            return {"status": "failed", "error": f"{type(e).__name__}: {e}",
                    "request_payload_summary": summarize_for_logging(payload)}
        finally:
            if internal_session_created and session_to_use and not session_to_use.closed:
                await session_to_use.close()
                logger.info("N10: Closed temporary aiohttp.ClientSession for external API call.", extra=extra_log_data)

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        """
        업로드 결과, 외부 API 응답 등은 upload Section에, stage/error 등은 meta Section에 직접 할당합니다.
        반환값은 state_v2 구조에 맞게 {"upload": ..., "meta": ...} 형태로 반환합니다.
        """
        node_name = self.__class__.__name__
        meta = state.meta
        upload_sec = state.upload
        report_sec = state.report
        scenario_sec = state.scenario
        image_sec = state.image
        config_sec = state.config
        trace_id = meta.trace_id
        comic_id = meta.comic_id
        config = config_sec.config or {}
        error_log = list(meta.error_log or [])
        extra_log_fields = {'trace_id': trace_id, 'comic_id': comic_id, 'node_name': node_name}
        s3_uploaded_image_details: List[Dict[str, Optional[Any]]] = []
        s3_uploaded_original_report_https_url: Optional[str] = None
        s3_uploaded_translated_report_https_url: Optional[str] = None
        api_payload_source_news: List[Dict[str, str]] = []
        external_api_call_result: Optional[Dict[str, Any]] = None
        if not comic_id:
            error_msg = "N10 Critical: comic_id is missing."
            logger.error(error_msg, extra=extra_log_fields)
            error_log.append(
                {"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            meta.current_stage = "ERROR"
            meta.error_log = error_log
            meta.error_message = error_msg
            return {"upload": upload_sec.model_dump(), "meta": meta.model_dump()}
        writer_id_str = str(config.get('writer_id', '1'))
        extra_log_fields['writer_id'] = writer_id_str
        logger.info(f"Entering node {node_name}.", extra=extra_log_fields)
        try:
            s3_base_prefix = getattr(settings, 'S3_WEBTOON_DATA_PREFIX', "webtoon_data")
            # --- 1. 원본 보고서(en) S3 업로드 (N06 결과 활용) ---
            original_report_lang_code = str(
                config.get('ORIGINAL_REPORT_LANGUAGE', DEFAULT_ORIGINAL_REPORT_LANG_FROM_N06))
            if report_sec.saved_report_path:
                s3_target_orig_filename = f"report_{original_report_lang_code}.html"
                s3_uri_orig = await self._upload_local_file_to_s3(
                    local_file_path_str=report_sec.saved_report_path,
                    comic_id=comic_id,
                    s3_base_prefix=s3_base_prefix,
                    target_s3_filename=s3_target_orig_filename,
                    trace_id=trace_id,
                    extra_log_data=extra_log_fields
                )
                if s3_uri_orig:
                    s3_uploaded_original_report_https_url = self.s3_uri_to_https_url(s3_uri_orig)
                    api_payload_source_news.append({
                        "headline": f"AI Generated Report for {comic_id} (Original - {original_report_lang_code.upper()})",
                        "url": s3_uploaded_original_report_https_url})
                    logger.info(
                        f"N10: Original report ({original_report_lang_code}) uploaded to S3: {s3_uploaded_original_report_https_url}",
                        extra=extra_log_fields)
                else:
                    error_log.append({"stage": f"{node_name}.S3UploadOriginalReport",
                                      "error": f"Failed to upload original report ({original_report_lang_code}) from path '{report_sec.saved_report_path}' to S3.",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                logger.warning(
                    "N10: No path for original report (report_sec.saved_report_path) provided by N06. Original report not uploaded.",
                    extra=extra_log_fields)
            # --- 2. 번역된 보고서(ko) S3 업로드 (N06 결과 활용) ---
            translated_report_lang_code = str(
                config.get('N06_REPORT_TRANSLATION_TARGET_LANG', DEFAULT_TRANSLATED_REPORT_LANG_FROM_N06))
            if report_sec.translated_report_path:
                s3_target_trans_filename = f"report_{original_report_lang_code}_Translated_{translated_report_lang_code}.html"
                s3_uri_trans = await self._upload_local_file_to_s3(
                    local_file_path_str=report_sec.translated_report_path,
                    comic_id=comic_id,
                    s3_base_prefix=s3_base_prefix,
                    target_s3_filename=s3_target_trans_filename,
                    trace_id=trace_id,
                    extra_log_data=extra_log_fields
                )
                if s3_uri_trans:
                    s3_uploaded_translated_report_https_url = self.s3_uri_to_https_url(s3_uri_trans)
                    api_payload_source_news.append({
                        "headline": f"AI Generated Report for {comic_id} (Translated - {translated_report_lang_code.upper()})",
                        "url": s3_uploaded_translated_report_https_url})
                    logger.info(
                        f"N10: Translated report ({translated_report_lang_code}) uploaded to S3: {s3_uploaded_translated_report_https_url}",
                        extra=extra_log_fields)
                else:
                    error_log.append({"stage": f"{node_name}.S3UploadTranslatedReport",
                                      "error": f"Failed to upload translated report ({translated_report_lang_code}) from path '{report_sec.translated_report_path}' to S3.",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                logger.warning(
                    "N10: No path for translated report (report_sec.translated_report_path) provided by N06. Translated report not uploaded.",
                    extra=extra_log_fields)
            # --- 3. 이미지 S3 업로드 ---
            s3_upload_tasks = []
            if image_sec.generated_comic_images:
                for img_info_item in image_sec.generated_comic_images:
                    if isinstance(img_info_item, dict):
                        s3_upload_tasks.append(
                            self._upload_image_to_s3(img_info_item, comic_id, s3_base_prefix, trace_id,
                                                     extra_log_fields))
            if s3_upload_tasks:
                logger.info(f"N10: Starting {len(s3_upload_tasks)} S3 image uploads (max parallel: {MAX_PARALLEL_S3_UPLOADS}).", extra=extra_log_fields)
                semaphore = asyncio.Semaphore(MAX_PARALLEL_S3_UPLOADS)
                async def _run_task_with_sema(task, sema):
                    async with sema: return await task
                s3_image_upload_results = await asyncio.gather(*[_run_task_with_sema(task, semaphore) for task in s3_upload_tasks], return_exceptions=True)
                for i, img_res_item in enumerate(s3_image_upload_results):
                    orig_img_info_for_err = image_sec.generated_comic_images[i] if image_sec.generated_comic_images and i < len(image_sec.generated_comic_images) else {}
                    scene_id_for_err_log = orig_img_info_for_err.get("scene_identifier", f"ImageUploadTask_{i}")
                    is_thumb_for_err_log = orig_img_info_for_err.get("is_thumbnail", False)
                    if isinstance(img_res_item, Exception):
                        err_log_message = f"S3 image upload for '{scene_id_for_err_log}' (thumb: {is_thumb_for_err_log}) failed with exception: {img_res_item}"
                        logger.error(err_log_message, extra=extra_log_fields, exc_info=img_res_item)
                        error_log.append({"stage": f"{node_name}.S3ImageUploadException", "scene": scene_id_for_err_log,
                                          "is_thumbnail": is_thumb_for_err_log, "error": err_log_message,
                                          "timestamp": datetime.now(timezone.utc).isoformat()})
                        s3_uploaded_image_details.append({"scene_identifier": scene_id_for_err_log, "s3_url": None,
                                                          "is_thumbnail": is_thumb_for_err_log,
                                                          "error": err_log_message})
                    elif isinstance(img_res_item, dict):
                        s3_uploaded_image_details.append(img_res_item)
                        if img_res_item.get("error"):
                            error_log.append({"stage": f"{node_name}.S3ImageUploadError",
                                              "scene": img_res_item.get("scene_identifier"),
                                              "is_thumbnail": img_res_item.get("is_thumbnail"),
                                              "error": img_res_item["error"],
                                              "timestamp": datetime.now(timezone.utc).isoformat()})
                    else:
                        unexpected_upload_err_msg = f"S3 image upload for '{scene_id_for_err_log}' (thumb: {is_thumb_for_err_log}) returned unexpected result type: {type(img_res_item)}"
                        logger.error(unexpected_upload_err_msg, extra=extra_log_fields)
                        error_log.append(
                            {"stage": f"{node_name}.S3ImageUploadUnexpectedResult", "scene": scene_id_for_err_log,
                             "is_thumbnail": is_thumb_for_err_log, "error": unexpected_upload_err_msg,
                             "timestamp": datetime.now(timezone.utc).isoformat()})
                        s3_uploaded_image_details.append({"scene_identifier": scene_id_for_err_log, "s3_url": None,
                                                          "is_thumbnail": is_thumb_for_err_log,
                                                          "error": unexpected_upload_err_msg})
            # --- 4. 외부 API 페이로드 구성 ---
            logger.debug(
                f"N10: Final sourceNews for API payload (reports only): {summarize_for_logging(api_payload_source_news)}",
                extra=extra_log_fields)
            external_api_url = settings.EXTERNAL_NOTIFICATION_API_URL
            if external_api_url:
                payload_api_title = scenario_sec.selected_comic_idea_for_scenario.get(
                    "title") if scenario_sec.selected_comic_idea_for_scenario else \
                    (state.original_query[:50] if state.original_query else f"Comic {comic_id}")
                payload_api_content = ""
                original_report_html_from_n06_path = ""
                if report_sec.saved_report_path:
                    try:
                        with open(Path(report_sec.saved_report_path), "r", encoding="utf-8") as f_orig_rpt:
                            original_report_html_from_n06_path = f_orig_rpt.read()
                    except Exception as e_read_rpt:
                        logger.warning(
                            f"N10: Could not read original report from {report_sec.saved_report_path} for API content summary: {e_read_rpt}",
                            extra=extra_log_fields)
                if original_report_html_from_n06_path:
                    payload_api_content = summarize_for_logging(
                        extract_text_from_html(original_report_html_from_n06_path), 200)
                elif scenario_sec.selected_comic_idea_for_scenario:
                    idea_for_content = scenario_sec.selected_comic_idea_for_scenario
                    content_source_idea = idea_for_content.get("summary") or idea_for_content.get(
                        "logline") or idea_for_content.get("title")
                    payload_api_content = summarize_for_logging(content_source_idea, 200) if content_source_idea else ""
                payload_api_thumbnail_url = next((img.get("s3_url") for img in s3_uploaded_image_details if
                                                  img.get("is_thumbnail") and img.get("s3_url")), "")
                payload_api_slides = []
                scenario_details_map = {sc.get("scene_identifier"): sc for sc in scenario_sec.comic_scenarios or [] if
                                        isinstance(sc, dict)}
                current_slide_seq = 1
                for img_detail in s3_uploaded_image_details:
                    if not img_detail.get("is_thumbnail") and img_detail.get("s3_url"):
                        current_scene_id = img_detail.get("scene_identifier")
                        current_scenario = scenario_details_map.get(current_scene_id) if current_scene_id else None
                        current_slide_text = "-"
                        if current_scenario:
                            current_slide_text = current_scenario.get("dialogue") or current_scenario.get(
                                "scene_description") or "-"
                        if not current_slide_text or str(current_slide_text).strip().lower() in ["none",
                                                                                                 "none."]: current_slide_text = "-"
                        payload_api_slides.append({
                            "slideSeq": current_slide_seq,
                            "imageUrl": img_detail["s3_url"],
                            "content": summarize_for_logging(current_slide_text, 200)
                        })
                        current_slide_seq += 1
                try:
                    final_api_ai_author_id = int(writer_id_str)
                except ValueError:
                    final_api_ai_author_id = 1
                final_api_payload_to_send = {
                    "aiAuthorId": final_api_ai_author_id,
                    "category": str(config.get("category", "IT")),
                    "title": payload_api_title,
                    "content": payload_api_content,
                    "thumbnailImageUrl": payload_api_thumbnail_url,
                    "slides": payload_api_slides,
                    "sourceNews": api_payload_source_news
                }
                external_api_call_result = await self._send_to_external_api(final_api_payload_to_send, external_api_url,
                                                                            trace_id, extra_log_fields)
                if external_api_call_result.get("status") == "failed":
                    error_log.append({"stage": f"{node_name}.SendToExternalAPI",
                                      "error": f"External API call failed: {external_api_call_result.get('error', 'Unknown')}",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                external_api_call_result = {"status": "skipped", "reason": "External API URL not configured"}
            workflow_final_status = "DONE"
            if error_log or getattr(state, 'error_message', None):
                workflow_final_status = "DONE_WITH_ERRORS"
                if external_api_call_result and external_api_call_result.get("status") == "failed" and \
                        not getattr(settings, "IGNORE_EXTERNAL_API_FAILURE_FOR_WORKFLOW_STATUS", False):
                    logger.error(
                        "N10: External API call failed. Workflow marked as DONE_WITH_ERRORS (or ERROR based on policy).",
                        extra=extra_log_fields)
            upload_sec.uploaded_image_urls = s3_uploaded_image_details
            upload_sec.uploaded_report_s3_uri = s3_uploaded_original_report_https_url
            upload_sec.uploaded_translated_report_s3_uri = s3_uploaded_translated_report_https_url
            upload_sec.external_api_response = external_api_call_result
            meta.current_stage = workflow_final_status
            meta.error_log = error_log
            logger.info(
                f"Exiting node {node_name}. Status: {workflow_final_status}. External API status: {external_api_call_result.get('status') if external_api_call_result else 'N/A'}",
                extra=extra_log_fields)
            return {"upload": upload_sec.model_dump(), "meta": meta.model_dump()}
        except Exception as e_run_main_exc:
            critical_error_msg_in_run = f"N10: Unexpected critical error in run: {type(e_run_main_exc).__name__} - {e_run_main_exc}"
            logger.exception(critical_error_msg_in_run, extra=extra_log_fields)
            error_log.append({"stage": node_name, "error": critical_error_msg_in_run, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            meta.current_stage = "ERROR"
            meta.error_log = error_log
            meta.error_message = critical_error_msg_in_run
            return {"upload": upload_sec.model_dump(), "meta": meta.model_dump()}
        finally:
            if hasattr(self, '_external_api_session') and self._external_api_session and \
                    self._created_external_api_session and not self._external_api_session.closed:
                await self._external_api_session.close()
                log_extra = extra_log_fields if 'extra_log_fields' in locals() and isinstance(extra_log_fields,
                                                                                              dict) else None
                logger.info(f"N10: Closed internally created aiohttp.ClientSession.", extra=log_extra)


async def main_test_n10():
    # (이전 답변의 main_test_n10과 거의 동일, WorkflowState 구성 시 N06 결과 시뮬레이션 부분만 확인/강조)
    # ORIGINAL_REPORT_LANGUAGE 와 N06_REPORT_TRANSLATION_TARGET_LANG 설정을 config에 추가하여 테스트
    print("--- N10FinalizeAndNotifyNode Test (N06 Input, No N10 Translation, Corrected Lang Direction) ---")
    logger.info("N10 Test: 시작")
    import uuid
    # 필수 설정 확인 (getattr 사용으로 유연성 확보)
    if not getattr(settings, 'S3_BUCKET_NAME', None) or not getattr(settings, 'AWS_REGION', None):
        logger.error("N10 Test: S3_BUCKET_NAME 또는 AWS_REGION 설정이 없습니다. 테스트를 중단합니다.")
        return
    # GOOGLE_API_KEY는 N10에서 직접 보고서 번역에 사용하지 않으므로, 없어도 N10 자체는 동작 가능 (경고 불필요)
    if not getattr(settings, 'EXTERNAL_NOTIFICATION_API_URL', None):
        logger.warning("N10 Test: EXTERNAL_NOTIFICATION_API_URL is not set. External notification will be skipped.")
        # settings.EXTERNAL_NOTIFICATION_API_URL = "https://httpbin.org/post" # 테스트용 Mock URL

    storage_service = StorageService()
    # N10은 보고서 번역을 직접 수행하지 않으므로 translation_service 인스턴스는 None으로 전달 가능
    # 만약 다른 텍스트 필드 번역이 필요하다면, 해당 서비스 인스턴스를 생성하여 전달
    node = N10FinalizeAndNotifyNode(
        storage_service=storage_service,
        # translation_service=None, # 보고서 번역 안 하므로
    )

    trace_id = f"test-trace-n10-{uuid.uuid4().hex[:6]}"
    comic_id = f"test-comic-n10-{uuid.uuid4().hex[:6]}"

    # 테스트용 임시 파일 및 디렉토리 설정 (상대 경로 사용 권장)
    test_base_temp_dir = PROJECT_ROOT / "temp_n10_test_run_data"  # 모든 테스트 파일의 루트
    results_base_dir_test = test_base_temp_dir / comic_id / "reports"
    results_base_dir_test.mkdir(parents=True, exist_ok=True)
    img_output_root_test = test_base_temp_dir / comic_id / "images"
    img_output_root_test.mkdir(parents=True, exist_ok=True)

    # N06 결과 시뮬레이션 (원본: en, 번역본: ko)
    original_report_lang_code_for_test = "en"
    translated_report_lang_code_for_test = "ko"

    original_report_content_for_test_en = f"<html><body><h1>Original Report {comic_id} (EN)</h1><p>This is the original test report in English. It contains details about the webtoon generation process.</p></body></html>"
    # N06 저장 규칙에 따른 원본 파일명 (언어코드 포함)
    saved_report_filename_n06_en = f"report_{original_report_lang_code_for_test}.html"
    saved_report_path_test_en = results_base_dir_test / saved_report_filename_n06_en
    with open(saved_report_path_test_en, "w", encoding="utf-8") as f:
        f.write(original_report_content_for_test_en)

    translated_report_content_for_test_ko = f"<html><body><h1>번역된 보고서 {comic_id} (KO)</h1><p>이것은 한국어로 번역된 테스트 보고서입니다. 웹툰 생성 과정에 대한 상세 분석을 포함합니다.</p></body></html>"
    # N06 저장 규칙에 따른 번역본 파일명
    translated_report_filename_n06_ko = f"report_{original_report_lang_code_for_test}_Translated_{translated_report_lang_code_for_test}.html"
    translated_report_path_test_ko = results_base_dir_test / translated_report_filename_n06_ko
    with open(translated_report_path_test_ko, "w", encoding="utf-8") as f:
        f.write(translated_report_content_for_test_ko)

    # N09 이미지 결과 시뮬레이션
    generated_images_for_state = [
        {"scene_identifier": "Scene_EN_01", "image_path": str(img_output_root_test / "scene_en_01.png"),
         "prompt_used": "English prompt for scene 1", "is_thumbnail": False, "error": None},
        {"scene_identifier": "Thumbnail_EN", "image_path": str(img_output_root_test / "thumbnail_en.jpg"),
         "prompt_used": "English prompt for thumbnail", "is_thumbnail": True, "error": None}
    ]
    for img_data_item in generated_images_for_state:  # dummy 파일 생성
        with open(Path(img_data_item["image_path"]), "w", encoding="utf-8") as f_img: f_img.write(
            f"dummy image data for {img_data_item['scene_identifier']}")

    initial_state = WorkflowState(
        trace_id=trace_id, comic_id=comic_id,
        original_query="N10 Test with EN original / KO translated report from N06",
        config={  # N10이 참조할 수 있는 config 값들
            "writer_id": "2",
            "ORIGINAL_REPORT_LANGUAGE": original_report_lang_code_for_test,  # "en"
            "N06_REPORT_TRANSLATION_TARGET_LANG": translated_report_lang_code_for_test,  # "ko"
            # "N10_PERFORM_TRANSLATION_IF_NEEDED": False, # N10 자체 번역 안 하므로 이 설정은 영향 없음
            "category": "스릴러"
        },
        report_content=original_report_content_for_test_en,  # N05 결과 (영문 원본 HTML 내용)
        saved_report_path=str(saved_report_path_test_en),  # N06 결과 (영문 원본 로컬 경로)
        translated_report_path=str(translated_report_path_test_ko),  # N06 결과 (한글 번역본 로컬 경로)

        selected_comic_idea_for_scenario={"title": "Night Agent Kim (EN Idea)",
                                          "summary": "An English summary for the webtoon idea, used for API content field if report is unavailable."},
        comic_scenarios=[
            {"scene_identifier": "Scene_EN_01", "dialogue": "Target acquired. (EN)",
             "scene_description": "Agent Kim in a dark alley. (EN)"}
        ],
        generated_comic_images=generated_images_for_state,
        # referenced_urls_for_report는 N10의 sourceNews API 페이로드에 사용되지 않음
        referenced_urls_for_report=[
            {"title": "Some external article (EN)", "url": "http://external.example.com/article_en"}],
        current_stage="N09_IMAGES_GENERATED",
        error_log=[]  # 초기 에러 로그는 비어있음
    )

    final_state_update = None
    try:
        logger.info(f"N10 Test: Initializing and running N10FinalizeAndNotifyNode for comic_id={comic_id}")
        final_state_update = await node.run(initial_state)

        if final_state_update:
            print("\n--- N10 Test Run Final State Update Summary ---")
            # 순환 참조나 너무 큰 객체는 요약해서 출력하거나 제외
            import pprint
            summary_to_print = {
                k: (summarize_for_logging(v, 250) if isinstance(v, (
                str, list, dict)) and k != "external_api_response" else v)
                for k, v in final_state_update.items()
            }
            pprint.pprint(summary_to_print)

            if 'external_api_response' in final_state_update and final_state_update['external_api_response']:
                ext_api_resp = final_state_update['external_api_response']
                print(f"\nExternal API Response Status: {ext_api_resp.get('status')}")
                if ext_api_resp.get('status') == 'success':
                    print(f"  Response Status Code: {ext_api_resp.get('response_status')}")
                    # print(f"  Response Body Summary: {summarize_for_logging(ext_api_resp.get('response_body'), 200)}")
                else:
                    print(f"  Error: {summarize_for_logging(ext_api_resp.get('error'), 200)}")
                # print(f"  Request Payload Summary: {ext_api_resp.get('request_payload_summary')}")

        if final_state_update and final_state_update.get('error_log'):
            print(f"\nError Log ({len(final_state_update['error_log'])} entries):")
            for err_idx, err_item in enumerate(final_state_update['error_log']):
                print(f"  Error {err_idx + 1}: Stage='{err_item.get('stage')}', Message='{err_item.get('error')}'")

    except Exception as e_main_test_run:  # 변수명 변경
        logger.error(f"N10 Test: Exception during node.run() in main_test: {e_main_test_run}", exc_info=True)
    finally:
        # TranslationService 인스턴스는 N10 노드에 주입되지 않았으므로 닫을 필요 없음 (만약 주입했다면 닫아야 함)
        # if translation_service_for_n10 and hasattr(translation_service_for_n10, 'close'):
        #     if asyncio.iscoroutinefunction(translation_service_for_n10.close): await translation_service_for_n10.close()
        #     logger.info("N10 Test: Closed TranslationService (if any).")

        # N10 노드가 내부적으로 생성한 http_session은 노드의 finally 블록에서 이미 처리됨.

        # 테스트용 임시 파일/디렉토리 정리 (test_base_temp_dir 전체 삭제)
        if test_base_temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(test_base_temp_dir)
                logger.info(f"N10 Test: Cleaned up base test directory: {test_base_temp_dir}")
            except Exception as e_del_dir:
                logger.warning(f"N10 Test: Failed to clean base test directory {test_base_temp_dir}: {e_del_dir}")

    logger.info("N10 Test: 완료")
    print("--- N10FinalizeAndNotifyNode Test (N06 Input, No N10 Translation, Corrected Lang Direction) End ---")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
    # 로거 레벨 조정으로 불필요한 상세 로그 줄이기
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx의 INFO 레벨 로그가 많을 수 있음
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    asyncio.run(main_test_n10())