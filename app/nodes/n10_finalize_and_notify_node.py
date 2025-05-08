# ai/app/nodes/n10_finalize_and_notify_node.py
import traceback
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
import asyncio
import aiohttp
import json
from pathlib import Path
import mimetypes
import os
import shutil  # main 테스트에서 디렉토리 삭제용
import uuid
import pprint

# --- 실제 애플리케이션 환경에 맞게 설정되어야 하는 임포트 ---
from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, summarize_for_logging, PROJECT_ROOT
from app.config.settings import Settings
from app.services.storage_service import StorageService
from app.services.google_translation_service import GoogleRestTranslationService as TranslationService  # API Key 방식

# --- 실제 애플리케이션 환경에 맞게 설정되어야 하는 임포트 끝 ---

logger = get_logger(__name__)
settings = Settings()

MAX_PARALLEL_S3_UPLOADS = settings.S3_MAX_PARALLEL_UPLOADS or 5


def extract_text_from_html(html_string: Optional[str]) -> str:
    # (이전 답변의 extract_text_from_html 함수 내용과 동일)
    if not html_string: return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_string, "html.parser")
        text = ' '.join(soup.stripped_strings)
        return text
    except ImportError:
        logger.warning("BeautifulSoup4 not installed. HTML content used as is for summary.")
        import re
        text = re.sub(r'<[^>]+>', ' ', html_string)
        text = ' '.join(text.split())
        return text
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {e}");
        return ""


class N10FinalizeAndNotifyNode:
    def __init__(
            self,
            storage_service: StorageService,
            translation_service: TranslationService,  # GoogleRestTranslationService 인스턴스
            http_session: Optional[aiohttp.ClientSession] = None  # 외부 알림 API용
    ):
        self.storage_service = storage_service
        self.translation_service = translation_service
        self._external_api_session = http_session
        self._created_external_api_session = False

        if self._external_api_session is None and settings.EXTERNAL_NOTIFICATION_API_URL:
            timeout_seconds = settings.EXTERNAL_API_TIMEOUT_SECONDS if hasattr(settings,
                                                                               'EXTERNAL_API_TIMEOUT_SECONDS') else 60
            timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
            self._external_api_session = aiohttp.ClientSession(timeout=timeout)
            self._created_external_api_session = True
            logger.info(f"N10: Created internal aiohttp session for EXTERNAL API (timeout: {timeout_seconds}s).")

    async def _upload_html_report_to_s3(  # (이전 답변과 동일)
            self, html_content: str, comic_id: str, s3_base_prefix: str,
            trace_id: str, extra_log_data: dict
    ) -> Optional[str]:
        if not html_content:
            logger.warning("N10: Report HTML content is empty, skipping S3 upload for report.", extra=extra_log_data)
            return None
        report_filename = "report.html"
        object_key_path = Path(s3_base_prefix.strip('/')) / comic_id / "report" / report_filename
        object_key_str = object_key_path.as_posix()
        temp_report_file_path: Optional[Path] = None
        try:
            temp_dir = PROJECT_ROOT / "temp_reports_for_upload"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_report_file_path = temp_dir / f"{comic_id}_{report_filename}"
            with open(temp_report_file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"N10: Uploading HTML report to S3: '{object_key_str}' from {temp_report_file_path}",
                        extra=extra_log_data)
            upload_result = await self.storage_service.upload_file(
                file_path=str(temp_report_file_path), object_key=object_key_str,
                content_type="text/html; charset=utf-8", acl=None)
            if not upload_result.get("error"):
                s3_uri = upload_result.get("s3_uri")
                logger.info(f"N10: HTML report successfully uploaded to S3: {s3_uri}", extra=extra_log_data)
                return s3_uri
            else:
                logger.error(f"N10: Failed to upload HTML report to S3: {upload_result.get('error')}",
                             extra=extra_log_data)
                return None
        except Exception as e:
            logger.exception(f"N10: Unexpected error during HTML report S3 upload: {e}", extra=extra_log_data)
            return None
        finally:
            if temp_report_file_path and temp_report_file_path.exists():
                try:
                    temp_report_file_path.unlink()
                except Exception as e_del:
                    logger.warning(f"N10: Failed to delete temp report file {temp_report_file_path}: {e_del}",
                                   extra=extra_log_data)

    async def _upload_image_to_s3(  # (이전 답변과 동일)
            self, image_info: Dict[str, Any], comic_id: str, s3_base_prefix: str,
            trace_id: str, extra_log_data: dict
    ) -> Dict[str, Optional[str]]:
        # (이전 답변의 _upload_image_to_s3 코드와 동일, is_thumbnail에 따라 경로 분기)
        is_thumbnail = image_info.get("is_thumbnail", False)
        scene_identifier = image_info.get("scene_identifier", "thumbnail" if is_thumbnail else "UnknownScene")
        result = {"scene_identifier": scene_identifier, "s3_url": None, "is_thumbnail": is_thumbnail,
                  "error": image_info.get("error")}
        if result["error"]:
            logger.warning(
                f"N10: Skipping S3 upload for {result['scene_identifier']} due to N09 error: {result['error']}",
                extra=extra_log_data)
            return result
        local_image_path_str = image_info.get("image_path")
        remote_image_url = image_info.get("image_url")
        if local_image_path_str:
            local_file = Path(local_image_path_str)
            if not local_file.is_file():
                result["error"] = f"Local image file not found: {local_image_path_str}";
                logger.warning(result["error"], extra=extra_log_data);
                return result
            filename = local_file.name
            object_key_path = Path(s3_base_prefix.strip('/')) / comic_id / (
                "thumbnail" if is_thumbnail else "images") / filename
            object_key_str = object_key_path.as_posix()
            content_type, _ = mimetypes.guess_type(local_image_path_str)
            content_type = content_type or (
                'image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png')
            try:
                upload_result = await self.storage_service.upload_file(str(local_file), object_key_str,
                                                                       content_type=content_type, acl=None)
                if not upload_result.get("error"):
                    result["s3_url"] = upload_result.get("s3_uri")
                    log_msg_prefix = "Thumbnail" if is_thumbnail else f"Scene '{scene_identifier}'"
                    logger.info(f"N10: {log_msg_prefix} image uploaded to S3: {result['s3_url']}", extra=extra_log_data)
                    if upload_result.get("warning"): logger.warning(
                        f"N10: S3 Upload for {log_msg_prefix} completed with warning: {upload_result['warning']}",
                        extra=extra_log_data)
                else:
                    result["error"] = f"S3 upload failed: {upload_result['error']}";
                    logger.warning(result["error"], extra=extra_log_data)
            except Exception as e:
                error_msg = f"N10: Unexpected S3 upload error for {local_image_path_str}: {type(e).__name__} - {e}"
                logger.exception(error_msg, extra=extra_log_data);
                result["error"] = error_msg
        elif remote_image_url:
            logger.info(
                f"N10: Using remote image URL for {'thumbnail' if is_thumbnail else scene_identifier}: {remote_image_url}",
                extra=extra_log_data)
            result["s3_url"] = remote_image_url
        else:
            result["error"] = "Image path/URL missing.";
            logger.warning(f"N10: {result['error']} for {'thumbnail' if is_thumbnail else scene_identifier}",
                           extra=extra_log_data)
        return result

    async def _translate_report(  # (이전 답변과 동일 - GoogleRestTranslationService 사용)
            self, report_html: str, target_lang: str, trace_id: str, extra_log_data: dict
    ) -> Optional[str]:
        # (이전 답변의 _translate_report 코드와 동일)
        if not self.translation_service or not self.translation_service.is_enabled:
            logger.warning("Translation service disabled, skipping report translation.", extra=extra_log_data)
            return None
        if not report_html:
            logger.warning("Report content is empty, skipping translation.", extra=extra_log_data)
            return None
        source_lang = "ko"
        logger.info(
            f"Translating report from '{source_lang}' to '{target_lang}' using {type(self.translation_service).__name__}...",
            extra=extra_log_data)
        try:
            translated_html = await self.translation_service.translate(text=report_html, source_lang=source_lang,
                                                                       target_lang=target_lang, trace_id=trace_id)
            if translated_html:
                logger.info(f"Report translated successfully to '{target_lang}'.", extra=extra_log_data)
            else:
                logger.warning("Report translation failed or returned empty.", extra=extra_log_data)
            return translated_html
        except Exception as e:
            logger.exception(f"Report translation failed after retries: {type(e).__name__} - {e}", extra=extra_log_data)
            return None

    def _extract_referenced_urls(self, state: WorkflowState) -> List[Dict[str, str]]:  # (이전 답변과 동일)
        # (이전 답변의 _extract_referenced_urls 코드와 동일 - headline, url 반환)
        source_news_list: List[Dict[str, str]] = []
        if state.raw_search_results and isinstance(state.raw_search_results, list):
            for i, item in enumerate(state.raw_search_results):
                if isinstance(item, dict) and item.get("url") and isinstance(item.get("url"), str):
                    url = item["url"]
                    if not url.startswith("file://"):
                        headline = item.get("title") or item.get("name") or f"Referenced Link {i + 1}"
                        try:
                            parsed_url = Path(url)
                            headline = parsed_url.name if parsed_url.name else headline
                        except Exception:
                            pass
                        source_news_list.append({"headline": headline, "url": url})
        logger.info(f"Extracted {len(source_news_list)} source news items for external API.")
        return source_news_list

    async def _send_to_external_api(  # (이전 답변과 동일 - api_token 파라미터 제거됨)
            self, payload: Dict[str, Any], api_url: str, trace_id: str, extra_log_data: dict
    ) -> Dict[str, Any]:
        # (이전 답변의 _send_to_external_api 코드와 동일, api_token 인자 없이 호출)
        if not api_url:
            logger.warning("External API URL not configured. Skipping.", extra=extra_log_data)
            return {"status": "skipped", "reason": "API URL not configured"}
        headers = {"Content-Type": "application/json"}
        logger.info(f"Sending data to external API: {api_url}", extra=extra_log_data)
        logger.debug(f"External API Payload: {summarize_for_logging(payload, max_len=1000)}", extra=extra_log_data)
        session_to_use = self._external_api_session
        internal_session_created = False
        if not session_to_use or session_to_use.closed:
            logger.warning("External API session not available or closed. Creating temporary session.",
                           extra=extra_log_data)
            timeout_s = settings.EXTERNAL_API_TIMEOUT_SECONDS if hasattr(settings,
                                                                         'EXTERNAL_API_TIMEOUT_SECONDS') else 60
            session_to_use = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=float(timeout_s)))
            internal_session_created = True
        if not session_to_use: return {"status": "failed", "error": "HTTP session unavailable"}
        try:
            async with session_to_use.post(api_url, headers=headers, json=payload) as response:
                resp_text = await response.text()
                status = response.status
                log_resp_summary = resp_text[:500]
                if 200 <= status < 300:
                    logger.info(f"External API success (Status: {status}). Resp: {log_resp_summary}")
                    try:
                        return {"status": "success", "response_status": status, "response_body": json.loads(resp_text)}
                    except json.JSONDecodeError:
                        return {"status": "success", "response_status": status, "response_body": resp_text}
                else:
                    logger.error(f"External API failed (Status: {status}): {log_resp_summary}", extra=extra_log_data)
                    return {"status": "failed", "response_status": status, "error": resp_text}
        except Exception as e:
            logger.exception(f"External API error: {e}", extra=extra_log_data)
            return {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        finally:
            if internal_session_created and session_to_use and not session_to_use.closed:
                await session_to_use.close()
                logger.info("Closed temporary session for external API call.", extra=extra_log_data)

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        # (이전 답변의 run 메소드와 거의 동일, 외부 API URL 및 페이로드 구성 부분은 명세에 맞춤)
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        config = state.config or {}
        writer_id_str = config.get('writer_id', '1')
        try:
            ai_author_id = int(writer_id_str)
        except ValueError:
            ai_author_id = 1; logger.warning(f"N10: writer_id '{writer_id_str}' invalid, defaulting to {ai_author_id}.")
        error_log = list(state.error_log or [])
        extra = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id_str, 'node_name': node_name}
        logger.info(f"Entering node {node_name}.", extra=extra)

        s3_uploads_final: List[Dict[str, Optional[Any]]] = []
        translated_report_final: Optional[str] = None
        source_news_final: List[Dict[str, str]] = []
        api_call_final_result: Optional[Dict[str, Any]] = None
        s3_report_uri: Optional[str] = None

        if not comic_id:
            # ... (이전과 동일한 comic_id 부재 시 오류 처리) ...
            error_msg = "Critical: comic_id is missing."
            logger.error(error_msg, extra=extra)
            error_log.append(
                {"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            return {"current_stage": "ERROR", "error_log": error_log, "error_message": error_msg}

        try:
            s3_prefix = settings.S3_WEBTOON_DATA_PREFIX if hasattr(settings,
                                                                   'S3_WEBTOON_DATA_PREFIX') else "webtoon_data"
            if state.report_content:
                s3_report_uri = await self._upload_html_report_to_s3(state.report_content, comic_id, s3_prefix,
                                                                     trace_id, extra)  # type: ignore
                if s3_report_uri:
                    source_news_final.append({"headline": f"AI Generated Report for {comic_id}", "url": s3_report_uri})
                else:
                    error_log.append({"stage": f"{node_name}._upload_html_report_to_s3",
                                      "error": "Failed to upload report.html to S3.",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})

            upload_img_tasks = []
            if state.generated_comic_images:
                for img_info in state.generated_comic_images:
                    if isinstance(img_info, dict):
                        upload_img_tasks.append(
                            self._upload_image_to_s3(img_info, comic_id, s3_prefix, trace_id, extra))  # type: ignore
            if upload_img_tasks:
                logger.info(
                    f"N10: Starting {len(upload_img_tasks)} S3 image uploads (max parallel: {MAX_PARALLEL_S3_UPLOADS}).",
                    extra=extra)
                semaphore = asyncio.Semaphore(MAX_PARALLEL_S3_UPLOADS)

                async def run_upload_task(task):
                    async with semaphore: return await task

                s3_img_results = await asyncio.gather(*(run_upload_task(task) for task in upload_img_tasks),
                                                      return_exceptions=True)
                for i, item_res in enumerate(s3_img_results):
                    orig_img_info = state.generated_comic_images[i] if i < len(
                        state.generated_comic_images) else {}  # type: ignore
                    scene_id = orig_img_info.get("scene_identifier", f"Task_{i}")
                    is_thumb = orig_img_info.get("is_thumbnail", False)
                    if isinstance(item_res, Exception):
                        err = f"S3 upload task for '{scene_id}' (thumb: {is_thumb}) failed: {item_res}"
                        logger.error(err, extra=extra, exc_info=item_res)
                        error_log.append(
                            {"stage": f"{node_name}.s3_img_upload_exc", "scene": scene_id, "is_thumbnail": is_thumb,
                             "error": err, "timestamp": datetime.now(timezone.utc).isoformat()})
                        s3_uploads_final.append(
                            {"scene_identifier": scene_id, "s3_url": None, "is_thumbnail": is_thumb, "error": err})
                    elif isinstance(item_res, dict):
                        s3_uploads_final.append(item_res)
                        if item_res.get("error"):
                            error_log.append(
                                {"stage": f"{node_name}.s3_img_upload_err", "scene": item_res.get("scene_identifier"),
                                 "is_thumbnail": item_res.get("is_thumbnail"), "error": item_res["error"],
                                 "timestamp": datetime.now(timezone.utc).isoformat()})

            if state.report_content:
                target_lang = settings.REPORT_TRANSLATION_TARGET_LANGUAGE if hasattr(settings,
                                                                                     'REPORT_TRANSLATION_TARGET_LANGUAGE') else "en"
                translated_report_final = await self._translate_report(state.report_content, target_lang, trace_id,
                                                                       extra)  # type: ignore
                if not translated_report_final and state.report_content:
                    error_log.append({"stage": f"{node_name}._translate_report", "error": "Report translation failed.",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})

            source_news_final.extend(self._extract_referenced_urls(state))

            # 외부 API 페이로드 구성 (명세에 따라)
            ext_api_url_final = "http://34.64.219.46:8080/api/v1/webtoons"  # 명세 URL
            if ext_api_url_final:  # 이 URL은 고정
                title_for_payload = state.selected_comic_idea_for_scenario.get(
                    "title") if state.selected_comic_idea_for_scenario else \
                    (state.original_query[:50] if state.original_query else f"Comic {comic_id}")

                report_summary_text = extract_text_from_html(translated_report_final or state.report_content)
                content_for_payload = (report_summary_text[:497] + "...") if len(report_summary_text or "") > 500 else (
                            report_summary_text or "")

                thumbnail_url_for_payload = next(
                    (img.get("s3_url") for img in s3_uploads_final if img.get("is_thumbnail") and img.get("s3_url")),
                    "")

                slides_for_payload = []
                scenarios_map_for_payload = {sc.get("scene_identifier"): sc for sc in state.comic_scenarios or [] if
                                             isinstance(sc, dict)}
                seq_counter = 1
                for img_item in s3_uploads_final:
                    if not img_item.get("is_thumbnail") and img_item.get("s3_url"):
                        scene_id_slide = img_item.get("scene_identifier")
                        scenario_data = scenarios_map_for_payload.get(scene_id_slide) if scene_id_slide else None
                        slide_content_text = scenario_data.get("dialogue") or scenario_data.get(
                            "scene_description") or f"Content for {scene_id_slide}" if scenario_data else f"Content for {scene_id_slide}"
                        slides_for_payload.append({
                            "slideSeq": seq_counter, "imageUrl": img_item["s3_url"],
                            "content": summarize_for_logging(slide_content_text, 200)  # 길이 제한
                        })
                        seq_counter += 1

                expected_slides_count = len(state.comic_scenarios or [])
                slides_count = len(slides_for_payload)
                if expected_slides_count > 0 and slides_count < expected_slides_count:
                    warn_msg = f"Only {slides_count} slides to send to external API (target: {expected_slides_count})."
                    logger.warning(warn_msg, extra=extra)
                    error_log.append({
                        "stage": f"{node_name}.slides_count",
                        "error": warn_msg,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    final_stage_status = "DONE_WITH_PARTIAL_ERRORS"
                elif expected_slides_count > 0 and slides_count == 0:
                    error_msg = "No slides generated for external API payload."
                    logger.error(error_msg, extra=extra)
                    error_log.append({
                        "stage": f"{node_name}.slides_count",
                        "error": error_msg,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    final_stage_status = "ERROR"

                api_payload_final = {
                    "aiAuthorId": ai_author_id, "category": "IT", "title": title_for_payload,
                    "content": content_for_payload, "thumbnailImageUrl": thumbnail_url_for_payload,
                    "slides": slides_for_payload, "sourceNews": source_news_final
                }
                api_call_final_result = await self._send_to_external_api(api_payload_final, ext_api_url_final, trace_id,
                                                                         extra)  # token=None
                if api_call_final_result.get("status") == "failed":
                    error_log.append({"stage": f"{node_name}._send_to_external_api",
                                      "error": f"External API call failed: {api_call_final_result.get('error', 'Unknown')}",
                                      "timestamp": datetime.now(timezone.utc).isoformat()})
            else:  # ext_api_url_final이 None인 경우는 없지만 (하드코딩)
                api_call_final_result = {"status": "skipped", "reason": "External API URL not configured"}

            final_stage_status = "DONE"
            if error_log or state.error_message: final_stage_status = "DONE_WITH_ERRORS"
            update_dict_final = {
                "uploaded_image_urls": s3_uploads_final, "translated_report_content": translated_report_final,
                "referenced_urls": [item['url'] for item in source_news_final],  # URL만 state에 저장
                "source_news_payload_for_api": source_news_final,  # API 전송용 페이로드 따로 저장
                "uploaded_report_s3_uri": s3_report_uri,
                "external_api_response": api_call_final_result, "current_stage": final_stage_status,
                "error_log": error_log}
            logger.info(
                f"Exiting node {node_name}. Status: {final_stage_status}. API Response: {api_call_final_result.get('status') if api_call_final_result else 'N/A'}",
                extra=extra)

            print(f"\n[INFO] N10 Node Run Complete. Final Stage: {final_stage_status}")
            print(f"  Uploaded Report S3 URI: {s3_report_uri}")
            print(f"  External API Response: {api_call_final_result.get('status') if api_call_final_result else 'N/A'}")

            # === 시나리오 결과만 간단하게 출력 ===
            print("\n================= [시나리오 결과 요약] =================")
            if not state.comic_scenarios:
                print("[경고] 생성된 시나리오가 없습니다.")
            else:
                for idx, sc in enumerate(state.comic_scenarios):
                    print(f"==== 시나리오 {idx+1} =========================================")
                    if sc.get('panel_number') is not None:
                        print(f"[패널 번호]    : {sc.get('panel_number')}")
                    print(f"[ID]         : {sc.get('scene_identifier')}")
                    print(f"[설명]        : {sc.get('scene_description')}")
                    if sc.get('dialogue') is not None:
                        print(f"[대사]        : {sc.get('dialogue')}")
                    if sc.get('sfx') is not None:
                        print(f"[효과음]      : {sc.get('sfx')}")
                    if sc.get('final_image_prompt') is not None:
                        print(f"[이미지 프롬프트] : {sc.get('final_image_prompt')}")
                    print("====================================================\n")

            if error_log:
                print(f"  Error Log ({len(error_log)} entries):")
                for err in error_log: print(f"    - Stage: {err.get('stage')}, Error: {err.get('error')}")

            return update_dict_final
        except Exception as e:
            # ... (이전과 동일한 최상위 예외 처리) ...
            error_msg = f"N10: Unexpected critical error: {type(e).__name__} - {e}"
            logger.exception(error_msg, extra=extra)
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            # 실패 알림은 API 명세에 없으므로 생략하거나 최소화
            return {"error_log": error_log, "current_stage": "ERROR", "error_message": error_msg}
        finally:
            if hasattr(self,
                       '_external_api_session') and self._external_api_session and self._created_external_api_session and not self._external_api_session.closed:
                await self._external_api_session.close()
                logger.info(f"N10: Closed internally created aiohttp session.")


async def main_test_n10():
    # (이전 답변의 main_test_n10 내용과 거의 동일하게 유지)
    # WorkflowState 구성 시 comic_scenarios에 scene_identifier와 dialogue/description 포함
    # EXTERNAL_NOTIFICATION_API_URL 설정은 이제 코드 내 URL 사용
    print("--- N10FinalizeAndNotifyNode Test (API Spec Applied) ---")
    logger.info("N10 Test (API Spec): 시작")

    if not settings.S3_BUCKET_NAME or not settings.AWS_REGION:  # type: ignore
        logger.error("N10 Test: S3_BUCKET_NAME 또는 AWS_REGION 설정이 없습니다.")
        return
    if not settings.GOOGLE_API_KEY:  # type: ignore
        logger.warning("N10 Test: GOOGLE_API_KEY가 없습니다. Google 번역이 실패할 수 있습니다.")

    storage_service = StorageService()
    translation_service = TranslationService()  # GoogleRestTranslationService

    # 외부 API URL이 코드에 명시되어 있으므로, settings.EXTERNAL_NOTIFICATION_API_URL 값은 여기선 사용 안함
    # main에서 세션 생성 및 주입
    external_api_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))

    node = N10FinalizeAndNotifyNode(
        storage_service=storage_service,
        translation_service=translation_service,
        http_session=external_api_session
    )

    trace_id = f"test-trace-n10-finalapi-{uuid.uuid4().hex[:8]}"
    comic_id = f"test-comic-n10-finalapi-{uuid.uuid4().hex[:8]}"

    test_image_output_root = Path(
        settings.IMAGE_STORAGE_PATH) if settings.IMAGE_STORAGE_PATH else PROJECT_ROOT / "temp_n10_images_finalapi"  # type: ignore
    test_image_output_root.mkdir(parents=True, exist_ok=True)

    generated_images_for_state = []
    # 컷 이미지 1개
    cut_image_path = test_image_output_root / f"cut_{comic_id}_01.png"
    with open(cut_image_path, "w", encoding="utf-8") as f:
        f.write("dummy cut")
    generated_images_for_state.append({
        "scene_identifier": "FinalCut_01_API", "image_path": str(cut_image_path),
        "prompt_used": "Prompt for cut 01", "is_thumbnail": False, "error": None})
    # 썸네일 이미지 1개
    thumb_image_path = test_image_output_root / f"thumb_{comic_id}_01.jpg"
    with open(thumb_image_path, "w", encoding="utf-8") as f:
        f.write("dummy thumb")
    generated_images_for_state.append({
        "scene_identifier": "FinalThumb_01_API", "image_path": str(thumb_image_path),
        "prompt_used": "Prompt for thumbnail 01", "is_thumbnail": True, "error": None})

    state = WorkflowState(
        trace_id=trace_id, comic_id=comic_id,
        original_query="N10 Final API Payload Test",
        config={"writer_id": "123"},  # 정수형 ID로 잘 변환되는지 확인
        comic_scenarios=[  # slides의 content 필드 채우기 위함
            {"scene_identifier": "FinalCut_01_API",
             "scene_description": "This is the detailed description for the first cut scene, which can be quite long.",
             "dialogue": "Slide 1 Dialogue!"},
            # 썸네일은 comic_scenarios에 직접 대응되지 않음
        ],
        selected_comic_idea_for_scenario={"title": "N10 Final API Test Title",
                                          "summary": "This is a concise summary of the webtoon, hopefully less than 500 characters to fit the API specification for content. It describes the main plot and theme."},
        generated_comic_images=generated_images_for_state,
        report_content="<h1>N10 Final API Test Report</h1><p>Full HTML report content. This will be uploaded to S3. The text part will be summarized for the 'content' field in the API payload. It includes various analysis and information generated throughout the workflow.</p>",
        raw_search_results=[
            {"url": "https://example.com/news/original-article-1", "title": "Original Article 1 Headline"},
            {"url": "https://example.org/blog/another-source"}  # 제목 없는 경우
        ],
        current_stage="N09_COMPLETED", error_log=[]
    )

    logger.info(f"N10 Test (Final API): WorkflowState prepared. Comic ID: {comic_id}")
    result_update = None
    try:
        result_update = await node.run(state)
        logger.info(
            f"N10 Test (Final API): node.run() completed. Result: {summarize_for_logging(result_update, max_len=1200)}")
        print(f"\n[INFO] N10 Node Run Complete. Final Stage: {result_update.get('current_stage')}")
        print(f"  Uploaded Report S3 URI: {result_update.get('uploaded_report_s3_uri')}")
        print(f"  External API Response: {result_update.get('external_api_response')}")
        if result_update.get('error_log'):
            print(f"  Error Log ({len(result_update['error_log'])} entries):")
            for err in result_update['error_log']: print(f"    - Stage: {err.get('stage')}, Error: {err.get('error')}")

    except Exception as e:
        logger.error(f"N10 Test (Final API): Exception during node.run(): {e}", exc_info=True)
    finally:
        if external_api_session and not external_api_session.closed:
            await external_api_session.close()
            logger.info("N10 Test (Final API): Closed aiohttp session.")
        if hasattr(translation_service, 'close') and asyncio.iscoroutinefunction(
                translation_service.close):  # type: ignore
            await translation_service.close()  # type: ignore
        if test_image_output_root.exists():
            try:
                shutil.rmtree(test_image_output_root)
            except Exception as e_del:
                logger.warning(f"N10 Test: Failed to clean test dir {test_image_output_root}: {e_del}")

    logger.info("N10 Test (Final API): 완료")
    print("--- N10FinalizeAndNotifyNode Test (Updated API Payload) End ---")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] [%(node_name)s] %(message)s')
    asyncio.run(main_test_n10())