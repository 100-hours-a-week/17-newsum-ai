# ai/app/nodes/n06_save_report_node.py
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import html  # HTML 엔티티 처리를 위해 추가 (번역 서비스가 이미 처리한다면 불필요할 수 있음)

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, PROJECT_ROOT
from app.config.settings import Settings  # Settings 사용 가정
# 번역 서비스 임포트 (경로는 실제 프로젝트 구조에 맞게 조정)
from app.services.google_translation_service import GoogleRestTranslationService as TranslationService

logger = get_logger(__name__)
settings = Settings()  # Settings 인스턴스화

DEFAULT_RESULTS_BASE_DIR = PROJECT_ROOT / "results"
DEFAULT_TRANSLATION_TARGET_LANG = getattr(settings, 'REPORT_TRANSLATION_TARGET_LANGUAGE', 'en')


class N06SaveReportNode:
    """
    N05에서 생성된 보고서 내용을 로컬 파일 시스템에 저장하고,
    저장된 보고서를 번역하여 병행 저장하는 노드.
    """

    def __init__(self,
                 results_base_dir: Optional[Path] = None,
                 translation_service: Optional[TranslationService] = None):  # TranslationService 주입
        """
        노드 초기화.
        Args:
            results_base_dir (Optional[Path]): 보고서를 저장할 기본 디렉토리 경로.
                                              None이면 기본값(PROJECT_ROOT/results) 사용.
            translation_service (Optional[TranslationService]): 번역 서비스 인스턴스.
                                                              None이면 내부적으로 생성 시도.
        """
        self.results_base_dir = results_base_dir or DEFAULT_RESULTS_BASE_DIR
        if translation_service:
            self.translation_service = translation_service
        else:
            # 설정에서 GOOGLE_API_KEY 로드 확인 필요
            if settings.GOOGLE_API_KEY:
                self.translation_service = TranslationService()
                logger.info("N06SaveReportNode: Internally initialized TranslationService.")
            else:
                self.translation_service = None
                logger.warning("N06SaveReportNode: GOOGLE_API_KEY not found. Translation service disabled.")

        logger.info(f"Report save directory base set to: {self.results_base_dir}")

    async def _translate_and_save_report(
            self,
            original_report_path: Path,
            report_content: str,
            target_lang: str,
            trace_id: str,
            extra_log_data: dict
    ) -> Optional[str]:
        """
        보고서 내용을 번역하고 별도 파일로 저장합니다.
        """
        if not self.translation_service or not self.translation_service.is_enabled:
            logger.warning("Translation service is disabled in N06. Skipping report translation.", extra=extra_log_data)
            return None

        if not report_content:
            logger.warning("Report content is empty for translation in N06.", extra=extra_log_data)
            return None

        logger.info(f"Attempting to translate report to '{target_lang}'.", extra=extra_log_data)
        try:
            # HTML 번역 시 text_format='html', unescape_result=False (브라우저가 처리)
            translated_html = await self.translation_service.translate(
                text=report_content,
                target_lang=target_lang,
                source_lang="en",  # 원본 보고서가 한국어라고 가정
                text_format='html',
                unescape_result=False,
                trace_id=trace_id
            )

            if not translated_html:
                logger.warning(f"Report translation to '{target_lang}' returned empty or failed.", extra=extra_log_data)
                return None

            # 번역된 파일 경로 생성 (예: report_Translated_en.html)
            translated_file_name = f"{original_report_path.stem}_Translated_{target_lang}{original_report_path.suffix}"
            translated_report_file_path = original_report_path.parent / translated_file_name

            with open(translated_report_file_path, "w", encoding="utf-8") as f:
                f.write(translated_html)

            translated_path_str = str(translated_report_file_path.resolve())
            logger.info(f"Translated report successfully saved to: {translated_path_str}", extra=extra_log_data)
            return translated_path_str

        except Exception as e:
            logger.exception(f"Error during report translation or saving translated report in N06: {e}",
                             extra=extra_log_data)
            return None

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        """보고서 내용을 파일로 저장하고, 번역본도 저장한 후 상태 업데이트."""
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        report_content = state.report_content
        error_log = list(state.error_log or [])

        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'node_name': node_name}
        logger.info(f"Entering node. Attempting to save report content.", extra=extra_log_data)

        saved_path_str: Optional[str] = None
        translated_report_path_str: Optional[str] = None  # 번역된 파일 경로

        if not comic_id:
            error_msg = "Comic ID is missing, cannot determine save path."
            logger.error(error_msg, extra=extra_log_data)
            error_log.append(
                {"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            return {"error_log": error_log, "current_stage": "ERROR", "error_message": error_msg}

        if not report_content or not isinstance(report_content, str):
            error_msg = "Report content is missing or not a string. Nothing to save."
            logger.warning(error_msg, extra=extra_log_data)
            update_dict = {
                "saved_report_path": None,
                "translated_report_path": None,  # 번역 경로도 없음
                "current_stage": "DONE",
                "error_log": error_log
            }
            logger.info("Exiting node. No report content to save.", extra=extra_log_data)
            return update_dict

        try:
            report_dir = self.results_base_dir / comic_id
            report_dir.mkdir(parents=True, exist_ok=True)
            original_report_file_path = report_dir / "report.html"

            with open(original_report_file_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            saved_path_str = str(original_report_file_path.resolve())
            logger.info(f"Original report successfully saved to: {saved_path_str}", extra=extra_log_data)

            # 번역 및 저장 로직 호출
            # settings에서 번역 대상 언어 가져오기 (없으면 기본값 'en' 사용)
            target_translation_lang = getattr(settings, 'N06_REPORT_TRANSLATION_TARGET_LANG',
                                              DEFAULT_TRANSLATION_TARGET_LANG)

            # 번역 서비스가 활성화 되어 있고, 번역 대상 언어가 설정되어 있을 경우 번역 시도
            if self.translation_service and self.translation_service.is_enabled and target_translation_lang:
                translated_report_path_str = await self._translate_and_save_report(
                    original_report_path=original_report_file_path,
                    report_content=report_content,
                    target_lang=target_translation_lang,  # 예: "en" 또는 settings에서 가져옴
                    trace_id=trace_id,
                    extra_log_data=extra_log_data
                )
                if translated_report_path_str:
                    logger.info(f"Translated report path: {translated_report_path_str}", extra=extra_log_data)
                else:  # 번역 실패 또는 내용 없음
                    error_log.append({
                        "stage": f"{node_name}._translate_and_save_report",
                        "error": f"Failed to translate or save translated report for lang '{target_translation_lang}'.",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            elif not (self.translation_service and self.translation_service.is_enabled):
                logger.info("Translation service not enabled in N06. Skipping translation of saved report.",
                            extra=extra_log_data)
            elif not target_translation_lang:
                logger.info(
                    f"N06_REPORT_TRANSLATION_TARGET_LANG not set in settings. Skipping translation of saved report.",
                    extra=extra_log_data)

            update_dict = {
                "saved_report_path": saved_path_str,
                "translated_report_path": translated_report_path_str,  # 상태에 추가
                "current_stage": "DONE",
                "error_log": error_log
            }
            logger.info(f"Exiting node. Original report saved. Translated report path: {translated_report_path_str}",
                        extra=extra_log_data)
            return update_dict

        except OSError as e:
            error_msg = f"Failed to save report to file system: {e}"
            logger.exception(error_msg, extra=extra_log_data)
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            return {
                "saved_report_path": None,
                "translated_report_path": None,
                "current_stage": "ERROR",
                "error_message": f"N06 File Save Error: {error_msg}",
                "error_log": error_log
            }
        except Exception as e:
            error_msg = f"Unexpected error during report saving or translation in N06: {e}"
            logger.exception(error_msg, extra=extra_log_data)
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            return {
                "saved_report_path": None,
                "translated_report_path": None,
                "current_stage": "ERROR",
                "error_message": f"N06 Unexpected Error: {error_msg}",
                "error_log": error_log
            }

    # TranslationService 인스턴스를 닫는 로직 (필요하다면 워크플로우 매니저에서 관리)
    # async def close_services(self):
    #     if self.translation_service and hasattr(self.translation_service, 'close'):
    #         await self.translation_service.close()
    #         logger.info("N06SaveReportNode: Closed TranslationService.")