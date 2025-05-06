# ai/app/nodes/n06_save_report_node.py (신규 파일)
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from app.workflows.state import WorkflowState
from app.utils.logger import get_logger, PROJECT_ROOT  # logger.py 에서 PROJECT_ROOT 가져오기 가정

# 또는 설정에서 가져오기: from app.config.settings import Settings; settings = Settings()

logger = get_logger(__name__)

# 결과 저장 기본 디렉토리 (PROJECT_ROOT 기준)
DEFAULT_RESULTS_BASE_DIR = PROJECT_ROOT / "results"


class N06SaveReportNode:
    """
    N05에서 생성된 보고서 내용을 로컬 파일 시스템에 저장하는 노드.
    """
    def __init__(self, results_base_dir: Optional[Path] = None):
        """
        노드 초기화.
        Args:
            results_base_dir (Optional[Path]): 보고서를 저장할 기본 디렉토리 경로.
                                              None이면 기본값(PROJECT_ROOT/results) 사용.
        """
        self.results_base_dir = results_base_dir or DEFAULT_RESULTS_BASE_DIR
        logger.info(f"Report save directory base set to: {self.results_base_dir}")

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        """보고서 내용을 파일로 저장하고 상태 업데이트."""
        node_name = self.__class__.__name__
        trace_id = state.trace_id
        comic_id = state.comic_id
        report_content = state.report_content
        error_log = list(state.error_log or [])

        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'node_name': node_name}
        logger.info(f"Entering node. Attempting to save report content.", extra=extra_log_data)

        saved_path_str: Optional[str] = None

        # 입력 데이터 유효성 검사
        if not comic_id:
            error_msg = "Comic ID is missing, cannot determine save path."
            logger.error(error_msg, extra=extra_log_data)
            error_log.append(
                {"stage": node_name, "error": error_msg, "timestamp": datetime.now(timezone.utc).isoformat()})
            return {"error_log": error_log, "current_stage": "ERROR", "error_message": error_msg}

        if not report_content or not isinstance(report_content, str):
            error_msg = "Report content is missing or not a string. Nothing to save."
            logger.warning(error_msg, extra=extra_log_data)
            # 보고서 내용이 없어도 오류는 아닐 수 있으므로 경고만 하고 다음 단계로 진행
            # 필요시 에러 처리 변경 가능
            update_dict = {
                "saved_report_path": None,  # 저장 경로 없음
                "current_stage": "DONE",  # 저장 작업은 완료 (할 내용 없음)
                "error_log": error_log  # 기존 오류 로그 유지
            }
            logger.info("Exiting node. No report content to save.", extra=extra_log_data)
            return update_dict

        try:
            # 저장 경로 생성: PROJECT_ROOT/results/{comic_id}/report.html
            report_dir = self.results_base_dir / comic_id
            report_dir.mkdir(parents=True, exist_ok=True)  # 디렉토리 생성 (이미 있어도 오류 없음)

            report_file_path = report_dir / "report.html"

            # 파일 쓰기 (UTF-8 인코딩 사용)
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            saved_path_str = str(report_file_path.resolve())  # 절대 경로 문자열로 저장
            logger.info(f"Report successfully saved to: {saved_path_str}", extra=extra_log_data)

            update_dict = {
                "saved_report_path": saved_path_str,  # 저장된 경로 상태에 추가
                "current_stage": "DONE",  # 최종 완료 상태
                "error_log": error_log  # 오류 로그 전달
            }
            logger.info(f"Exiting node. Report saved.", extra=extra_log_data)
            return update_dict

        except OSError as e:  # 파일 시스템 관련 오류 (권한, 디스크 공간 등)
            error_msg = f"Failed to save report to file system: {e}"
            logger.exception(error_msg, extra=extra_log_data)
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            # 실패 시에도 다음 단계로 진행하되, 오류 기록 및 경로 None 설정
            return {
                "saved_report_path": None,
                "current_stage": "ERROR",  # 파일 저장 실패는 오류로 간주
                "error_message": f"N06 File Save Error: {error_msg}",
                "error_log": error_log
            }
        except Exception as e:  # 기타 예상치 못한 오류
            error_msg = f"Unexpected error during report saving: {e}"
            logger.exception(error_msg, extra=extra_log_data)
            error_log.append({"stage": node_name, "error": error_msg, "detail": traceback.format_exc(),
                              "timestamp": datetime.now(timezone.utc).isoformat()})
            return {
                "saved_report_path": None,
                "current_stage": "ERROR",
                "error_message": f"N06 Unexpected Error: {error_msg}",
                "error_log": error_log
            }