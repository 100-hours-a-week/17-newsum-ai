# ai/app/nodes/n06_save_report_node.py

import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger, PROJECT_ROOT
from app.config.settings import Settings

logger = get_logger(__name__)
settings = Settings()

DEFAULT_RESULTS_BASE_DIR = PROJECT_ROOT / "results"


class N06SaveReportNode:
    """
    N05에서 생성된 보고서 내용을 로컬 파일 시스템에 저장하는 노드.
    (Qwen3가 한국어로 출력하기 때문에 별도 번역은 수행하지 않음)
    """

    def __init__(self, results_base_dir: Optional[Path] = None):
        """
        노드 초기화.

        Args:
            results_base_dir (Optional[Path]): 보고서를 저장할 기본 디렉토리 경로.
                                               None이면 기본값(PROJECT_ROOT/results) 사용.
        """
        self.results_base_dir = results_base_dir or DEFAULT_RESULTS_BASE_DIR
        logger.info(f"N06: Report save directory base set to: {self.results_base_dir}")

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        """
        보고서 내용을 파일로 저장합니다.
        """
        meta = state.meta
        report_sec = state.report
        node_name = self.__class__.__name__
        trace_id = meta.trace_id
        comic_id = meta.comic_id
        report_content = report_sec.report_content
        error_log = list(meta.error_log or [])

        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'node_name': node_name}
        logger.info(f"N06: Entering node. Attempting to save report content.", extra=extra_log_data)

        if not comic_id:
            error_msg = "Comic ID is missing, cannot determine save path."
            logger.error(error_msg, extra=extra_log_data)
            error_log.append({
                "stage": node_name,
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            meta.current_stage = "ERROR"
            meta.error_log = error_log
            meta.error_message = error_msg
            return {"report": report_sec.model_dump(), "meta": meta.model_dump()}

        if not report_content or not isinstance(report_content, str):
            error_msg = "Report content is missing or not a string. Nothing to save."
            logger.warning(error_msg, extra=extra_log_data)
            meta.current_stage = "DONE"
            report_sec.saved_report_path = None
            return {"report": report_sec.model_dump(), "meta": meta.model_dump()}

        try:
            report_dir = self.results_base_dir / comic_id
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "report.html"

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            saved_path_str = str(report_path.resolve())
            logger.info(f"N06: Report successfully saved to: {saved_path_str}", extra=extra_log_data)
            report_sec.saved_report_path = saved_path_str

            meta.current_stage = "DONE"
            meta.error_log = error_log
            return {"report": report_sec.model_dump(), "meta": meta.model_dump()}

        except Exception as e:
            error_msg = f"N06: Error while saving report: {e}"
            logger.exception(error_msg, extra=extra_log_data)
            error_log.append({
                "stage": node_name,
                "error": error_msg,
                "detail": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            meta.current_stage = "ERROR"
            meta.error_log = error_log
            meta.error_message = error_msg
            report_sec.saved_report_path = None
            return {"report": report_sec.model_dump(), "meta": meta.model_dump()}
