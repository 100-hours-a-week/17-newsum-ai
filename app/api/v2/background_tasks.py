# ai/app/api/v2/background_tasks.py (Trace ID 일관성 및 최종 상태 처리 수정)

# version 2.2 (2025-05-27)
import uuid
import traceback
from typing import Dict, Any, Optional
from fastapi import BackgroundTasks, HTTPException
from datetime import datetime, timezone

from app.utils.logger import get_logger, summarize_for_logging
from app.dependencies import CompiledWorkflowDep, DatabaseClientDep

logger = get_logger(__name__)


# Helper shortcuts to safely pull nested keys
def _g(d: Dict[str, Any], *path, default=None):
    cur = d
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


async def trigger_workflow_task(
    query: str,
    config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    compiled_app: CompiledWorkflowDep,
    db_client: DatabaseClientDep,
) -> str:
    master_comic_id = str(uuid.uuid4())
    master_trace_id = master_comic_id
    langgraph_thread_id = master_comic_id

    extra_log = {"trace_id": master_trace_id, "comic_id": master_comic_id}
    logger.info("BG workflow trigger", extra=extra_log)

    # --- DB PENDING 상태 기록 ---
    initial_db_state = {
        "comic_id": master_comic_id,
        "status": "PENDING",
        "timestamp_accepted": datetime.now(timezone.utc).isoformat(),
        "query": query,
    }
    await db_client.set(master_comic_id, initial_db_state)

    # 내부 실행 함수 -------------------------------------------------
    async def workflow_runner(job_id: str, raw_query: str, cfg: Dict[str, Any], trace_id: str):
        start_ts = datetime.now(timezone.utc)
        runner_log = {"trace_id": trace_id, "comic_id": job_id}

        try:
            await db_client.set(job_id, {**initial_db_state, "status": "STARTED", "timestamp_start": start_ts.isoformat()})

            init_input = {
                "query": {"original_query": raw_query},  # section 경로에 맞춰 배치
                "config": {"config": cfg},
                "meta": {"comic_id": job_id, "trace_id": trace_id},
            }

            final_output: Optional[Dict[str, Any]] = await compiled_app.ainvoke(
                init_input, config={"configurable": {"thread_id": langgraph_thread_id}}
            )

            end_ts = datetime.now(timezone.utc)
            duration = (end_ts - start_ts).total_seconds()

            # ----- 결과 요약 -----
            final_status = "FAILED"
            final_msg = "Unexpected result"
            error_details = None
            result_summary = None

            if isinstance(final_output, dict):
                err_msg = _g(final_output, "meta", "error_message")
                is_ok = err_msg is None
                final_status = "DONE" if is_ok else "FAILED"
                final_msg = "성공" if is_ok else err_msg
                error_details = None if is_ok else err_msg

                # report, idea, scenario, image 등 Section별로 dict를 저장할 때, raw_search_results 등 dict 리스트에 rank 필드가 누락되지 않도록 보완
                report_content = _g(final_output, "report", "report_content", default="")
                saved_report_path = _g(final_output, "report", "saved_report_path")
                comic_ideas = _g(final_output, "idea", "comic_ideas", default=[])
                comic_scenarios = _g(final_output, "scenario", "comic_scenarios", default=[])
                generated_comic_images = _g(final_output, "image", "generated_comic_images", default=[])
                raw_search_results = _g(final_output, "search", "raw_search_results", default=[])
                if isinstance(raw_search_results, list):
                    for idx, item in enumerate(raw_search_results):
                        if isinstance(item, dict) and 'rank' not in item:
                            item['rank'] = idx + 1

                result_summary = {
                    "trace_id": _g(final_output, "meta", "trace_id"),
                    "comic_id": _g(final_output, "meta", "comic_id"),
                    "final_stage": _g(final_output, "meta", "current_stage"),
                    # report
                    "report_len": len(report_content),
                    "saved_report_path": saved_report_path,
                    # idea
                    "ideas_cnt": len(comic_ideas),
                    # scenario
                    "scenarios_cnt": len(comic_scenarios),
                    # images
                    "images_cnt": len(generated_comic_images),
                }
            else:
                error_details = f"Output type: {type(final_output)}"

            await db_client.set(job_id, {
                **initial_db_state,
                "status": final_status,
                "message": final_msg,
                "result": result_summary,
                "error_details": error_details,
                "timestamp_start": start_ts.isoformat(),
                "timestamp_end": end_ts.isoformat(),
                "duration_seconds": round(duration, 2),
            })
            logger.info(f"Workflow {final_status}", extra=runner_log)

        except Exception as e:
            logger.exception("Workflow fatal", extra=runner_log)
            await db_client.set(job_id, {**initial_db_state, "status": "FAILED", "message": str(e)})

    # FastAPI Background task 등록
    background_tasks.add_task(workflow_runner, master_comic_id, query, config, master_trace_id)
    return master_comic_id

