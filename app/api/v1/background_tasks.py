# ai/app/api/v1/background_tasks.py (Trace ID 일관성 및 최종 상태 처리 수정)

# version 2. (2025-05-22)
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


# version 1. (2025-05-21 back-up)
'''
import uuid
import traceback
from typing import Dict, Any, Optional
from fastapi import BackgroundTasks, HTTPException
from datetime import datetime, timezone

# --- 유틸리티 및 애플리케이션 구성 요소 임포트 ---
from app.utils.logger import get_logger, summarize_for_logging
from app.dependencies import CompiledWorkflowDep, DatabaseClientDep

logger = get_logger(__name__)


async def trigger_workflow_task(
        query: str,
        config: Dict[str, Any],
        background_tasks: BackgroundTasks,
        # 주석: FastAPI의 Depends를 직접 사용하거나, main.py/lifespan.py에서 생성된 객체를 전달받습니다.
        # 여기서는 의존성 타입 힌트만 사용하고, 실제 객체는 app.state 또는 Depends를 통해 가져온다고 가정합니다.
        compiled_app: CompiledWorkflowDep, # = Depends(get_compiled_app) 등으로 주입 가능
        db_client: DatabaseClientDep      # = Depends(get_db_client) 등으로 주입 가능
) -> str:
    """
    백그라운드로 LangGraph 워크플로우 실행을 트리거하고 초기 상태를 DB에 기록합니다.
    (업그레이드됨: ID 일관성 확보, N09 결과 처리 추가)
    """
    master_comic_id = str(uuid.uuid4())
    master_trace_id = master_comic_id
    langgraph_thread_id = master_comic_id

    writer_id_for_log = config.get('writer_id', 'default')
    extra_log_data = {'trace_id': master_trace_id, 'comic_id': master_comic_id, 'writer_id': writer_id_for_log}

    logger.info(f"백그라운드 워크플로우 트리거 시작: query='{query}', config='{summarize_for_logging(config)}'", extra=extra_log_data)

    # 1. 초기 DB 상태 설정: PENDING
    try:
        initial_status_data = {
            "comic_id": master_comic_id, "status": "PENDING", "message": "워크플로우 작업 수락됨.",
            "query": query, "writer_id": config.get("writer_id"),
            "user_site_preferences_provided": "user_site_preferences" in config,
            "timestamp_accepted": datetime.now(timezone.utc).isoformat(),
            "timestamp_start": None, "timestamp_end": None, "duration_seconds": None,
            "result": None, "error_details": None,
        }
        await db_client.set(master_comic_id, initial_status_data)
        logger.info("DB 상태 'PENDING'으로 설정 완료.", extra=extra_log_data)
    except Exception as db_err:
        logger.error(f"초기 DB 상태(PENDING) 설정 실패: {db_err}", exc_info=True, extra=extra_log_data)
        raise HTTPException(status_code=500, detail=f"워크플로우 상태 초기화 실패: {db_err}") from db_err

    # 2. 백그라운드에서 실행될 워크플로우 함수 정의
    async def workflow_runner(job_id: str, input_query: str, initial_api_config: Dict[str, Any], job_trace_id: str):
        runner_writer_id = initial_api_config.get('writer_id', 'default')
        runner_extra_log = {'trace_id': job_trace_id, 'comic_id': job_id, 'writer_id': runner_writer_id}
        start_run_time = datetime.now(timezone.utc)
        current_state_for_update = initial_status_data.copy()

        try:
            # 상태 업데이트: STARTED
            current_state_for_update.update({
                "status": "STARTED", "message": "워크플로우 실행 시작.",
                "timestamp_start": start_run_time.isoformat()
            })
            await db_client.set(job_id, current_state_for_update)
            logger.info("워크플로우 실행 시작됨 (DB 업데이트).", extra=runner_extra_log)

            # N01InitializeNode로 전달될 초기 상태값 구성
            initial_workflow_input = {
                "original_query": input_query,
                "config": initial_api_config,
                "comic_id": job_id,       # 생성된 ID 전달
                "trace_id": job_trace_id,   # 생성된 ID 전달
                # 주석: N01에서 다른 필드들은 기본값으로 초기화될 것이므로 여기서 설정 불필요
            }

            # LangGraph 실행 설정
            langgraph_execution_config = {"configurable": {"thread_id": langgraph_thread_id}}

            logger.debug(f"LangGraph ainvoke 시작... Input: {summarize_for_logging(initial_workflow_input)}",
                         extra=runner_extra_log)

            # 워크플로우 실행
            final_output: Optional[Dict] = await compiled_app.ainvoke(initial_workflow_input, config=langgraph_execution_config)

            logger.debug(f"LangGraph ainvoke 완료. Output type: {type(final_output)}", extra=runner_extra_log)
            end_run_time = datetime.now(timezone.utc)
            run_duration = (end_run_time - start_run_time).total_seconds()

            # 최종 결과 및 상태 처리
            final_status = "UNKNOWN"
            final_message = "워크플로우 완료 (상태 불명확)."
            db_result_data: Optional[Dict[str, Any]] = None
            error_details: Optional[str] = None

            if isinstance(final_output, dict):
                error_message = final_output.get("error_message")
                current_stage = final_output.get("current_stage") # 최종 스테이지 확인

                # 주석: current_stage가 END가 아닐 경우에도 오류로 간주 가능
                is_successful = not error_message and current_stage not in ["ERROR", None] # None도 오류로 간주

                final_status = "DONE" if is_successful else "FAILED"
                final_message = error_message if not is_successful else "워크플로우 성공적으로 완료됨."
                error_details = error_message if not is_successful else None

                # DB 'result' 필드 요약 정보 구성 (N09 결과 포함)
                db_result_data = {
                    "trace_id": final_output.get("trace_id"),
                    "comic_id": final_output.get("comic_id"),
                    "final_stage": current_stage,
                    "error_log_summary": summarize_for_logging(final_output.get("error_log", [])),
                    "config_summary": summarize_for_logging(final_output.get("config", {}), max_len=100),
                    "original_query": final_output.get("original_query"),
                    # --- 보고서 관련 ---
                    "report_content_length": len(final_output.get("report_content", "")),
                    "saved_report_path": final_output.get("saved_report_path"),
                    # --- 아이디어 관련 ---
                    "comic_ideas_count": len(final_output.get("comic_ideas", [])),
                    "comic_ideas_titles": [idea.get('title') for idea in final_output.get("comic_ideas", [])],
                    # --- 시나리오 관련 ---
                    "selected_comic_idea_title": final_output.get("selected_comic_idea_for_scenario", {}).get('title'),
                    "comic_scenarios_count": len(final_output.get("comic_scenarios", [])),
                    "scenario_scenes_approx": final_output.get("comic_scenarios", [{}])[0].get("generated_scenes_approx") if final_output.get("comic_scenarios") else None,
                    # --- 이미지 관련 ---
                    "generated_comic_images_count": len(final_output.get("generated_comic_images", [])),
                    "generated_images_summary": [ # 이미지 경로/URL 또는 오류 요약
                         f"{img_info.get('scene_identifier', 'Unknown')}: {img_info.get('image_path') or img_info.get('image_url') or img_info.get('error', 'Status Unknown')}"
                         for img_info in final_output.get("generated_comic_images", [])
                    ]
                }
            else:
                final_status = "FAILED"
                final_message = f"워크플로우가 예상치 못한 결과 타입({type(final_output)})을 반환했습니다."
                error_details = final_message
                logger.error(final_message, extra=runner_extra_log)

            # 최종 상태 DB 업데이트
            current_state_for_update.update({
                "status": final_status,
                "message": final_message,
                "result": db_result_data,
                "timestamp_end": end_run_time.isoformat(),
                "duration_seconds": round(run_duration, 2),
                "error_details": error_details
            })
            await db_client.set(job_id, current_state_for_update)
            logger.info(f"워크플로우 완료. 상태: {final_status}, 소요시간: {run_duration:.2f}초", extra=runner_extra_log)

        except Exception as e:
            end_run_time = datetime.now(timezone.utc)
            run_duration = (end_run_time - start_run_time).total_seconds() if start_run_time else 0
            error_msg = f"워크플로우 실행 중 심각한 오류 발생: {str(e)}"
            logger.exception(error_msg, extra=runner_extra_log)
            detailed_error_trace = traceback.format_exc()

            try:
                if 'current_state_for_update' not in locals():
                    current_state_for_update = initial_status_data.copy()
                    current_state_for_update["timestamp_start"] = start_run_time.isoformat() if start_run_time else None

                current_state_for_update.update({
                    "status": "FAILED", "message": error_msg, "error_details": detailed_error_trace,
                    "timestamp_end": end_run_time.isoformat(), "duration_seconds": round(run_duration, 2)
                })
                await db_client.set(job_id, current_state_for_update)
            except Exception as db_err:
                logger.error(f"오류 상태 DB 업데이트 실패: {db_err}", extra=runner_extra_log)

    # 3. 백그라운드 작업 추가
    background_tasks.add_task(workflow_runner, master_comic_id, query, config, master_trace_id)
    logger.info("백그라운드 작업 스케줄 완료.", extra=extra_log_data)

    return master_comic_id
'''