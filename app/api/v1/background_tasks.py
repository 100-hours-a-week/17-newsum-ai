# ai/app/api/v1/background_tasks.py (최종 수정본 - 중복 호출 제거 및 타입 처리 수정)
import uuid
import traceback
from typing import Dict, Any, Optional
from fastapi import BackgroundTasks, HTTPException
from datetime import datetime, timezone

# --- 유틸리티 및 애플리케이션 구성 요소 임포트 ---
from app.utils.logger import get_logger, summarize_for_logging
from app.dependencies import CompiledWorkflowDep, DatabaseClientDep
# WorkflowState는 타입 힌트용으로만 사용
from app.workflows.state import WorkflowState

logger = get_logger(__name__)


async def trigger_workflow_task(
        query: str,
        config: Dict[str, Any],
        background_tasks: BackgroundTasks,
        compiled_app: CompiledWorkflowDep,
        db_client: DatabaseClientDep
) -> str:
    """
    백그라운드로 LangGraph 워크플로우 실행을 트리거하고 초기 상태를 DB에 기록합니다.
    (업그레이드됨: 중복 ainvoke 호출 제거, 최종 상태 타입 처리 수정)
    """
    comic_id = str(uuid.uuid4())
    trace_id = comic_id
    writer_id_for_log = config.get('writer_id', 'default')
    extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'writer_id': writer_id_for_log}

    logger.info(f"백그라운드 워크플로우 트리거 시작: query='{query}', config='{summarize_for_logging(config)}'", extra=extra_log_data)

    # 1. 초기 DB 상태 설정: PENDING (이전과 동일)
    try:
        initial_status_data = {
            "comic_id": comic_id, "status": "PENDING", "message": "워크플로우 작업 수락됨.",
            "query": query, "writer_id": config.get("writer_id"),
            "user_site_preferences_provided": "user_site_preferences" in config,
            "timestamp_accepted": datetime.now(timezone.utc).isoformat(),
            "timestamp_start": None, "timestamp_end": None, "duration_seconds": None,
            "result": None, "error_details": None,
        }
        await db_client.set(comic_id, initial_status_data)
        logger.info("DB 상태 'PENDING'으로 설정 완료.", extra=extra_log_data)
    except Exception as db_err:
        logger.error(f"초기 DB 상태(PENDING) 설정 실패: {db_err}", exc_info=True, extra=extra_log_data)
        raise HTTPException(status_code=500, detail=f"워크플로우 상태 초기화 실패: {db_err}") from db_err

    # 2. 백그라운드에서 실행될 워크플로우 함수 정의
    async def workflow_runner(job_id: str, input_query: str, initial_config: Dict[str, Any], job_trace_id: str):
        runner_writer_id = initial_config.get('writer_id', 'default')
        runner_extra_log = {'trace_id': job_trace_id, 'comic_id': job_id, 'writer_id': runner_writer_id}
        start_run_time = datetime.now(timezone.utc)

        current_state_for_update = initial_status_data.copy()

        try:
            # 상태 업데이트: STARTED (이전과 동일)
            current_state_for_update.update({
                "status": "STARTED", "message": "워크플로우 실행 시작.",
                "timestamp_start": start_run_time.isoformat()
            })
            await db_client.set(job_id, current_state_for_update)
            logger.info("워크플로우 실행 시작됨 (DB 업데이트).", extra=runner_extra_log)

            # --- LangGraph 워크플로우 실행 ---
            initial_input = {
                "original_query": input_query,
                "config": initial_config
            }
            langgraph_config = {"configurable": {"thread_id": job_id}}
            logger.debug(f"LangGraph ainvoke 시작... Input: {summarize_for_logging(initial_input)}",
                         extra=runner_extra_log)

            # --- 워크플로우 호출 (단일 호출) ---
            # <<< 수정: ainvoke 한 번만 호출하고 결과 저장 >>>
            final_output: Optional[Dict] = await compiled_app.ainvoke(initial_input, config=langgraph_config)
            # ---------------------------------

            logger.debug(f"LangGraph ainvoke 완료. Output type: {type(final_output)}", extra=runner_extra_log)
            end_run_time = datetime.now(timezone.utc)
            run_duration = (end_run_time - start_run_time).total_seconds()

            # --- 최종 결과 및 상태 처리 (dict 타입으로 처리) ---
            final_status = "UNKNOWN"
            final_message = "워크플로우 완료 (상태 불명확)."
            db_result_data: Optional[Dict[str, Any]] = None
            error_details: Optional[str] = None

            # <<< 수정: final_output 타입을 dict로 간주하고 처리 >>>
            if isinstance(final_output, dict):  # AddableValuesDict도 dict처럼 처리
                # .get()을 사용하여 안전하게 값 접근
                # 상태 객체에 정의된 최상위 에러 메시지 필드 확인 (WorkflowState.error_message)
                error_message = final_output.get("error_message")
                current_stage = final_output.get("current_stage")

                # 상태 결정: error_message가 있거나 최종 stage가 ERROR면 FAILED
                final_status = "DONE" if not error_message and current_stage != "ERROR" else "FAILED"
                final_message = error_message if final_status == "FAILED" else "워크플로우 성공적으로 완료됨."
                error_details = error_message if final_status == "FAILED" else None

                # DB result 필드 구성 (dict의 .get() 사용, 요약 로직 유지)
                db_result_data = {
                    "trace_id": final_output.get("trace_id"),
                    "comic_id": final_output.get("comic_id"),
                    "timestamp": final_output.get("timestamp"),
                    "current_stage": current_stage,
                    "error_log_summary": summarize_for_logging(final_output.get("error_log", []), max_len=200),
                    "config": final_output.get("config", {}),
                    "original_query": final_output.get("original_query"),
                    "query_context_summary": summarize_for_logging(
                        final_output.get("query_context", {}), max_len=150,
                        fields_to_show=["refined_intent", "key_aspects_to_search", "unresolved_ambiguities",
                                        "clarification_error"]
                    ),
                    "initial_context_results_summary": summarize_for_logging(
                        final_output.get("initial_context_results", []), max_len=150),
                    "raw_search_results_summary": summarize_for_logging(final_output.get("raw_search_results", []),
                                                                        max_len=150),
                    "saved_report_path": final_output.get("saved_report_path"),  # N06 결과
                    "report_content_length": len(final_output.get("report_content", "")) if final_output.get(
                        "report_content") else 0,
                    "report_content_preview": summarize_for_logging(final_output.get("report_content"),
                                                                    max_len=200) if final_output.get(
                        "report_content") else None,
                    "search_strategy_summary": summarize_for_logging(final_output.get("search_strategy", {}),
                                                                     max_len=150,
                                                                     fields_to_show=["writer_concept", "selected_tools",
                                                                                     "queries"])
                }
            else:  # dict 형태가 아닐 경우
                final_status = "FAILED"
                final_message = f"워크플로우가 예상치 못한 결과 타입을 반환했습니다: {type(final_output)}"
                error_details = final_message
                logger.error(final_message, extra=runner_extra_log)
            # --------------------------------------------------------

            # --- 최종 상태 DB 업데이트 ---
            current_state_for_update.update({
                "status": final_status,
                "message": final_message,
                "result": db_result_data,  # 수정된 db_result_data 사용
                "timestamp_end": end_run_time.isoformat(),
                "duration_seconds": round(run_duration, 2),
                "error_details": error_details
            })
            await db_client.set(job_id, current_state_for_update)
            logger.info(f"워크플로우 완료. 상태: {final_status}, 소요시간: {run_duration:.2f}초", extra=runner_extra_log)

        except Exception as e:
            # ... (오류 처리 로직은 이전과 동일) ...
            end_run_time = datetime.now(timezone.utc)
            run_duration = (end_run_time - start_run_time).total_seconds() if start_run_time else 0
            error_msg = f"워크플로우 실행 중 심각한 오류 발생: {str(e)}"
            logger.exception(error_msg, extra=runner_extra_log)
            detailed_error_trace = traceback.format_exc()

            try:
                if 'current_state_for_update' not in locals(): current_state_for_update = initial_status_data.copy()
                current_state_for_update.update({
                    "status": "FAILED", "message": error_msg, "error_details": detailed_error_trace,
                    "timestamp_end": end_run_time.isoformat(), "duration_seconds": round(run_duration, 2)
                })
                await db_client.set(job_id, current_state_for_update)
            except Exception as db_err:
                logger.error(f"오류 상태 DB 업데이트 실패: {db_err}", extra=runner_extra_log)

    # 3. 정의된 워크플로우 함수를 백그라운드 작업으로 추가 (이전과 동일)
    background_tasks.add_task(workflow_runner, comic_id, query, config, trace_id)
    logger.info("백그라운드 작업 스케줄 완료.", extra=extra_log_data)

    return comic_id