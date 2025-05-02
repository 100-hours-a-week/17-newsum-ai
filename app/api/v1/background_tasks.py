# app/api/v1/background_tasks.py
import logging
import uuid
import traceback # 상세 오류 로깅용
from typing import Dict, Any, Optional # 타입 힌트용
from fastapi import BackgroundTasks, Depends, HTTPException # Depends, HTTPException 추가
from datetime import datetime, timezone

# --- 필요한 클래스 및 설정 임포트 ---
# (경로는 실제 프로젝트 구조에 맞게 조정)
from app.workflows.state import ComicState
from app.services.database_con_client_v2 import DatabaseClientV2
from app.config.settings import settings
from app.utils.logger import get_logger
# --- FastAPI 앱 컨텍스트에서 생성/관리된 객체를 가져오기 위한 의존성 주입 함수 (예시) ---
from app.dependencies import get_compiled_workflow_app, get_db_client # 실제 구현 필요
from langgraph.graph import StateGraph # StateGraph 타입 힌트용

logger = get_logger("BackgroundTasks") # 로거 이름 변경

# --- 백그라운드 작업 트리거 함수 ---
async def trigger_workflow_task(
    query: str,
    background_tasks: BackgroundTasks,
    # --- 의존성 주입 사용 ---
    compiled_app: StateGraph = Depends(get_compiled_workflow_app), # 미리 컴파일된 앱 주입
    db_client: DatabaseClientV2 = Depends(get_db_client)       # DB 클라이언트 주입
) -> str:
    """
    (수정됨) 백그라운드로 LangGraph 워크플로우 실행을 트리거하고 DB 상태를 업데이트합니다.
    FastAPI 의존성 주입을 통해 미리 컴파일된 LangGraph 앱과 DB 클라이언트를 받습니다.

    Args:
        query (str): 사용자의 초기 쿼리.
        background_tasks (BackgroundTasks): FastAPI의 백그라운드 태스크 객체.
        compiled_app (StateGraph): 미리 컴파일된 LangGraph 애플리케이션 (주입됨).
        db_client (DatabaseClientV2): 데이터베이스 클라이언트 인스턴스 (주입됨).

    Returns:
        str: 생성된 코믹 작업의 고유 ID (comic_id).

    Raises:
        HTTPException: 초기 DB 상태 설정 실패 시.
    """
    comic_id = str(uuid.uuid4()) # 고유 ID 생성
    trace_id = comic_id # trace_id로 comic_id 사용
    extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}

    logger.info(f"백그라운드 워크플로우 트리거: query='{query}'", extra=extra_log_data)

    # --- 워크플로우 실행을 위한 내부 비동기 함수 ---
    async def workflow_runner(comic_id: str, initial_query: str, trace_id: str):
        """(수정됨) 실제 워크플로우를 실행하고 상태를 업데이트하는 내부 함수."""
        runner_extra_log = {'trace_id': trace_id, 'comic_id': comic_id}
        start_run_time = datetime.now(timezone.utc) # 실행 시작 시간

        try:
            # 1. 상태 업데이트: 시작됨 (STARTED)
            # DB 클라이언트가 주입되므로 직접 사용 가능
            await db_client.set(comic_id, {
                "status": "STARTED",
                "message": "워크플로우 실행 시작.",
                "query": initial_query,
                "timestamp_start": start_run_time.isoformat()
            })
            logger.info("워크플로우 실행 시작됨.", extra=runner_extra_log)

            # 2. 초기 입력 준비
            initial_input = {"initial_query": initial_query}

            # 3. 워크플로우 실행 (ainvoke - 블로킹 가능성 있음!)
            # compiled_app은 외부에서 주입받아 사용
            config = {"configurable": {"thread_id": comic_id}} # 체크포인터용 설정
            logger.debug("LangGraph ainvoke 시작...", extra=runner_extra_log)

            final_output = await compiled_app.ainvoke(initial_input, config=config)

            logger.debug("LangGraph ainvoke 완료.", extra=runner_extra_log)
            end_run_time = datetime.now(timezone.utc) # 종료 시간
            run_duration = (end_run_time - start_run_time).total_seconds() # 소요 시간

            # 4. 최종 상태 추출 및 DB 업데이트
            final_state_data: Dict[str, Any] = {} # 최종 상태 저장용
            error_message: Optional[str] = None

            # ainvoke 결과에서 상태 추출 시도 (이전과 동일한 로직)
            if isinstance(final_output, ComicState):
                final_state_data = final_output.model_dump()
                error_message = final_state_data.get("error_message")
            elif isinstance(final_output, dict):
                 possible_state = next((v for v in final_output.values() if isinstance(v, ComicState)), None)
                 if possible_state:
                      final_state_data = possible_state.model_dump()
                      error_message = final_state_data.get("error_message")
                 else:
                      logger.warning("ainvoke 결과에서 ComicState 객체를 찾지 못함. 반환된 dict 사용 시도.", extra=runner_extra_log)
                      final_state_data = final_output
                      error_message = final_state_data.get("error_message") # 오류 메시지 키 확인
            else:
                 logger.error(f"예상치 못한 ainvoke 결과 타입: {type(final_output)}", extra=runner_extra_log)
                 error_message = f"예상치 못한 최종 상태 타입: {type(final_output)}"

            final_status = "DONE" if not error_message else "FAILED"
            final_message = error_message or "워크플로우 성공적으로 완료됨."
            # 최종 결과에서 필요한 데이터 추출 (예: final_comic 필드)
            result_data = {"final_comic": final_state_data.get("final_comic", {})}

            # 최종 상태 DB 업데이트
            await db_client.set(comic_id, {
                "status": final_status,
                "message": final_message,
                "result": result_data,
                "timestamp_end": end_run_time.isoformat(),
                "duration_seconds": round(run_duration, 2), # 소요 시간 추가
                # "processing_stats": final_state_data.get("processing_stats", {}), # 처리 통계 추가 (선택 사항)
            })
            logger.info(f"워크플로우 완료. 상태: {final_status}, 소요시간: {run_duration:.2f}초", extra=runner_extra_log)

        except Exception as e:
            # 워크플로우 실행 중 예외 처리
            end_run_time = datetime.now(timezone.utc)
            run_duration = (end_run_time - start_run_time).total_seconds()
            error_msg = f"워크플로우 실행 오류: {str(e)}"
            logger.exception(error_msg, extra=runner_extra_log)
            detailed_error = traceback.format_exc() # 상세 오류 스택
            logger.error(f"Traceback:\n{detailed_error}", extra=runner_extra_log)
            # 오류 상태 DB 기록
            try:
                await db_client.set(comic_id, {
                    "status": "FAILED",
                    "message": error_msg,
                    "error_details": detailed_error,
                    "timestamp_end": end_run_time.isoformat(),
                    "duration_seconds": round(run_duration, 2)
                })
            except Exception as db_err:
                 logger.error(f"오류 상태 DB 업데이트 실패: {db_err}", extra=runner_extra_log)

    # 1. 초기 DB 상태 설정: PENDING
    try:
        await db_client.set(comic_id, {
            "status": "PENDING",
            "message": "워크플로우 작업 수락됨.",
            "query": query,
            "timestamp_accepted": datetime.now(timezone.utc).isoformat()
        })
        logger.info("DB 상태 'PENDING'으로 설정됨.", extra=extra_log_data)
    except Exception as db_err:
        # 초기 DB 설정 실패 시, 500 오류 반환하여 클라이언트에게 알림
        logger.error(f"초기 DB 상태(PENDING) 설정 실패: {db_err}", exc_info=True, extra=extra_log_data)
        raise HTTPException(status_code=500, detail="워크플로우 상태 초기화 실패") from db_err

    # 2. 백그라운드 작업 스케줄링
    # workflow_runner에 필요한 인자(comic_id, query, trace_id) 전달
    background_tasks.add_task(workflow_runner, comic_id, query, trace_id)
    logger.info("백그라운드 작업 스케줄됨.", extra=extra_log_data)

    return comic_id # 생성된 코믹 ID 반환