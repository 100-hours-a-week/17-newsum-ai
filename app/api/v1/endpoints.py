# ai/app/api/v1/endpoints.py

import json
from fastapi import APIRouter, HTTPException, Body, BackgroundTasks, Path, Depends
from typing import Annotated, Dict, Any, Optional  # 필요한 타입 임포트

from app.utils.logger import get_logger
# 스키마 파일에서 모델 임포트 (경로는 실제 프로젝트 구조에 맞게 조정 필요)
from .schemas import AsyncComicRequest, AsyncComicResponse, ComicStatusResponse
from .background_tasks import trigger_workflow_task
# 의존성 주입 관련 임포트
from app.dependencies import CompiledWorkflowDep, DatabaseClientDep

# LLM 큐 라우터 임포트
from .llm_endpoints import router as llm_router

logger = get_logger(__name__)

router = APIRouter(tags=["Comics V1"])  # API 라우터 설정

# LLM 큐 라우터 포함
router.include_router(llm_router, prefix="/llm", tags=["LLM Queue"])


@router.post(
    "/comics",
    response_model=AsyncComicResponse,
    summary="비동기 만화 생성 요청",
    description="쿼리와 선택적 작가 ID, 사이트 설정을 포함하는 구조로 요청합니다.",
    status_code=202  # Accepted 상태 코드 반환
)
async def request_comic_generation(
        # FastAPI의 의존성 주입 시스템 사용 예시 (또는 _shared_state 직접 사용)
        compiled_app: CompiledWorkflowDep,
        db_client: DatabaseClientDep,
        # 요청 본문은 AsyncComicRequest 스키마로 유효성 검사 및 파싱
        request_data: AsyncComicRequest = Body(...),
        # 백그라운드 작업 실행을 위한 FastAPI 객체
        background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    만화 생성을 비동기적으로 요청하는 API 엔드포인트입니다.
    사용자 쿼리, 작가 ID, 선호 검색 사이트 정보를 받아 워크플로우를 시작합니다.
    """
    # 요청 데이터에서 필요한 정보 추출
    user_query = request_data.data.query
    user_writer_id = request_data.writer_id
    user_site_preferences_model = request_data.data.site  # Pydantic 모델 형태

    log_extra = {'writer_id': user_writer_id or 'default'}  # 로그용 컨텍스트
    logger.info(
        f"POST /comics 요청 수신: query='{user_query}', "
        f"writer_id='{user_writer_id or 'default'}', "
        f"site_preferences_provided={user_site_preferences_model is not None}",
        extra=log_extra
    )

    # --- 워크플로우 시작 시 전달할 config 딕셔너리 구성 ---
    config_for_workflow: Dict[str, Any] = {
        "writer_id": user_writer_id
        # 필요 시 request_data에서 다른 설정값 추출하여 config에 추가 가능
        # 예: "target_audience": request_data.data.target_audience
    }
    # 사용자 지정 사이트 정보가 있으면 config에 추가
    if user_site_preferences_model:
        # Pydantic 모델을 dict로 변환하여 저장 (None 값 제외)
        config_for_workflow["user_site_preferences"] = user_site_preferences_model.model_dump(exclude_none=True)
    # ---------------------------------------------------

    try:
        # 백그라운드 작업 트리거 (config 딕셔너리 전체 전달)
        comic_id = await trigger_workflow_task(
            query=user_query,
            config=config_for_workflow,  # config 딕셔너리 전달
            background_tasks=background_tasks,
            compiled_app=compiled_app,
            db_client=db_client
        )
        logger.info(f"백그라운드 작업 시작됨. comic_id: {comic_id}", extra=log_extra)

        # 클라이언트에 작업 수락 응답 반환
        return AsyncComicResponse(
            comic_id=comic_id,
            status="PENDING",  # 초기 상태
            message="만화 생성 작업이 수락되어 백그라운드에서 시작되었습니다."
        )
    except HTTPException as http_exc:
        # trigger_workflow_task 등에서 발생시킨 HTTP 예외 처리
        logger.error(f"HTTP 예외 발생: {http_exc.status_code} - {http_exc.detail}", extra=log_extra)
        raise http_exc  # FastAPI로 예외 다시 전달
    except Exception as e:
        # 기타 예상치 못한 서버 오류 처리
        logger.exception(f"POST /comics 요청 처리 중 예기치 않은 예외 발생: {e}", extra=log_extra)
        raise HTTPException(status_code=500, detail=f"워크플로우 시작 중 내부 서버 오류 발생.")


@router.get(
    "/comics/status/{comic_id}",
    response_model=ComicStatusResponse,
    summary="만화 생성 상태 조회",
    description="제공된 comic_id에 해당하는 작업의 현재 상태와 결과를 조회합니다."
)
async def get_comic_status(
        db_client: DatabaseClientDep,  # DB 클라이언트 의존성 주입
        comic_id: str = Path(..., description="조회할 작업의 고유 ID", example="9d8be988-833e-4500-8e38-8d45ca150449")
        # 경로 파라미터
):
    """
    특정 만화 생성 작업의 상태와 요약된 결과를 DB에서 조회하는 엔드포인트입니다.
    """
    logger.info(f"GET /comics/status/{comic_id} 요청 수신")
    extra_log_data = {'comic_id': comic_id}
    try:
        # DB에서 해당 comic_id의 데이터 조회
        status_data_raw = await db_client.get(comic_id)

        if status_data_raw:
            # DB 데이터 파싱 (JSON 문자열 또는 이미 dict 형태일 수 있음)
            parsed_data = {}
            if isinstance(status_data_raw, str):
                try:
                    parsed_data = json.loads(status_data_raw)
                except json.JSONDecodeError:
                    logger.warning(f"DB 상태 데이터 파싱 실패 (JSON 형식 오류).", extra=extra_log_data)
                    raise HTTPException(status_code=500, detail="저장된 상태 데이터 형식 오류")
            elif isinstance(status_data_raw, dict):
                parsed_data = status_data_raw
            else:
                logger.warning(f"DB 상태 데이터 타입 미지원: type={type(status_data_raw)}", extra=extra_log_data)
                raise HTTPException(status_code=500, detail="저장된 상태 데이터 형식 오류")

            # Pydantic 모델을 사용하여 응답 데이터 검증 및 반환
            # DB에 저장된 키와 ComicStatusResponse 모델 필드가 일치해야 함
            try:
                return ComicStatusResponse(**parsed_data)
            except Exception as pydantic_err:  # Pydantic 유효성 검사 오류 등
                logger.error(f"DB 데이터를 응답 모델로 변환 실패: {pydantic_err}", extra=extra_log_data)
                raise HTTPException(status_code=500, detail="상태 데이터 처리 중 오류 발생")

        else:
            # 해당 ID의 상태 정보가 없을 경우 404 오류 반환
            logger.warning(f"요청한 comic_id의 상태 정보 없음.", extra=extra_log_data)
            raise HTTPException(status_code=404, detail="해당 Comic ID의 작업 상태를 찾을 수 없습니다.")

    except HTTPException as http_exc:
        # 이미 처리된 HTTP 예외는 그대로 전달
        raise http_exc
    except Exception as e:
        # 기타 예상치 못한 오류 (DB 연결 오류 등)
        logger.exception(f"GET /comics/status/{comic_id} 요청 처리 중 예기치 않은 예외 발생: {e}", extra=extra_log_data)
        raise HTTPException(status_code=500, detail=f"상태 조회 중 내부 서버 오류 발생.")