# ai/app/api/v2/process_turn_v2.py

# FastAPI 관련 모듈 임포트
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
import time # 처리 시간 측정을 위한 time 모듈

# 현재 프로젝트 내 다른 모듈들 임포트
from app.api.v2.schemas.request_v2 import ProcessTurnRequest # v2 API 요청 스키마
from app.api.v2.schemas.response_v2 import ProcessTurnResponse # v2 API 응답 스키마
from app.config.settings import settings # 프로젝트 설정값 (예: DEBUG_MODE, API URL 등)
from app.utils.logger import get_logger # 로깅 유틸리티
from app.utils.performance_monitor import performance_monitor # (가상) 성능 모니터링 유틸리티
from app.utils.error_handler import error_handler # (가상) 오류 처리 통계 유틸리티
from app.dependencies import TurnProcessingServiceDep # 의존성 주입을 통해 TurnProcessingService 인스턴스를 받기 위함

# 로거 인스턴스 생성
logger = get_logger(__name__)

# APIRouter 인스턴스 생성. 이 라우터에 등록된 API들은 /v2 접두사를 가지며, "HEMA V2" 태그로 그룹화됨.
router = APIRouter(tags=["HEMA V2"], prefix="/v2")


# 메인 턴 처리 엔드포인트 정의
@router.post(
    "/process_turn", # API 경로: /v2/process_turn
    response_model=ProcessTurnResponse, # 응답 본문의 스키마 정의
    summary="HEMA 통합 턴 처리", # API 문서에 표시될 요약 정보
    description="사용자 메시지를 받아 HEMA 컨텍스트와 함께 SLM 처리 후 응답 생성", # API 문서에 표시될 상세 설명
    status_code=200 # 성공 시 기본 HTTP 상태 코드 (FastAPI는 성공 시 기본 200을 사용하므로 명시하지 않아도 됨)
)
async def process_turn(
    request_data: ProcessTurnRequest, # 요청 본문을 ProcessTurnRequest 스키마로 유효성 검사 및 파싱
    background_tasks: BackgroundTasks, # FastAPI의 BackgroundTasks 의존성 주입 (현재 코드에서는 직접 사용되지 않음)
    turn_service: TurnProcessingServiceDep, # 의존성 주입을 통해 TurnProcessingService 인스턴스 획득
):
    """HEMA 통합 턴 처리 메인 엔드포인트"""
    start_time = time.time() # 요청 처리 시작 시간 기록

    try:
        # --- 입력값 검증 ---
        # user_id가 비어있는지 확인 (공백 제거 후)
        if not request_data.user_id.strip():
            # 유효하지 않은 경우 400 Bad Request 오류 발생
            raise HTTPException(status_code=400, detail="user_id는 필수입니다.")

        # session_id가 비어있는지 확인 (공백 제거 후)
        if not request_data.session_id.strip():
            # 유효하지 않은 경우 400 Bad Request 오류 발생
            raise HTTPException(status_code=400, detail="session_id는 필수입니다.")

        # user_message가 비어있는지 확인 (공백 제거 후)
        if not request_data.user_message.strip():
            # 유효하지 않은 경우 400 Bad Request 오류 발생
            raise HTTPException(status_code=400, detail="user_message는 필수입니다.")

        # 처리 시작 로깅
        logger.info(f"Processing turn for user {request_data.user_id}, session {request_data.session_id}")

        # --- 핵심 로직 호출 (TurnProcessingService) ---
        # 의존성 주입으로 받은 turn_service의 process_user_turn 메소드 호출
        # 실제 HEMA 컨텍스트 구성, SLM 작업 요청 및 결과 처리 등의 복잡한 로직은 이 서비스 내부에 구현됨
        response = await turn_service.process_user_turn(
            user_id=request_data.user_id,
            session_id=request_data.session_id,
            user_message=request_data.user_message,
            metadata=request_data.metadata # metadata는 Optional이므로 그대로 전달
        )

        # --- 성공 응답 반환 ---
        # turn_service로부터 받은 결과(딕셔너리 형태 예상)를 ProcessTurnResponse 스키마에 맞춰 반환
        # ProcessTurnResponse(**response)는 response 딕셔너리의 키-값 쌍을 ProcessTurnResponse 모델의 필드에 매핑
        return ProcessTurnResponse(**response)

    except HTTPException:
        # FastAPI의 HTTPException 예외는 그대로 다시 발생시켜 FastAPI가 처리하도록 함
        # (예: 위에서 발생시킨 400 오류)
        raise
    except Exception as e:
        # 기타 예상치 못한 예외 처리
        processing_time = time.time() - start_time # 예외 발생까지의 처리 시간 계산
        logger.exception(f"Turn processing failed: {e}") # 예외 정보 로깅 (스택 트레이스 포함)

        # --- 오류 응답 반환 ---
        # 사용자에게는 일반적인 오류 메시지를 전달하고, 필요한 내부 정보(처리 시간, 디버그 정보 등)를 포함
        return ProcessTurnResponse(
            session_id=request_data.session_id, # 오류 발생 시에도 session_id는 유지하여 추적 용이성 확보
            response_to_user="죄송합니다. 현재 서비스 준비 중입니다.", # 사용자 친화적 오류 메시지
            hema_update_status="failed", # HEMA 데이터 업데이트 상태 (오류로 인해 실패)
            processing_time=processing_time, # 처리 시간 정보
            debug_info={ # 디버그 모드일 경우에만 오류 상세 정보 포함
                "error": str(e)
            } if settings.DEBUG_MODE else None
        )


# 서비스 상태 확인 엔드포인트
@router.get(
    "/health", # API 경로: /v2/health
    summary="HEMA v2 서비스 상태 확인",
    description="HEMA v2 서비스의 상태를 확인합니다."
    # response_model을 명시하지 않으면 FastAPI가 자동으로 JSON 응답 처리
)
async def health_check():
    """HEMA v2 서비스 헬스 체크"""
    # 서비스가 정상적으로 응답하는지 확인하기 위한 간단한 상태 정보 반환
    return {
        "status": "healthy", # 서비스 상태
        "version": "2.0.0", # 서비스 버전 (하드코딩 또는 설정값 사용 가능)
        "services": { # (가상) 의존하는 주요 서비스들의 상태 (실제 연결 상태 체크 로직 추가 가능)
            "hema_service": "development", # HEMA 서비스 상태
            "slm_task_manager": "development", # SLM 작업 관리자 상태
            "turn_processing": "development", # 턴 처리 서비스 상태
            "front_backend_api": "not_connected" # 앞단 백엔드 API 연결 상태 (예시)
        },
        "timestamp": time.time() # 현재 시간 타임스탬프
    }


# 서비스 상세 상태 정보 엔드포인트
@router.get(
    "/status", # API 경로: /v2/status
    summary="HEMA v2 상세 상태",
    description="HEMA v2 서비스의 상세 상태 정보를 반환합니다."
)
async def get_status():
    """HEMA v2 상세 상태 조회"""
    # 서비스의 좀 더 상세한 구성 및 기능 구현 상태를 반환
    return {
        "service_name": "HEMA v2 API",
        "version": "2.0.0-dev", # 개발 버전 정보
        "environment": "development", # 현재 실행 환경 (설정값 사용 가능)
        "features": { # 주요 기능들의 구현 상태
            "hema_context_processing": "implemented",
            "slm_task_management": "implemented",
            "turn_processing": "implemented",
            "worker_integration": "implemented",
            "api_endpoints": "implemented"
        },
        "configuration": { # 주요 설정값들 요약 (민감 정보 제외)
            "max_prompt_tokens": settings.MAX_PROMPT_TOKENS,
            "hema_context_budget": settings.HEMA_CONTEXT_TOKEN_BUDGET,
            "slm_response_timeout": settings.SLM_RESPONSE_TIMEOUT,
            "worker_concurrent_requests": settings.WORKER_MAX_CONCURRENT_VLLM_REQUESTS
        },
        "timestamp": time.time()
    }


# 성능 메트릭 조회 엔드포인트
@router.get(
    "/metrics", # API 경로: /v2/metrics
    summary="성능 메트릭 조회",
    description="HEMA v2 서비스의 성능 메트릭을 조회합니다."
)
async def get_metrics():
    """성능 메트릭 조회"""
    try:
        # (가상) performance_monitor 및 error_handler 모듈로부터 메트릭 데이터 수집
        # 이 모듈들은 실제로는 Prometheus 클라이언트 라이브러리 연동, 내부 카운터 등으로 구현될 수 있음
        recent_metrics = performance_monitor.get_recent_metrics() # 최근 요청 처리 성능
        vllm_metrics = performance_monitor.get_vllm_metrics() # vLLM 관련 메트릭
        redis_metrics = performance_monitor.get_redis_metrics() # Redis 관련 메트릭
        error_stats = error_handler.get_error_stats() # 오류 통계

        return {
            "status": "success",
            "timestamp": time.time(),
            "recent_performance": recent_metrics,
            "vllm_metrics": vllm_metrics,
            "redis_metrics": redis_metrics,
            "error_statistics": error_stats,
            "global_metrics": { # (가상) 전역 집계 메트릭
                "total_requests": performance_monitor.global_metrics.request_count,
                "avg_processing_time": performance_monitor.global_metrics.avg_processing_time,
                "error_rate": performance_monitor.global_metrics.error_rate,
                "timeout_rate": performance_monitor.global_metrics.timeout_rate
            }
        }
    except Exception as e:
        logger.exception(f"Failed to get metrics: {e}") # 메트릭 조회 실패 시 로깅
        return { # 오류 발생 시에도 표준화된 응답 형식 유지
            "status": "error",
            "message": "메트릭 조회 중 오류가 발생했습니다.",
            "timestamp": time.time()
        }


# 메트릭 초기화 엔드포인트 (관리자/개발용)
@router.post(
    "/admin/reset_metrics", # API 경로: /v2/admin/reset_metrics
    summary="메트릭 초기화 (관리자 전용)",
    description="성능 메트릭을 초기화합니다."
)
async def reset_metrics():
    """메트릭 초기화 (개발/테스트용)"""
    # 디버그 모드가 아닐 경우, 접근 거부 (403 Forbidden)
    if not settings.DEBUG_MODE:
        raise HTTPException(status_code=403, detail="Debug mode에서만 사용 가능합니다.")

    try:
        # (가상) 성능 모니터 및 오류 핸들러의 내부 데이터 초기화
        # 실제 구현 시에는 각 모니터링 객체의 reset 메소드 등을 호출
        performance_monitor.global_metrics = type(performance_monitor.global_metrics)() # 객체 재 생성으로 초기화
        performance_monitor.recent_times.clear() # 리스트 비우기
        performance_monitor.recent_errors.clear()
        performance_monitor.recent_timeouts.clear()
        performance_monitor.service_metrics.clear() # 딕셔너리 비우기

        error_handler.error_counts.clear()
        error_handler.last_errors.clear()

        return {
            "status": "success",
            "message": "메트릭이 초기화되었습니다.",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.exception(f"Failed to reset metrics: {e}") # 초기화 실패 시 로깅
        raise HTTPException(status_code=500, detail="메트릭 초기화에 실패했습니다.") # 500 Internal Server Error


# 시스템 정보 조회 엔드포인트 (관리자/개발용)
@router.get(
    "/admin/system_info", # API 경로: /v2/admin/system_info
    summary="시스템 정보 조회 (관리자 전용)",
    description="HEMA v2 시스템의 상세 정보를 조회합니다."
)
async def get_system_info():
    """시스템 정보 조회"""
    # 디버그 모드가 아닐 경우, 접근 거부
    if not settings.DEBUG_MODE:
        raise HTTPException(status_code=403, detail="Debug mode에서만 사용 가능합니다.")

    # 시스템 정보를 얻기 위한 라이브러리 임포트
    import psutil # 시스템 및 프로세스 유틸리티 (CPU, 메모리, 디스크 등)
    import sys # 파이썬 인터프리터 정보
    from datetime import datetime # 현재 시간 정보

    try:
        return {
            "system": { # 호스트 시스템 정보
                "python_version": sys.version, # 파이썬 버전
                "platform": sys.platform, # OS 플랫폼
                "cpu_usage": psutil.cpu_percent(interval=1), # CPU 사용률 (1초간 측정)
                "memory_usage": { # 메모리 사용량
                    "total": psutil.virtual_memory().total, # 전체 물리 메모리
                    "available": psutil.virtual_memory().available, # 사용 가능한 메모리
                    "percent": psutil.virtual_memory().percent # 사용률
                },
                "disk_usage": { # 디스크 사용량 (루트 파티션 기준)
                    "total": psutil.disk_usage('/').total, # 전체 디스크 공간
                    "free": psutil.disk_usage('/').free, # 사용 가능한 공간
                    "percent": psutil.disk_usage('/').percent # 사용률
                }
            },
            "application": { # 애플리케이션 관련 정보
                "version": "2.0.0-dev", # 애플리케이션 버전
                "start_time": datetime.now().isoformat(), # 애플리케이션 (또는 현재 요청 처리) 시작 시간 (실제로는 앱 시작 시간을 저장해둬야 함)
                "debug_mode": settings.DEBUG_MODE, # 디버그 모드 활성화 여부
                "configuration": { # 주요 설정값 확인용 (민감 정보 제외)
                    "max_prompt_tokens": settings.MAX_PROMPT_TOKENS,
                    "hema_context_budget": settings.HEMA_CONTEXT_TOKEN_BUDGET,
                    "slm_response_timeout": settings.SLM_RESPONSE_TIMEOUT,
                    "worker_concurrent_requests": settings.WORKER_MAX_CONCURRENT_VLLM_REQUESTS,
                    "front_backend_url": settings.FRONT_BACKEND_API_URL,
                    "redis_host": settings.REDIS_HOST,
                    "redis_port": settings.REDIS_PORT
                }
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.exception(f"Failed to get system info: {e}") # 정보 조회 실패 시 로깅
        return {
            "status": "error",
            "message": "시스템 정보 조회 중 오류가 발생했습니다.",
            "timestamp": time.time()
        }