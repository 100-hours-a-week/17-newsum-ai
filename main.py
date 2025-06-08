# ai/main.py

import uvicorn
from fastapi import FastAPI

# --- 추후 구현될 애플리케이션 구성 요소 임포트 (예상 경로) ---
# 설정 값 (host, port, app 이름 등)
# from app.config.settings import settings

# API 라우트 정의
# from app.api.v1 import endpoints as v1_endpoints
from app.api.v2 import endpoints as v2_endpoints

# 애플리케이션 시작/종료 시 실행될 로직 (예: 모델 로딩, 워크플로우 컴파일)
from app.lifespan import lifespan

from app.config.settings import Settings # 변경: 중앙 설정 객체 임포트
from app.utils.logger import setup_logging, get_logger

# 로거 설정
logger = get_logger(__name__)
settings = Settings()

# 로깅 설정 함수 (lifespan 등에서 호출될 수 있음)
# from app.utils.logging_config import setup_logging, get_logger

# --- 로깅 설정 (실제 설정은 lifespan 또는 별도 config에서 진행) ---
setup_logging() # 예시: 로깅 설정 함수 호출
# logger = get_logger(__name__) # 예시: 로거 가져오기

# --- FastAPI 애플리케이션 인스턴스 생성 ---
# title, description, version 등은 추후 settings에서 로드
# lifespan 매개변수를 통해 앱 시작 시 워크플로우 컴파일 등을 수행
app = FastAPI(
    title="NewSum AI Service", # 임시 값, 추후 settings.APP_NAME 등으로 대체
    description="뉴스/의견 기반 만화 생성을 위한 LangGraph 워크플로우 API", # 임시 값
    version="0.1.0", # 임시 값, 추후 settings.APP_VERSION 등으로 대체
    lifespan=lifespan, # 애플리케이션 수명 주기 이벤트 처리기 등록
    docs_url="/swagger",        # 기본 /docs → /swagger
    redoc_url=None,             # ReDoc 비활성화
    openapi_url="/schema.json", # 기본 /openapi.json → /schema.json
)

# --- API 라우터 등록 ---
# '/api/v1' 경로로 들어오는 요청을 v1_endpoints.router 가 처리하도록 설정
# app.include_router(v1_endpoints.router, prefix="/api/v1")
# '/api/v2' 경로로 들어오는 요청을 v2_endpoints.router 가 처리하도록 설정  
app.include_router(v2_endpoints.router, prefix="/api/v2")

# --- Uvicorn 서버 실행 (로컬 개발 환경용) ---
# 이 스크립트가 직접 실행될 때만 동작
if __name__ == "__main__":
    # logger.info("로컬 개발 서버 (Uvicorn) 시작...") # 예시: 로깅
    # host, port, log_level, reload 등은 추후 settings 객체에서 읽어오도록 수정
    # 워크플로우 테스트용 디버그 출력 임포트
    try:
        from app.utils.debug import print_final_state_debug
    except ImportError:
        print_final_state_debug = None

    # 아래는 실제 워크플로우 실행 예시 (state/result_dict는 실제 워크플로우 실행 결과에 맞게 전달)
    # 예시:
    # state, result_dict = run_full_workflow(...)
    # if print_final_state_debug:
    #     print_final_state_debug(state, result_dict)
    width = 50
    line = f"+{'-' * (width - 2)}+"

    # f-string을 사용하여 여러 줄의 로그 메시지 생성
    startup_message = f"""
    {line}
    | {'Application Settings Initialized':^{width - 4}} |
    {line}
    | {'App Name:':<15} {str(settings.APP_NAME):<{width - 22}} |
    | {'App Version:':<15} {str(settings.APP_VERSION):<{width - 22}} |
    | {'Host:':<15} {str(settings.APP_HOST):<{width - 22}} |
    | {'Port:':<15} {str(settings.APP_PORT):<{width - 22}} |
    | {'Log Level:':<15} {str(settings.LOG_LEVEL):<{width - 22}} |
    {line}
    """

    # 생성된 메시지를 logger.info로 출력
    logger.info(startup_message)
    uvicorn.run(
        "main:app",      # 실행할 FastAPI 앱 (main.py 파일의 app 객체)
        host=settings.APP_HOST,   # 외부 접속 허용 (예: settings.APP_HOST)
        port=settings.APP_PORT,        # 사용할 포트 (예: settings.APP_PORT)
        log_level=settings.LOG_LEVEL, # 로그 레벨 (예: settings.LOG_LEVEL)
        #reload=True       # 코드 변경 시 자동 재시작 (개발 시 유용, 예: settings.APP_RELOAD)
    )
