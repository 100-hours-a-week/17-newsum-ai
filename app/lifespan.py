# ai/app/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import AsyncGenerator

# --- 로깅 및 설정 임포트 ---
from app.utils.logger import setup_logging, get_logger # setup_logging 임포트
from app.config.settings import Settings

from app.workflows.main_workflow import compile_workflow
from app.dependencies import _shared_state

# --- 서비스 및 도구 클래스 임포트 ---
from app.services.database_client import DatabaseClient
from app.services.llm_service import LLMService
from app.services.image_service import ImageService
from app.services.translation_service import TranslationService
from app.services.spam_service import SpamService
from app.services.storage_service import StorageService
from app.services.langsmith_service import LangSmithService
from app.tools.search.Google_Search_tool import GoogleSearchTool

# lifespan 시작 전에 get_logger 호출 시 기본 설정만 적용될 수 있으므로 주의
# logger = get_logger("AppLifespan") # 여기서 호출하면 setup_logging 전일 수 있음

_service_instances = []
settings = Settings()

async def startup_event():
    """애플리케이션 시작 시 실행될 작업들"""
    # --- 로깅 설정 적용 (가장 먼저 수행) ---
    try:
        # settings.LOG_CONFIG_PATH가 None일 경우 기본 경로 사용
        log_config_path = settings.LOG_CONFIG_PATH or 'logging_config.yaml'
        # Path 객체로 변환하여 setup_logging에 전달
        from pathlib import Path
        config_path_obj = Path(log_config_path)
        # setup_logging 함수는 절대 경로 또는 상대 경로를 처리할 수 있어야 함
        # 여기서는 logger.py의 기본 경로 설정을 신뢰
        setup_logging(config_path=config_path_obj)
        # setup_logging 호출 후 로거 가져오기
        global logger # 전역 변수 사용 대신 함수 내에서 로거 가져오기
        logger = get_logger("AppLifespan")
        logger.info(f"로깅 설정 완료됨.") # 설정 완료 후 첫 로그

    except Exception as e:
        # 로깅 설정 실패 시에도 기본 로깅으로 메시지 출력 시도
        import logging
        logging.basicConfig()
        logging.critical(f"!!! 로깅 설정 중 치명적 오류 발생: {e}", exc_info=True)
        # 로깅이 안될 수 있으므로 print도 사용
        print(f"!!! FATAL ERROR during logging setup: {e}")
        # 로깅 설정 실패 시 앱 시작을 중단할 수 있음
        raise RuntimeError(f"Logging setup failed: {e}") from e

    logger.info("애플리케이션 시작 프로세스 개시...")

    # --- 서비스 및 도구 초기화 ---
    global _service_instances
    _service_instances = [] # 초기화

    try:
        # DatabaseClient
        db_client = DatabaseClient()
        _shared_state['db_client'] = db_client
        _service_instances.append(db_client)
        logger.info("DatabaseClient 초기화 완료.")

        # LLMService
        llm_service = LLMService()
        _shared_state['llm_service'] = llm_service
        _service_instances.append(llm_service)
        logger.info("LLMService 초기화 완료.")

        # ImageService
        image_service = ImageService()
        _shared_state['image_service'] = image_service
        _service_instances.append(image_service)
        logger.info("ImageService 초기화 완료.")

        # TranslationService
        translation_service = TranslationService()
        _shared_state['translation_service'] = translation_service
        # _service_instances.append(translation_service) # close 메서드 없으면 추가 불필요
        logger.info("TranslationService 초기화 완료.")

        # SpamService
        spam_service = SpamService()
        _shared_state['spam_service'] = spam_service
        # _service_instances.append(spam_service) # close 메서드 없으면 추가 불필요
        logger.info("SpamService 초기화 완료.")

        # StorageService
        storage_service = StorageService()
        _shared_state['storage_service'] = storage_service
        # _service_instances.append(storage_service) # close 메서드 없으면 추가 불필요
        logger.info("StorageService 초기화 완료.")

        # LangSmithService
        langsmith_service = LangSmithService()
        _shared_state['langsmith_service'] = langsmith_service
        _service_instances.append(langsmith_service) # close 메서드 있음
        logger.info("LangSmithService 초기화 완료.")

        # GoogleSearchTool
        google_search_tool = GoogleSearchTool()
        _shared_state['Google_Search_tool'] = google_search_tool
        _service_instances.append(google_search_tool) # close 메서드 있음
        logger.info("GoogleSearchTool 초기화 완료.")

        # --- 워크플로우 컴파일 ---
        logger.info("LangGraph 워크플로우 컴파일 시작...")
        compiled_app_instance = await compile_workflow(
            llm_service=llm_service,  # <<< llm_service 인스턴스 전달
            Google_Search_tool = google_search_tool  # <<< Google Search_tool 인스턴스 전달
        )
        _shared_state['compiled_app'] = compiled_app_instance
        logger.info("LangGraph 워크플로우 컴파일 완료.")

    except Exception as e:
        logger.error(f"서비스/워크플로우 초기화 중 오류 발생: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed during service initialization: {e}") from e

    logger.info("애플리케이션 시작 준비 완료.")

async def shutdown_event():
    """애플리케이션 종료 시 실행될 작업들"""
    logger = get_logger("AppLifespan") # 종료 시에도 로거 가져오기
    logger.info("애플리케이션 종료 프로세스 개시...")

    global _service_instances
    for instance in reversed(_service_instances):
        instance_name = type(instance).__name__
        if hasattr(instance, 'close') and callable(instance.close):
            try:
                # 비동기 close 메서드 확인
                import inspect
                if inspect.iscoroutinefunction(instance.close):
                    await instance.close()
                else:
                    instance.close() # 동기 메서드일 경우
                logger.info(f"{instance_name} 종료 완료.")
            except Exception as e:
                logger.error(f"{instance_name} 종료 중 오류 발생: {e}", exc_info=True)

    # 공유 상태 정리
    _shared_state.clear()
    logger.info("공유 상태 정리 완료.")
    logger.info("애플리케이션 종료 완료.")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan 컨텍스트 관리자"""
    await startup_event()
    yield
    await shutdown_event()