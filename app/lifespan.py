# ai/app/lifespan.py
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import AsyncGenerator
from pathlib import Path
import aiohttp # 외부 API 호출용 세션 생성 위해 추가
from dotenv import load_dotenv

# --- 로깅 및 설정 임포트 ---
from app.utils.logger import setup_logging, get_logger
from app.config.settings import Settings

# --- 워크플로우 및 공유 상태 ---
from app.workflows.main_workflow import compile_workflow
from app.dependencies import _shared_state # 공유 상태 딕셔너리

# --- 서비스 및 도구 클래스 임포트 ---
from app.services.database_client import DatabaseClient # <<< DatabaseClient 임포트
from app.services.postgresql_service import PostgreSQLService
from app.services.llm_service import LLMService
from app.services.image_service import ImageService
from app.services.translation_service import TranslationService
from app.services.spam_service import SpamService
from app.services.storage_service import StorageService
from app.services.langsmith_service import LangSmithService
from app.tools.search.Google_Search_tool import GoogleSearchTool

logger = get_logger("AppLifespan") # 로깅 설정 후 로거 가져오기

_service_instances = [] # 종료 시 정리할 인스턴스 목록
settings = Settings()

import os
from transformers import AutoTokenizer
TOKENIZER = None
 # __file__은 현재 실행 중인 스크립트의 경로를 나타냄
# chat_worker.py가 있는 디렉토리의 'handlers' 하위 'qwen_tokenizer'를 찾음
current_script_dir = Path(__file__).resolve().parent
print(current_script_dir)
QWEN_TOKENIZER_PATH = current_script_dir / "workers" / "handlers"
if QWEN_TOKENIZER_PATH.is_dir():
    TOKENIZER = AutoTokenizer.from_pretrained(str(QWEN_TOKENIZER_PATH), trust_remote_code=True)
    logger.info(f"ChatWorker: Tokenizer loaded successfully from {QWEN_TOKENIZER_PATH}.")
else:
    logger.error(f"ChatWorker: Tokenizer directory NOT found at {QWEN_TOKENIZER_PATH}. Please check the path.")

async def startup_event():
    """애플리케이션 시작 시 실행될 작업들"""
    # --- 로깅 설정 적용 (가장 먼저 수행) ---
    try:
        log_config_path_str = settings.LOG_CONFIG_PATH or 'logging_config.yaml'
        config_path_obj = Path(log_config_path_str)
        setup_logging(config_path=config_path_obj)
        global logger
        logger = get_logger("AppLifespan")
        logger.info(f"Logging setup complete using config: {config_path_obj.resolve()}")
    except Exception as e:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("AppLifespanFallback")
        logger.critical(f"!!! FATAL ERROR during logging setup: {e}", exc_info=True)
        print(f"!!! FATAL ERROR during logging setup: {e}")
        raise RuntimeError(f"Logging setup failed: {e}") from e

    logger.info("Application startup process initiated...")

    # --- 서비스 및 도구 초기화 ---
    global _service_instances
    _service_instances = []

    # --- 외부 API 호출 및 번역용 공유 aiohttp 세션 (선택적) ---
    shared_aiohttp_session = None # 기본값 None
    try:
        timeout = aiohttp.ClientTimeout(total=settings.EXTERNAL_API_TIMEOUT_SECONDS or 30)
        shared_aiohttp_session = aiohttp.ClientSession(timeout=timeout)
        logger.info("Shared aiohttp session created for external/translation calls.")
        _service_instances.append(shared_aiohttp_session) # 종료 시 닫기 위해 추가
    except Exception as e:
        logger.error(f"Failed to create shared aiohttp session: {e}", exc_info=True)
        # 공유 세션 생성 실패 시에도 일단 진행 (개별 서비스/노드가 자체 관리하도록)


    try:
        # DatabaseClient
        db_client = DatabaseClient() # 설정은 내부적으로 settings 사용 가정
        # <<< 수정: DatabaseClient에 connect 메서드가 없으므로 호출 제거 >>>
        # await db_client.connect() # connect 메서드가 없으므로 제거 또는 실제 초기화 메서드로 변경
        # Redis 클라이언트는 __init__에서 이미 연결 시도/생성됨
        _shared_state['db_client'] = db_client
        _service_instances.append(db_client) # disconnect/close 메서드가 있다고 가정
        logger.info("DatabaseClient initialized.") # 메시지 수정

        # PostgreSQLService
        postgresql_service = PostgreSQLService()
        await postgresql_service.connect()
        _shared_state['postgresql_service'] = postgresql_service
        _service_instances.append(postgresql_service)
        logger.info("PostgreSQLService initialized.")

        # LLMService (close 메서드 유무 확인 필요)
        llm_service = LLMService(tokenizer=TOKENIZER)
        _shared_state['llm_service'] = llm_service
        if hasattr(llm_service, 'close') and callable(getattr(llm_service, 'close')):
             _service_instances.append(llm_service)
        logger.info("LLMService initialized.")

        # ImageService (close 메서드 있음 - httpx client)
        image_service = ImageService()
        _shared_state['image_service'] = image_service
        _service_instances.append(image_service)
        logger.info("ImageService initialized.")

        # TranslationService (close 메서드 없음, 외부 세션 필요 가능성)
        translation_service = TranslationService()
        # 필요 시 생성자 또는 별도 메서드로 공유 세션 주입 가능
        # translation_service.set_session(shared_aiohttp_session)
        _shared_state['translation_service'] = translation_service
        logger.info("TranslationService initialized.")

        # SpamService (close 메서드 없다고 가정)
        spam_service = SpamService()
        _shared_state['spam_service'] = spam_service
        logger.info("SpamService initialized.")

        # StorageService (boto3 사용, 명시적 close 불필요)
        storage_service = StorageService()
        _shared_state['storage_service'] = storage_service
        logger.info("StorageService initialized.")

        # LangSmithService (close 메서드 있다고 가정)
        langsmith_service = LangSmithService()
        _shared_state['langsmith_service'] = langsmith_service
        _service_instances.append(langsmith_service)
        logger.info("LangSmithService initialized.")

        # GoogleSearchTool (close 메서드 있음 - aiohttp session)
        # Google_Search_tool = GoogleSearchTool(session=shared_aiohttp_session) # 공유 세션 사용 시
        Google_Search_tool = GoogleSearchTool() # 내부 세션 사용 시
        _shared_state['Google_Search_tool'] = Google_Search_tool # 키 이름 일관성 유지
        _service_instances.append(Google_Search_tool)
        logger.info("GoogleSearchTool initialized.")


        # --- 워크플로우 컴파일 ---
        logger.info("Compiling LangGraph workflow...")
        compiled_app_instance = await compile_workflow(
            llm_service=llm_service,
            Google_Search_tool=Google_Search_tool,
            image_generation_service=image_service,
            storage_service=storage_service,
            translation_service=translation_service,
            # N10에 공유 세션 전달 시 필요
            # external_api_session=shared_aiohttp_session
        )
        _shared_state['compiled_app'] = compiled_app_instance
        logger.info("LangGraph workflow compiled successfully.")

    except Exception as e:
        logger.error(f"Error during service/workflow initialization: {e}", exc_info=True)
        await shutdown_event(graceful=False)
        raise RuntimeError(f"Application startup failed during service initialization: {e}") from e

    logger.info("Application startup complete. Ready to accept requests.")

async def shutdown_event(graceful: bool = True):
    """애플리케이션 종료 시 실행될 작업들"""
    try:
        logger = get_logger("AppLifespan")
    except Exception:
        import logging
        logger = logging.getLogger("AppLifespanFallback")

    if graceful:
        logger.info("Application shutdown process initiated...")
    else:
        logger.warning("Attempting partial cleanup after startup failure...")

    global _service_instances
    for instance in reversed(_service_instances):
        instance_name = type(instance).__name__
        # close 또는 disconnect 메서드 찾기
        # aiohttp.ClientSession도 close() 메서드를 가짐
        close_method = getattr(instance, 'close', None) or getattr(instance, 'disconnect', None)

        if close_method and callable(close_method):
            try:
                import inspect
                if inspect.iscoroutinefunction(close_method):
                    logger.debug(f"Closing {instance_name} (async)...")
                    await close_method()
                else:
                    logger.debug(f"Closing {instance_name} (sync)...")
                    # 동기 메서드는 이벤트 루프에서 직접 호출 시 블로킹될 수 있으므로 주의
                    await asyncio.to_thread(close_method) # 비동기로 실행 시도
                logger.info(f"{instance_name} closed/disconnected successfully.")
            except Exception as e:
                logger.error(f"Error during {instance_name} shutdown: {e}", exc_info=True)

    # 공유 상태 정리
    if graceful:
        _shared_state.clear()
        logger.info("Shared state cleared.")
        logger.info("Application shutdown complete.")
    else:
        logger.warning("Partial cleanup finished.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan 컨텍스트 관리자"""
    load_dotenv()
    await startup_event()
    try:
        yield
    finally:
        await shutdown_event()