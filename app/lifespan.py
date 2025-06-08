# ai/app/lifespan.py
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import AsyncGenerator, Any, List, Optional
from pathlib import Path
import aiohttp
from dotenv import load_dotenv

from app.utils.logger import get_logger
from app.config.settings import Settings

# 워크플로우 컴파일 함수 임포트
from app.workflows.main_workflow import compile_workflow
from app.dependencies import _shared_state

from app.services.database_client import DatabaseClient
from app.services.postgresql_service import PostgreSQLService
from app.services.llm_service import LLMService
from app.services.image_service import ImageService
from app.services.translation_service import TranslationService
from app.services.spam_service import SpamService
from app.services.storage_service import StorageService
from app.services.langsmith_service import LangSmithService
from app.tools.search.Google_Search_tool import GoogleSearchTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

logger_lifespan = get_logger("AppLifespan")

_service_instances_lifespan: List[Any] = []
settings_lifespan = Settings()  # Settings 객체 생성

from transformers import AutoTokenizer

TOKENIZER_LIFESPAN: Optional[AutoTokenizer] = None


async def startup_event():
    """애플리케이션 시작 시 실행될 작업들"""
    global logger_lifespan, TOKENIZER_LIFESPAN, settings_lifespan

    logger_lifespan.info("Application startup process initiated...")

    global _service_instances_lifespan
    _service_instances_lifespan = []
    shared_aiohttp_session = None

    try:
        timeout = aiohttp.ClientTimeout(total=settings_lifespan.EXTERNAL_API_TIMEOUT_SECONDS or 30)
        shared_aiohttp_session = aiohttp.ClientSession(timeout=timeout)
        _service_instances_lifespan.append(shared_aiohttp_session)
        logger_lifespan.info("Shared aiohttp session created.")
    except Exception as e:
        logger_lifespan.error(f"Failed to create shared aiohttp session: {e}", exc_info=True)

    try:
        # --- Settings 객체를 _shared_state에 저장 ---
        _shared_state['settings'] = settings_lifespan  # <<< 이 줄 추가됨
        logger_lifespan.info("Settings object stored in shared state.")

        PROJECT_ROOT = Path(__file__).resolve().parent  # 경로 수정: nodes_v3 -> nodes -> app -> project_root
        # --- 토크나이저 로드 ---
        TOKENIZER_PATH_FROM_SETTINGS = PROJECT_ROOT / "workers" / "handlers" # settings_lifespan.QWEN_TOKENIZER_PATH
        if TOKENIZER_PATH_FROM_SETTINGS:
            qwen_tokenizer_actual_path = Path(TOKENIZER_PATH_FROM_SETTINGS)
            if qwen_tokenizer_actual_path.is_dir():
                try:
                    TOKENIZER_LIFESPAN = AutoTokenizer.from_pretrained(str(qwen_tokenizer_actual_path),
                                                                       trust_remote_code=True)
                    logger_lifespan.info(f"Tokenizer loaded successfully from {qwen_tokenizer_actual_path}.")
                except Exception as e_tok:
                    logger_lifespan.error(f"Tokenizer FAILED to load from {qwen_tokenizer_actual_path}: {e_tok}",
                                          exc_info=True)
                    # 토크나이저 로드 실패 시 LLMService 초기화가 문제될 수 있으므로, 여기서 애플리케이션 중단 고려 가능
                    # raise RuntimeError(f"Critical error: Tokenizer failed to load from {qwen_tokenizer_actual_path}") from e_tok
            else:
                logger_lifespan.error(
                    f"Tokenizer directory NOT found at {qwen_tokenizer_actual_path} (from settings). Please check the path.")
        else:
            logger_lifespan.warning(
                "QWEN_TOKENIZER_PATH not set in settings. Tokenizer not loaded. LLMService might fail.")
        _shared_state['tokenizer'] = TOKENIZER_LIFESPAN  # 로드된 토크나이저(또는 None)도 공유 상태에 저장

        # --- 서비스 초기화 (순서 중요할 수 있음) ---
        db_client = DatabaseClient()
        _shared_state['db_client'] = db_client
        _service_instances_lifespan.append(db_client)
        logger_lifespan.info("DatabaseClient (Redis) initialized.")

        pg_service = PostgreSQLService()
        await pg_service.connect()
        _shared_state['postgresql_service'] = pg_service
        _service_instances_lifespan.append(pg_service)
        logger_lifespan.info("PostgreSQLService initialized and connected.")

        # LLMService 초기화 시 토크나이저 전달
        if TOKENIZER_LIFESPAN is None:
            # 토크나이저 로드 실패에 대한 더 강력한 처리 (예: 애플리케이션 시작 중단)
            logger_lifespan.critical(
                "Tokenizer is None, LLMService cannot be initialized. Application startup will fail or be unstable.")
            # raise RuntimeError("LLMService cannot be initialized without a valid tokenizer.") # 여기서 중단하는 것이 안전할 수 있음
        llm_service = LLMService(tokenizer=TOKENIZER_LIFESPAN)  # settings는 내부적으로 사용
        _shared_state['llm_service'] = llm_service
        if hasattr(llm_service, 'close') and callable(getattr(llm_service, 'close')):
            _service_instances_lifespan.append(llm_service)
        logger_lifespan.info("LLMService initialized.")

        Google_Search_tool = GoogleSearchTool()
        _shared_state['Google Search_tool'] = Google_Search_tool
        _service_instances_lifespan.append(Google_Search_tool)
        logger_lifespan.info("GoogleSearchTool initialized.")

        image_service = ImageService()
        _shared_state['image_service'] = image_service
        _service_instances_lifespan.append(image_service)
        logger_lifespan.info("ImageService initialized.")

        translation_service = TranslationService()
        _shared_state['translation_service'] = translation_service
        logger_lifespan.info("TranslationService initialized.")

        spam_service = SpamService()
        _shared_state['spam_service'] = spam_service
        logger_lifespan.info("SpamService initialized.")

        storage_service = StorageService()
        _shared_state['storage_service'] = storage_service
        logger_lifespan.info("StorageService initialized.")

        langsmith_service = LangSmithService()
        _shared_state['langsmith_service'] = langsmith_service
        if hasattr(langsmith_service, 'close') and callable(getattr(langsmith_service, 'close')):
            _service_instances_lifespan.append(langsmith_service)
        logger_lifespan.info("LangSmithService initialized.")

        logger_lifespan.info("Compiling LangGraph v3 workflow...")
        checkpointer: BaseCheckpointSaver = MemorySaver()
        _shared_state['checkpointer'] = checkpointer

        compiled_app_instance = await compile_workflow(
            redis_client=db_client,
            llm_service=llm_service,
            pg_service=pg_service,
            google_search_tool_instance = Google_Search_tool,
            settings_obj = settings_lifespan,
            checkpointer = checkpointer
        )
        _shared_state['compiled_app'] = compiled_app_instance
        logger_lifespan.info("LangGraph v3 workflow compiled and stored in shared state.")

    except Exception as e:
        logger_lifespan.error(f"Error during service/workflow initialization: {e}", exc_info=True)
        await shutdown_event(graceful=False)
        raise RuntimeError(f"Application startup failed: {e}") from e

    logger_lifespan.info("Application startup complete.")


async def shutdown_event(graceful: bool = True):
    global logger_lifespan
    try:
        logger_lifespan = get_logger("AppLifespan")
    except Exception:
        import logging
        logger_lifespan = logging.getLogger("AppLifespanFallbackShutdown")

    if graceful:
        logger_lifespan.info("Application shutdown process initiated...")
    else:
        logger_lifespan.warning("Attempting partial cleanup after startup failure...")

    global _service_instances_lifespan
    for instance in reversed(_service_instances_lifespan):
        instance_name = type(instance).__name__
        close_method = None
        if hasattr(instance, 'close') and callable(getattr(instance, 'close')):
            close_method = getattr(instance, 'close')
        elif hasattr(instance, 'disconnect') and callable(getattr(instance, 'disconnect')):
            close_method = getattr(instance, 'disconnect')

        if close_method:
            try:
                import inspect
                if inspect.iscoroutinefunction(close_method):
                    logger_lifespan.debug(f"Closing {instance_name} (async)...")
                    await close_method()
                else:
                    logger_lifespan.debug(f"Closing {instance_name} (sync)...")
                    await asyncio.to_thread(close_method)
                logger_lifespan.info(f"{instance_name} closed/disconnected successfully.")
            except Exception as e:
                logger_lifespan.error(f"Error during {instance_name} shutdown: {e}", exc_info=True)

    if graceful:
        _shared_state.clear()
        logger_lifespan.info("Shared state cleared.")
        logger_lifespan.info("Application shutdown complete.")
    else:
        logger_lifespan.warning("Partial cleanup finished.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan 컨텍스트 관리자"""
    env_path = Path(".env")  # .env 파일 경로
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, verbose=True)
        logger_lifespan.info(f".env file loaded from: {env_path.resolve()}")
    else:
        logger_lifespan.info(".env file not found, relying on environment variables or default settings.")

    await startup_event()
    try:
        yield
    finally:
        await shutdown_event()

