# ai/app/dependencies.py
from typing import Annotated, Any, Dict, TYPE_CHECKING

from fastapi import Depends # Request는 현재 미사용, 필요시 유지
from langgraph.graph import StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.services.backend_client import BackendApiClient
from app.services.database_client import DatabaseClient
from app.services.postgresql_service import PostgreSQLService
from app.services.llm_service import LLMService
from app.services.image_service import ImageService
from app.services.translation_service import TranslationService
from app.services.spam_service import SpamService
from app.services.storage_service import StorageService
from app.services.langsmith_service import LangSmithService
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.config.settings import Settings # Settings 의존성 추가

if TYPE_CHECKING:
    from app.workflows.workflow_controller import WorkflowControllerV3

_shared_state: Dict[str, Any] = {}

def _get_shared_object(key: str) -> Any:
    obj = _shared_state.get(key)
    if obj is None:
        raise RuntimeError(f"Shared object '{key}' not found in _shared_state. Check lifespan setup and key names.")
    return obj

# --- Settings 의존성 ---
def get_settings() -> Settings:
    return _get_shared_object('settings') # lifespan에서 settings도 저장한다고 가정

# --- 서비스 및 도구 의존성 ---
def get_compiled_workflow_app() -> StateGraph:
    return _get_shared_object('compiled_app')

def get_db_client() -> DatabaseClient:
    return _get_shared_object('db_client')

def get_postgresql_service() -> PostgreSQLService:
    return _get_shared_object('postgresql_service')

def get_llm_service() -> LLMService:
    return _get_shared_object('llm_service')

def get_Google_Search_tool() -> GoogleSearchTool: # 함수명 변경 및 키 변경
    return _get_shared_object('Google Search_tool')

def get_image_service() -> ImageService:
    return _get_shared_object('image_service')

def get_backend_apiclient() -> BackendApiClient:
    return _get_shared_object('backend_apiclient')

def get_translation_service() -> TranslationService:
    return _get_shared_object('translation_service')

def get_spam_service() -> SpamService:
    return _get_shared_object('spam_service')

def get_storage_service() -> StorageService:
    return _get_shared_object('storage_service')

def get_langsmith_service() -> LangSmithService:
    return _get_shared_object('langsmith_service')

def get_checkpointer() -> BaseCheckpointSaver:
    return _get_shared_object('checkpointer')


SettingsDep = Annotated[Settings, Depends(get_settings)]
CompiledWorkflowDep = Annotated[StateGraph, Depends(get_compiled_workflow_app)]
DatabaseClientDep = Annotated[DatabaseClient, Depends(get_db_client)]
PostgreSQLServiceDep = Annotated[PostgreSQLService, Depends(get_postgresql_service)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
GoogleSearchToolDep = Annotated[GoogleSearchTool, Depends(get_Google_Search_tool)] # 변경된 함수 사용
ImageServiceDep = Annotated[ImageService, Depends(get_image_service)]
TranslationServiceDep = Annotated[TranslationService, Depends(get_translation_service)]
SpamServiceDep = Annotated[SpamService, Depends(get_spam_service)]
StorageServiceDep = Annotated[StorageService, Depends(get_storage_service)]
LangSmithServiceDep = Annotated[LangSmithService, Depends(get_langsmith_service)]
CheckpointSaverDep = Annotated[BaseCheckpointSaver, Depends(get_checkpointer)]
BackendApiClientDep = Annotated[BackendApiClient, Depends(get_backend_apiclient)]


# WorkflowControllerV3 의존성 주입 함수
def get_workflow_controller_v3(
    pg_service: PostgreSQLServiceDep, # <<< [수정] PostgreSQLServiceDep 사용
    redis_client: DatabaseClientDep,
    compiled_app: CompiledWorkflowDep,
    checkpointer: CheckpointSaverDep,
    # WorkflowControllerV3 생성자에 필요한 다른 서비스들을 여기에 추가하고,
    # lifespan._shared_state에도 해당 서비스들이 저장되어야 함.
    # 현재 WorkflowControllerV3는 redis_client, compiled_graph, checkpointer만 받음.
    # 만약 llm_service, pg_service 등을 직접 사용한다면 아래 주석 해제 및 lifespan, controller 수정 필요.
    # llm_service: LLMServiceDep,
    # pg_service: PostgreSQLServiceDep,
    # Google Search_tool: GoogleSearchToolDep,
    # settings_obj: SettingsDep,
) -> "WorkflowControllerV3":
    from app.workflows.workflow_controller import WorkflowControllerV3
    return WorkflowControllerV3(
        pg_service=pg_service, # <<< [수정]
        redis_client=redis_client,
        compiled_graph=compiled_app,
        checkpointer=checkpointer
        # llm_service=llm_service, # 필요시 주석 해제
        # pg_service=pg_service,   # 필요시 주석 해제
        # Google Search_tool=Google Search_tool, # 필요시 주석 해제
        # settings_obj=settings_obj # 필요시 주석 해제
    )

WorkflowControllerV3Dep = Annotated["WorkflowControllerV3", Depends(get_workflow_controller_v3)]