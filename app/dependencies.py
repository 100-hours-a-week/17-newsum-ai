# ai/app/dependencies.py
from typing import Annotated, Any, Dict
from fastapi import Depends, HTTPException, Request
from langgraph.graph import StateGraph # 또는 CompiledGraph

# --- 서비스 및 도구 클래스 임포트 ---
from app.services.database_client import DatabaseClient
from app.services.llm_service import LLMService
from app.services.image_service import ImageService
from app.services.translation_service import TranslationService
from app.services.spam_service import SpamService
from app.services.storage_service import StorageService
from app.services.langsmith_service import LangSmithService
from app.tools.search.Google_Search_tool import GoogleSearchTool
# TODO: Naver, Reddit 등 다른 검색 도구 추가 시 임포트

# --- HEMA v2 서비스 임포트 ---
from app.services.hema_service import HEMAService
from app.services.turn_processing_service import TurnProcessingService
from app.services.slm_task_manager import SLMTaskManager
from app.clients.front_backend_api_client import FrontBackendAPIClient

_shared_state: Dict[str, Any] = {} # lifespan에서 관리할 공유 객체 저장소

def _get_shared_object(key: str) -> Any:
    """공유 상태에서 객체를 가져오는 내부 함수"""
    obj = _shared_state.get(key)
    if obj is None:
        raise RuntimeError(f"Shared object '{key}' not found. Check lifespan setup.")
    return obj

# --- 의존성 주입 함수 정의 ---

def get_compiled_workflow_app() -> StateGraph:
    """컴파일된 LangGraph 워크플로우 인스턴스 반환"""
    return _get_shared_object('compiled_app')

def get_db_client() -> DatabaseClient: # <<< 실제 타입 사용
    """데이터베이스 클라이언트 인스턴스 반환"""
    return _get_shared_object('db_client')

def get_llm_service() -> LLMService:
    """LLM 서비스 인스턴스 반환"""
    return _get_shared_object('llm_service')

def get_image_service() -> ImageService:
    """이미지 생성 서비스 인스턴스 반환"""
    return _get_shared_object('image_service')

def get_translation_service() -> TranslationService:
    """번역 서비스 인스턴스 반환"""
    return _get_shared_object('translation_service')

def get_spam_service() -> SpamService:
    """스팸 탐지 서비스 인스턴스 반환"""
    return _get_shared_object('spam_service')

def get_storage_service() -> StorageService:
    """스토리지 서비스 인스턴스 반환"""
    return _get_shared_object('storage_service')

def get_langsmith_service() -> LangSmithService:
    """LangSmith 서비스 인스턴스 반환"""
    return _get_shared_object('langsmith_service')

def get_google_search_tool() -> GoogleSearchTool:
    """Google 검색 도구 인스턴스 반환"""
    return _get_shared_object('Google Search_tool')

# --- HEMA v2 의존성 주입 함수 ---
def get_hema_service() -> HEMAService:
    """HEMA 서비스 인스턴스 반환"""
    return _get_shared_object('hema_service')

def get_turn_processing_service() -> TurnProcessingService:
    """턴 처리 서비스 인스턴스 반환"""
    return _get_shared_object('turn_processing_service')

def get_slm_task_manager() -> SLMTaskManager:
    """SLM 작업 관리자 인스턴스 반환"""
    return _get_shared_object('slm_task_manager')

def get_front_backend_client() -> FrontBackendAPIClient:
    """앞단 백엔드 API 클라이언트 인스턴스 반환"""
    return _get_shared_object('front_backend_client')

# --- 타입 어노테이션 기반 의존성 ---
CompiledWorkflowDep = Annotated[StateGraph, Depends(get_compiled_workflow_app)]
DatabaseClientDep = Annotated[DatabaseClient, Depends(get_db_client)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
ImageServiceDep = Annotated[ImageService, Depends(get_image_service)]
TranslationServiceDep = Annotated[TranslationService, Depends(get_translation_service)]
SpamServiceDep = Annotated[SpamService, Depends(get_spam_service)]
StorageServiceDep = Annotated[StorageService, Depends(get_storage_service)]
LangSmithServiceDep = Annotated[LangSmithService, Depends(get_langsmith_service)]
GoogleSearchToolDep = Annotated[GoogleSearchTool, Depends(get_google_search_tool)]

# --- HEMA v2 타입 어노테이션 기반 의존성 ---
HEMAServiceDep = Annotated[HEMAService, Depends(get_hema_service)]
TurnProcessingServiceDep = Annotated[TurnProcessingService, Depends(get_turn_processing_service)]
SLMTaskManagerDep = Annotated[SLMTaskManager, Depends(get_slm_task_manager)]
FrontBackendClientDep = Annotated[FrontBackendAPIClient, Depends(get_front_backend_client)]