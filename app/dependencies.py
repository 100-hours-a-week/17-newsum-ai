# app/dependencies.py

from fastapi import Request, HTTPException, status

# 필요한 클래스 임포트 (경로 주의)
from app.services.database_con_client_v2 import DatabaseClientV2
from langgraph.graph import StateGraph # StateGraph 타입 힌트용
from app.utils.logger import get_logger

logger = get_logger(__name__)

# --- 컴파일된 워크플로우 앱 의존성 ---

def get_compiled_workflow_app(request: Request) -> StateGraph:
    """
    FastAPI 앱 상태(`app.state`)에서 미리 컴파일된 LangGraph 앱 인스턴스를 가져옵니다.
    앱 시작 시 `lifespan` 컨텍스트 매니저를 통해 초기화되어야 합니다.

    Args:
        request (Request): FastAPI 요청 객체 (app.state 접근용).

    Returns:
        StateGraph: 컴파일된 LangGraph 애플리케이션.

    Raises:
        HTTPException: 앱 상태에 워크플로우 앱이 설정되지 않은 경우 503 오류 발생.
    """
    if not hasattr(request.app.state, 'compiled_workflow_app') or request.app.state.compiled_workflow_app is None:
        logger.critical("컴파일된 워크플로우 앱(compiled_workflow_app)이 app.state에 설정되지 않았습니다! 앱 시작 시 초기화 오류 가능성.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="워크플로우 서비스를 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요."
        )
    return request.app.state.compiled_workflow_app

# FastAPI 라우터에서 타입 힌트와 함께 사용하기 위한 Annotated 버전 (선택 사항)
# CompiledAppDep = Annotated[StateGraph, Depends(get_compiled_workflow_app)]


# --- 데이터베이스 클라이언트 의존성 ---

def get_db_client(request: Request) -> DatabaseClientV2:
    """
    FastAPI 앱 상태(`app.state`)에서 초기화된 데이터베이스 클라이언트 인스턴스를 가져옵니다.
    앱 시작 시 `lifespan` 컨텍스트 매니저를 통해 초기화되어야 합니다.

    Args:
        request (Request): FastAPI 요청 객체 (app.state 접근용).

    Returns:
        DatabaseClientV2: 초기화된 데이터베이스 클라이언트 인스턴스.

    Raises:
        HTTPException: 앱 상태에 DB 클라이언트가 설정되지 않은 경우 503 오류 발생.
    """
    if not hasattr(request.app.state, 'db_client') or request.app.state.db_client is None:
        logger.critical("데이터베이스 클라이언트(db_client)가 app.state에 설정되지 않았습니다! 앱 시작 시 초기화 오류 가능성.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="데이터베이스 서비스를 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요."
        )
    return request.app.state.db_client

# FastAPI 라우터에서 타입 힌트와 함께 사용하기 위한 Annotated 버전 (선택 사항)
# DBClientDep = Annotated[DatabaseClientV2, Depends(get_db_client)]

# --- (선택 사항) 기타 공유 서비스/도구 의존성 ---
# 만약 LLMService, ArticleScraperTool 등 다른 객체들도 앱 전체에서 공유하고 싶다면,
# 위와 유사한 방식으로 lifespan에서 초기화하고 의존성 주입 함수를 만들 수 있습니다.
# 예시:
# def get_llm_service(request: Request) -> LLMService:
#     # ... app.state.llm_service 확인 및 반환 ...
# LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]