# ai/app/clients/front_backend_api_client.py

# 비동기 HTTP 요청을 위한 httpx 라이브러리
import httpx
# 비동기 프로그래밍을 위한 asyncio (여기서는 직접 사용되지 않지만, httpx 내부에서 활용됨)
import asyncio
# 타입 힌팅을 위한 모듈들
from typing import Dict, Any, List, Optional
# 날짜/시간 처리를 위한 datetime (현재 코드에서는 직접 사용되지 않음)
from datetime import datetime
# JSON 처리를 위한 json (현재 코드에서는 직접 사용되지 않지만, httpx가 내부적으로 사용)
import json

# 프로젝트 설정값 가져오기 (예: API 기본 URL, 타임아웃, API 키)
from app.config.settings import settings
# 로깅 유틸리티 가져오기
from app.utils.logger import get_logger
# HEMA 데이터 관련 Pydantic 스키마들 임포트
from app.api.v2.schemas.hema_models import (
    HEMABulkOperationRequest,  # HEMA 데이터 일괄 처리 요청 스키마
    HEMABulkOperationResponse, # HEMA 데이터 일괄 처리 응답 스키마
    InformationSnippetSchema,  # 정보 조각 데이터 스키마
    IdeaNodeSchema,            # 아이디어 노드 데이터 스키마
    SummaryNodeSchema,         # 요약 노드 데이터 스키마
    HEMAInternalInteractionLogSchema # 내부 상호작용 로그 스키마
)

# 로거 인스턴스 생성
logger = get_logger(__name__)


class FrontBackendAPIClient:
    """
    "앞단 백엔드 서버"와 통신하여 HEMA 데이터를 관리하는 비동기 API 클라이언트.
    HTTP 요청을 위해 httpx 라이브러리를 사용합니다.
    """

    def __init__(self):
        """
        클라이언트 초기화.
        설정에서 API 기본 URL, 타임아웃, API 키를 가져옵니다.
        httpx.AsyncClient 인스턴스는 첫 호출 시 지연 초기화됩니다.
        """
        # 설정에서 API 기본 URL을 가져오고, 끝에 있을 수 있는 '/' 제거
        self.base_url = settings.FRONT_BACKEND_API_URL.rstrip('/')
        # 설정에서 API 요청 타임아웃 값 가져오기
        self.timeout = settings.FRONT_BACKEND_API_TIMEOUT
        # 설정에서 API 키 가져오기 (인증용)
        self.api_key = settings.FRONT_BACKEND_API_KEY
        # httpx.AsyncClient 인스턴스를 저장할 변수, 초기에는 None
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        비동기 HTTP 클라이언트 인스턴스를 반환합니다.
        클라이언트가 아직 생성되지 않았다면 새로 생성하고, 이미 있다면 기존 인스턴스를 재사용합니다.
        (Lazy Initialization 패턴)
        """
        if self._client is None:
            # 기본 HTTP 헤더 설정
            headers = {"Content-Type": "application/json"}
            # API 키가 설정되어 있다면, Authorization 헤더에 Bearer 토큰으로 추가
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # httpx.AsyncClient 인스턴스 생성
            self._client = httpx.AsyncClient(
                base_url=self.base_url, # API 기본 URL 설정
                timeout=self.timeout,   # 요청 타임아웃 설정
                headers=headers         # 공통 헤더 설정
            )
            logger.info(f"FrontBackendAPIClient: httpx.AsyncClient initialized for {self.base_url}")
        return self._client

    async def close(self):
        """
        사용 중인 httpx.AsyncClient 연결을 비동기적으로 종료합니다.
        애플리케이션 종료 시 호출되어 리소스를 정리합니다.
        """
        if self._client:
            logger.info(f"FrontBackendAPIClient: Closing httpx.AsyncClient for {self.base_url}")
            await self._client.aclose() # 비동기 클라이언트 종료
            self._client = None # 클라이언트 참조 제거
            logger.info(f"FrontBackendAPIClient: httpx.AsyncClient closed.")

    # --- HEMA 데이터 Bulk API ---

    async def bulk_operations(
        self,
        request: HEMABulkOperationRequest # 요청 본문은 HEMABulkOperationRequest 스키마를 따름
    ) -> HEMABulkOperationResponse:
        """
        HEMA 데이터 일괄 처리 API (/api/external/hema/bulk_operations)를 호출합니다.
        여러 HEMA 데이터 생성/수정/삭제 작업을 한 번의 요청으로 처리합니다.

        Args:
            request: 일괄 처리할 작업 목록을 담은 요청 객체.

        Returns:
            일괄 처리 결과를 담은 응답 객체.

        Raises:
            httpx.HTTPStatusError: API가 오류 상태 코드를 반환한 경우.
            Exception: 기타 통신 또는 처리 중 예외 발생 시.
        """
        try:
            # 비동기 HTTP 클라이언트 가져오기
            client = await self._get_client()
            logger.info(f"Calling bulk_operations API. Operations count: {len(request.operations)}")

            # POST 요청 전송. 요청 본문은 request 객체를 딕셔너리로 변환하여 JSON으로 전송.
            response = await client.post(
                "/api/external/hema/bulk_operations", # 대상 API 엔드포인트 경로
                json=request.dict() # Pydantic 모델을 딕셔너리로 변환
            )
            # HTTP 오류 상태 코드(4xx, 5xx) 발생 시 예외 발생
            response.raise_for_status()

            # 응답 JSON 데이터를 파싱
            result_data = response.json()
            logger.info(f"Bulk operations API call successful. Response status: {result_data.get('status')}")
            # 파싱된 데이터를 HEMABulkOperationResponse 스키마로 변환하여 반환
            return HEMABulkOperationResponse(**result_data)

        except httpx.HTTPStatusError as e:
            # HTTP 상태 오류 로깅 (응답 코드 및 내용 포함)
            logger.error(f"Bulk operations HTTP error: {e.response.status_code} - {e.response.text[:500]}...") # 응답 텍스트는 너무 길 수 있으므로 일부만 로깅
            raise # 예외를 다시 발생시켜 호출 측에서 처리하도록 함
        except Exception as e:
            # 기타 예외 로깅
            logger.exception(f"Bulk operations failed: {e}")
            raise # 예외를 다시 발생시켜 호출 측에서 처리하도록 함

    # --- HEMA 데이터 조회 API ---
    # 아래 조회 API들은 유사한 패턴을 가집니다:
    # 1. 클라이언트 가져오기
    # 2. 요청 파라미터 구성
    # 3. GET 요청 전송
    # 4. 응답 성공 시 JSON 파싱 및 스키마 객체 리스트로 변환하여 반환
    # 5. 예외 발생 시 로깅 후 빈 리스트 반환 (호출 측에서 오류 처리를 유연하게 하기 위함)

    async def get_information_snippets(
        self,
        user_id: str, # 필수 파라미터: 사용자 ID
        session_id: Optional[str] = None, # 선택 파라미터: 세션 ID
        keywords: Optional[List[str]] = None, # 선택 파라미터: 검색 키워드 리스트
        limit: int = 10 # 선택 파라미터: 최대 결과 수, 기본값 10
    ) -> List[InformationSnippetSchema]:
        """정보 조각(Information Snippets)을 조회합니다."""
        try:
            client = await self._get_client()
            logger.debug(f"Fetching information snippets for user_id: {user_id}, session_id: {session_id}, keywords: {keywords}, limit: {limit}")

            # API 요청 파라미터 구성
            params: Dict[str, Any] = {"user_id": user_id, "limit": limit}
            if session_id:
                params["session_id"] = session_id
            if keywords:
                # 키워드 리스트를 콤마(,)로 구분된 문자열로 변환 (API 규격에 따라 달라질 수 있음)
                params["keywords"] = ",".join(keywords)

            response = await client.get(
                "/api/external/hema/information_snippets", # 정보 조각 조회 API 엔드포인트
                params=params
            )
            response.raise_for_status()

            data = response.json()
            # 응답 데이터에서 "items" 키의 값(리스트)을 가져와 각 항목을 InformationSnippetSchema로 변환
            # "items" 키가 없거나 값이 비어있으면 빈 리스트 반환
            return [InformationSnippetSchema(**item) for item in data.get("items", [])]

        except Exception as e:
            logger.exception(f"Get information snippets failed for user_id {user_id}: {e}")
            return [] # 오류 발생 시 빈 리스트 반환

    async def get_idea_nodes(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        status: Optional[str] = None, # 선택 파라미터: 아이디어 상태 (예: "confirmed")
        node_type: Optional[str] = None, # 선택 파라미터: 아이디어 노드 타입 (예: "character")
        limit: int = 10
    ) -> List[IdeaNodeSchema]:
        """아이디어 노드(Idea Nodes)를 조회합니다."""
        try:
            client = await self._get_client()
            logger.debug(f"Fetching idea nodes for user_id: {user_id}, session_id: {session_id}, status: {status}, node_type: {node_type}, limit: {limit}")

            params: Dict[str, Any] = {"user_id": user_id, "limit": limit}
            if session_id:
                params["session_id"] = session_id
            if status:
                params["status"] = status
            if node_type:
                params["node_type"] = node_type

            response = await client.get(
                "/api/external/hema/idea_nodes", # 아이디어 노드 조회 API 엔드포인트
                params=params
            )
            response.raise_for_status()

            data = response.json()
            return [IdeaNodeSchema(**item) for item in data.get("items", [])]

        except Exception as e:
            logger.exception(f"Get idea nodes failed for user_id {user_id}: {e}")
            return []

    async def get_summary_nodes(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        summary_type: Optional[str] = None, # 선택 파라미터: 요약 타입 (예: "conversation_topic")
        limit: int = 5
    ) -> List[SummaryNodeSchema]:
        """요약 노드(Summary Nodes)를 조회합니다."""
        try:
            client = await self._get_client()
            logger.debug(f"Fetching summary nodes for user_id: {user_id}, session_id: {session_id}, summary_type: {summary_type}, limit: {limit}")

            params: Dict[str, Any] = {"user_id": user_id, "limit": limit}
            if session_id:
                params["session_id"] = session_id
            if summary_type:
                params["summary_type"] = summary_type

            response = await client.get(
                "/api/external/hema/summary_nodes", # 요약 노드 조회 API 엔드포인트
                params=params
            )
            response.raise_for_status()

            data = response.json()
            return [SummaryNodeSchema(**item) for item in data.get("items", [])]

        except Exception as e:
            logger.exception(f"Get summary nodes failed for user_id {user_id}: {e}")
            return []

    async def get_interaction_logs(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None, # 선택 파라미터: 이벤트 타입 (예: "user_query_to_slm")
        limit: int = 20
    ) -> List[HEMAInternalInteractionLogSchema]:
        """내부 상호작용 로그(Interaction Logs)를 조회합니다."""
        try:
            client = await self._get_client()
            logger.debug(f"Fetching interaction logs for user_id: {user_id}, session_id: {session_id}, event_type: {event_type}, limit: {limit}")

            params: Dict[str, Any] = {"user_id": user_id, "limit": limit}
            if session_id:
                params["session_id"] = session_id
            if event_type:
                params["event_type"] = event_type

            response = await client.get(
                "/api/external/hema/interaction_logs", # 상호작용 로그 조회 API 엔드포인트
                params=params
            )
            response.raise_for_status()

            data = response.json()
            return [HEMAInternalInteractionLogSchema(**item) for item in data.get("items", [])]

        except Exception as e:
            logger.exception(f"Get interaction logs failed for user_id {user_id}: {e}")
            return []

    # --- 유틸리티 메서드 ---

    async def health_check(self) -> bool:
        """
        "앞단 백엔드 서버"의 상태를 확인합니다.
        간단한 /health 엔드포인트 호출을 통해 서버가 응답하는지 확인합니다.

        Returns:
            bool: 서버가 건강하면 True, 그렇지 않으면 False.
        """
        try:
            client = await self._get_client()
            logger.info("Performing health check on front backend API...")
            # 타임아웃을 기본값보다 짧게 설정하여 빠른 응답 기대 (선택적)
            # response = await client.get("/health", timeout=5.0)
            response = await client.get("/health") # 기본 타임아웃 사용
            is_healthy = response.status_code == 200
            logger.info(f"Front backend health check status: {'Healthy' if is_healthy else 'Unhealthy'} (Status: {response.status_code})")
            return is_healthy
        except Exception as e:
            # 상태 확인 실패 시 경고 로깅 (오류로 처리하지 않고 False 반환)
            logger.warning(f"Front backend health check failed: {e}")
            return False
