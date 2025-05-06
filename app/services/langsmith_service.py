# ai/app/services/langsmith_service.py

import os
from typing import Dict, Any, Optional, Union
from app.config.settings import Settings # 변경: 중앙 설정 객체 임포트
from app.utils.logger import get_logger


settings = Settings()
# LangSmith 라이브러리 동적 로딩 시도
try:
    from langsmith import Client
    # LangChainTracer는 langchain_core 또는 langchain 패키지에 있을 수 있음
    # langchain_core가 더 표준적
    try:
        from langchain_core.tracers.langchain import LangChainTracer
    except ImportError:
        # 이전 버전 호환 또는 다른 위치일 경우
        from langchain.callbacks.tracers.langchain import LangChainTracer # type: ignore
    LANGSMITH_AVAILABLE = True
    logger_temp = get_logger("LangSmithService_Init") # 임시 로거
    logger_temp.info("LangSmith 및 LangChainTracer 라이브러리를 찾았습니다.")
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger_temp = get_logger("LangSmithService_Init") # 임시 로거
    logger_temp.warning("LangSmith 또는 LangChain 관련 라이브러리가 설치되지 않았습니다. LangSmith 연동 기능이 비활성화됩니다.")
    # Define dummy classes if not available to prevent NameError later if type hints are used extensively
    class Client: pass # type: ignore
    class LangChainTracer: pass # type: ignore

class LangSmithService:
    """
    LangSmith 추적 및 로깅을 관리하는 서비스.
    LangChain/LangGraph 자동 추적 설정을 돕거나 수동으로 Run을 기록하는 기능을 제공합니다.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: Optional[str] = None,
        endpoint: Optional[str] = None, # LangSmith API 엔드포인트 (온프레미스 등)
        logger_name: str = "LangSmithService"
    ):
        """
        LangSmith 서비스 초기화.

        Args:
            api_key (Optional[str]): LangSmith API 키. 기본값: 환경변수 LANGCHAIN_API_KEY.
            project_name (Optional[str]): 추적을 기록할 프로젝트 이름. 기본값: 환경변수 LANGCHAIN_PROJECT 또는 "default_project".
            endpoint (Optional[str]): LangSmith API 엔드포인트 URL. 기본값: 환경변수 LANGCHAIN_ENDPOINT 또는 LangSmith 기본값.
            logger_name (str): 로거 인스턴스 이름.
        """
        self.logger = get_logger(logger_name)
        self.client: Optional[Client] = None
        self.tracer: Optional[LangChainTracer] = None

        # LangSmith 라이브러리 설치 여부 확인
        if not LANGSMITH_AVAILABLE:
            self.logger.warning("LangSmith 라이브러리가 설치되지 않아 서비스를 초기화할 수 없습니다.")
            return # 서비스 비활성화 상태로 유지

        # 설정 값 로드 (인자 > 환경 변수 > 기본값)
        self.api_key = api_key or settings.LANGCHAIN_API_KEY
        self.project_name = project_name or settings.LANGCHAIN_PROJECT
        self.api_endpoint = endpoint or settings.LANGCHAIN_ENDPOINT

        # API 키 유효성 검사
        if not self.api_key:
            self.logger.warning("LangSmith API 키가 제공되지 않았습니다 (인자 또는 LANGCHAIN_API_KEY 환경 변수 필요). LangSmith 연동이 비활성화됩니다.")
            return # 서비스 비활성화

        try:
            # LangSmith 클라이언트 초기화 인자 구성
            client_args = {"api_key": self.api_key}
            # <<< 수정: AnyHttpUrl 객체일 경우 문자열로 변환하여 전달 >>>
            if self.api_endpoint:
                # Pydantic URL 타입은 str()로 변환 가능
                client_args["api_url"] = str(self.api_endpoint)

            # LangSmith 클라이언트 초기화
            self.client = Client(**client_args)

            # LangChain 트레이서 생성
            self.tracer = LangChainTracer(
                project_name=self.project_name,
                client=self.client
            )
            self.logger.info(f"LangSmith 서비스 초기화 완료. 프로젝트: '{self.project_name}'")
            # 환경 변수 LANGCHAIN_TRACING_V2=true 설정 여부도 안내하면 좋음
            if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
                self.logger.info("환경 변수 LANGCHAIN_TRACING_V2가 'true'로 설정되어 자동 추적이 활성화될 수 있습니다.")
            else:
                 self.logger.info("자동 추적을 위해서는 환경 변수 LANGCHAIN_TRACING_V2를 'true'로 설정하세요.")

        except Exception as e:
            self.logger.error(f"LangSmith 초기화 중 오류 발생: {e}", exc_info=True)
            self.client = None
            self.tracer = None

    def log_run(
        self,
        run_name: str, # LangChain의 @traceable 데코레이터와 유사한 역할
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[Union[Exception, str]] = None,
        run_type: str = "chain", # 또는 "llm", "tool", "retriever" 등
        **kwargs: Any # 추가 메타데이터 (예: execution_order, child_run_ids 등)
    ) -> None:
        """
        독립적인 실행(Run)을 LangSmith에 수동으로 기록합니다.
        LangChain/LangGraph 외부의 커스텀 로직 추적에 유용할 수 있습니다.

        Args:
            run_name (str): 실행의 이름 (예: "데이터_전처리", "최종_결정").
            inputs (Dict[str, Any]): 실행에 사용된 입력 데이터.
            outputs (Optional[Dict[str, Any]]): 실행 결과 데이터.
            error (Optional[Union[Exception, str]]): 실행 중 발생한 오류 (문자열 또는 예외 객체).
            run_type (str): 실행 유형 ('chain', 'llm', 'tool' 등). 기본값: "chain".
            **kwargs: LangSmith Run에 기록할 추가 속성.
        """
        if not self.client:
            self.logger.warning("LangSmith 클라이언트가 초기화되지 않아 Run을 기록할 수 없습니다.")
            return

        try:
            # 오류 객체를 문자열로 변환
            error_str = None
            if isinstance(error, Exception):
                error_str = f"{type(error).__name__}: {str(error)}"
            elif isinstance(error, str):
                error_str = error

            # LangSmith에 Run 생성 요청
            # 참고: create_run은 동기 메서드일 수 있음 (langsmith 라이브러리 버전에 따라 확인 필요)
            #       비동기 환경에서 오래 걸리는 동기 호출은 이벤트 루프를 차단할 수 있으므로 주의.
            #       현재 langsmith 0.1.x 버전 기준으로는 동기 메서드임.
            #       asyncio.to_thread 등을 사용하여 별도 스레드에서 실행 고려 가능.
            run = self.client.create_run(
                name=run_name, # API 파라미터 이름은 'name'
                inputs=inputs,
                outputs=outputs or {}, # None 대신 빈 딕셔너리 전달
                error=error_str,
                run_type=run_type,
                project_name=self.project_name, # 명시적으로 프로젝트 이름 지정 가능
                **kwargs
            )

            self.logger.debug(f"LangSmith Run 기록 성공: name='{run_name}', run_id='{run.id}'") # run 객체에서 id 접근 가능 가정

        except Exception as e:
            self.logger.error(f"LangSmith Run 기록 중 오류 발생: name='{run_name}', error={e}", exc_info=True)

    def get_tracer(self) -> Optional[LangChainTracer]:
        """
        초기화된 LangChainTracer 인스턴스를 반환합니다.
        LangChain/LangGraph 구성 요소에 콜백으로 전달하여 자동 추적을 활성화할 수 있습니다.
        (환경 변수 설정만으로도 자동 추적이 되는 경우가 많아 항상 필요한 것은 아님)

        Returns:
            Optional[LangChainTracer]: LangChainTracer 인스턴스 또는 None (초기화 실패 시).
        """
        if not self.tracer:
             self.logger.warning("LangSmith 트레이서가 초기화되지 않았습니다.")
        return self.tracer

    def close(self):
        """
        LangSmith 서비스 관련 리소스를 정리합니다. (현재 특별한 정리 작업은 없음)
        """
        if self.client:
            try:
                # 특별히 client.close() 같은 메서드는 langsmith 라이브러리에 없을 수 있음
                self.logger.info("LangSmith 서비스 종료 (정리 작업 없음).")
                # self.client = None # 필요시 상태 변경
                # self.tracer = None
            except Exception as e:
                self.logger.error(f"LangSmith 서비스 종료 중 오류 발생: {e}", exc_info=True)

# --- 선택적 싱글턴 인스턴스 ---
# 필요에 따라 주석 해제하여 사용
# langsmith_service = LangSmithService()