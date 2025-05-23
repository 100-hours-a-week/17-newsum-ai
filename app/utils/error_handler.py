# ai/app/utils/error_handler.py

import traceback
from typing import Any, Dict, Optional, Union
from datetime import datetime
from enum import Enum

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """오류 심각도 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """오류 카테고리"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RESOURCE_NOT_FOUND = "resource_not_found"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class HEMAError(Exception):
    """HEMA 시스템 기본 예외 클래스"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.retry_after = retry_after
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """오류 정보를 딕셔너리로 변환"""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp.isoformat()
        }


class SLMError(HEMAError):
    """SLM 관련 오류"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.EXTERNAL_API, **kwargs)


class HEMADataError(HEMAError):
    """HEMA 데이터 관련 오류"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE, **kwargs)


class RedisError(HEMAError):
    """Redis 관련 오류"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE, **kwargs)


class FrontBackendAPIError(HEMAError):
    """앞단 백엔드 API 관련 오류"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.EXTERNAL_API, **kwargs)
        self.details["status_code"] = status_code


class ErrorHandler:
    """통합 오류 처리기"""
    
    def __init__(self):
        self.error_counts = {}
        self.last_errors = {}
    
    def handle_error(
        self,
        error: Union[Exception, HEMAError],
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """오류 처리 및 로깅"""
        try:
            # HEMAError가 아닌 경우 변환
            if not isinstance(error, HEMAError):
                hema_error = self._convert_to_hema_error(error)
            else:
                hema_error = error
            
            # 컨텍스트 정보 추가
            error_context = {
                "user_id": user_id,
                "session_id": session_id,
                "context": context or {},
                "stack_trace": traceback.format_exc() if not isinstance(error, HEMAError) else None
            }
            
            # 로깅
            self._log_error(hema_error, error_context)
            
            # 통계 업데이트
            self._update_error_stats(hema_error)
            
            # 사용자 응답 생성
            user_response = self._generate_user_response(hema_error)
            
            return {
                "error": hema_error.to_dict(),
                "user_response": user_response,
                "should_retry": self._should_retry(hema_error),
                "retry_after": hema_error.retry_after
            }
            
        except Exception as e:
            logger.exception(f"Error in error handler: {e}")
            return {
                "error": {
                    "message": "시스템 오류가 발생했습니다.",
                    "category": ErrorCategory.SYSTEM.value,
                    "severity": ErrorSeverity.HIGH.value
                },
                "user_response": "죄송합니다. 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "should_retry": True,
                "retry_after": 60
            }
    
    def _convert_to_hema_error(self, error: Exception) -> HEMAError:
        """일반 예외를 HEMAError로 변환"""
        error_type = type(error).__name__
        message = str(error)
        
        # 예외 타입별 카테고리 매핑
        category_mapping = {
            "TimeoutError": ErrorCategory.TIMEOUT,
            "ConnectionError": ErrorCategory.NETWORK,
            "HTTPException": ErrorCategory.EXTERNAL_API,
            "ValidationError": ErrorCategory.VALIDATION,
            "PermissionError": ErrorCategory.PERMISSION,
            "FileNotFoundError": ErrorCategory.RESOURCE_NOT_FOUND,
            "KeyError": ErrorCategory.VALIDATION,
            "ValueError": ErrorCategory.VALIDATION,
        }
        
        category = category_mapping.get(error_type, ErrorCategory.UNKNOWN)
        
        # 심각도 결정
        severity = ErrorSeverity.MEDIUM
        if "timeout" in message.lower():
            severity = ErrorSeverity.HIGH
        elif "critical" in message.lower():
            severity = ErrorSeverity.CRITICAL
        
        return HEMAError(
            message=message,
            category=category,
            severity=severity,
            details={"original_error_type": error_type}
        )
    
    def _log_error(self, error: HEMAError, context: Dict[str, Any]):
        """오류 로깅"""
        log_data = {
            "error": error.to_dict(),
            "context": context
        }
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR: {error.message}", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR: {error.message}", extra=log_data)
        else:
            logger.info(f"LOW SEVERITY ERROR: {error.message}", extra=log_data)
    
    def _update_error_stats(self, error: HEMAError):
        """오류 통계 업데이트"""
        error_key = f"{error.category.value}:{error.severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = error.timestamp
    
    def _generate_user_response(self, error: HEMAError) -> str:
        """사용자 응답 메시지 생성"""
        user_messages = {
            ErrorCategory.VALIDATION: "입력 정보를 확인해 주세요.",
            ErrorCategory.AUTHENTICATION: "인증이 필요합니다.",
            ErrorCategory.PERMISSION: "권한이 없습니다.",
            ErrorCategory.RESOURCE_NOT_FOUND: "요청하신 정보를 찾을 수 없습니다.",
            ErrorCategory.EXTERNAL_API: "외부 서비스 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요.",
            ErrorCategory.DATABASE: "데이터 처리 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
            ErrorCategory.NETWORK: "네트워크 연결에 문제가 있습니다. 연결을 확인해주세요.",
            ErrorCategory.TIMEOUT: "요청 처리 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
            ErrorCategory.RATE_LIMIT: "요청이 너무 많습니다. 잠시 후 다시 시도해주세요.",
            ErrorCategory.SYSTEM: "시스템 오류가 발생했습니다. 관리자에게 문의해주세요.",
            ErrorCategory.UNKNOWN: "알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        }
        
        base_message = user_messages.get(error.category, user_messages[ErrorCategory.UNKNOWN])
        
        # 심각도에 따른 메시지 조정
        if error.severity == ErrorSeverity.CRITICAL:
            return f"심각한 시스템 오류가 발생했습니다. 즉시 관리자에게 문의해주세요."
        elif error.retry_after:
            return f"{base_message} ({error.retry_after}초 후 다시 시도 가능)"
        
        return base_message
    
    def _should_retry(self, error: HEMAError) -> bool:
        """재시도 가능 여부 판단"""
        retry_categories = {
            ErrorCategory.EXTERNAL_API,
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.DATABASE
        }
        
        non_retry_categories = {
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.PERMISSION,
            ErrorCategory.VALIDATION
        }
        
        if error.category in non_retry_categories:
            return False
        
        if error.severity == ErrorSeverity.CRITICAL:
            return False
        
        return error.category in retry_categories
    
    def get_error_stats(self) -> Dict[str, Any]:
        """오류 통계 반환"""
        return {
            "error_counts": dict(self.error_counts),
            "last_errors": {k: v.isoformat() for k, v in self.last_errors.items()},
            "total_errors": sum(self.error_counts.values())
        }


# 전역 오류 처리기 인스턴스
error_handler = ErrorHandler()
