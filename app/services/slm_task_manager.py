# ai/app/services/slm_task_manager.py

import asyncio
import uuid
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
import json

from app.services.database_client import DatabaseClient
from app.config.settings import settings
from app.utils.logger import get_logger
from app.api.v2.schemas.slm_task_schemas import (
    SLMTaskRequest, 
    SLMTaskResponse, 
    SLMTaskStatus,
    SLMTaskType,
    SLMTaskPriority
)

logger = get_logger(__name__)


class SLMTaskManager:
    """SLM 작업 관리자
    
    Redis Pub/Sub 기반으로 SLM 작업을 발행하고 결과를 수신하는 관리자
    """
    
    def __init__(self, database_client: DatabaseClient):
        self.db_client = database_client
        self.pending_requests: Dict[str, Dict[str, Any]] = {}
        self._response_callbacks: Dict[str, Callable] = {}
        
    async def publish_slm_task(
        self,
        task_request: SLMTaskRequest,
        target_channel: Optional[str] = None
    ) -> None:
        """SLM 작업을 Redis 채널에 발행"""
        try:
            # 채널 결정
            if target_channel is None:
                target_channel = self._determine_channel(task_request)
            
            # 메시지 발행
            message_data = task_request.dict()
            await self.db_client.publish(target_channel, json.dumps(message_data))
            
            # 대기 중인 요청 목록에 추가
            self.pending_requests[task_request.request_id] = {
                "task_request": task_request,
                "channel": target_channel,
                "timestamp": datetime.now(),
                "status": "published"
            }
            
            logger.info(f"Published SLM task {task_request.request_id} to {target_channel}")
            
        except Exception as e:
            logger.exception(f"Failed to publish SLM task {task_request.request_id}: {e}")
            raise
    
    async def await_slm_response(
        self,
        request_id: str,
        timeout: Optional[float] = None
    ) -> SLMTaskResponse:
        """SLM 작업 결과 대기"""
        if timeout is None:
            timeout = settings.SLM_RESPONSE_TIMEOUT
        
        try:
            # 응답 채널 구독
            response_channel = f"{settings.SLM_RESPONSE_CHANNEL_PREFIX}{request_id}"
            
            # 응답 대기용 Event 생성
            response_event = asyncio.Event()
            response_data = {}
            
            def response_callback(message_data: Dict[str, Any]):
                """응답 수신 콜백"""
                try:
                    response_data.update(message_data)
                    response_event.set()
                except Exception as e:
                    logger.error(f"Response callback error: {e}")
                    response_event.set()
            
            # 구독 설정
            await self.db_client.subscribe(response_channel, response_callback)
            self._response_callbacks[request_id] = response_callback
            
            # 응답 대기
            try:
                await asyncio.wait_for(response_event.wait(), timeout=timeout)
                
                # 응답 데이터 파싱
                if response_data:
                    return SLMTaskResponse(**response_data)
                else:
                    # 응답 데이터가 없는 경우
                    return SLMTaskResponse(
                        request_id=request_id,
                        error="No response data received"
                    )
                    
            except asyncio.TimeoutError:
                logger.warning(f"SLM task {request_id} timeout after {timeout}s")
                return SLMTaskResponse(
                    request_id=request_id,
                    error=f"Task timeout after {timeout} seconds"
                )
                
        except Exception as e:
            logger.exception(f"Failed to await SLM response for {request_id}: {e}")
            return SLMTaskResponse(
                request_id=request_id,
                error=f"Internal error: {str(e)}"
            )
        finally:
            # 정리 작업
            await self._cleanup_request(request_id)
    
    def _determine_channel(self, task_request: SLMTaskRequest) -> str:
        """작업 타입에 따른 채널 결정"""
        # 프롬프트 길이 예상
        slm_payload = task_request.slm_payload
        messages = slm_payload.get('messages', [])
        
        # 대략적인 토큰 수 계산 (4 chars ≈ 1 token)
        total_chars = sum(len(str(msg)) for msg in messages)
        estimated_tokens = total_chars // 4
        
        # 긴 컨텍스트 작업 판별
        if (task_request.task_type == SLMTaskType.LONG_SUMMARY or 
            estimated_tokens > settings.MAX_PROMPT_TOKENS // 2):
            return settings.SLM_LONG_CONTEXT_REQUEST_CHANNEL
        else:
            return settings.SLM_INTERACTIVE_REQUEST_CHANNEL
    
    async def _cleanup_request(self, request_id: str):
        """요청 정리"""
        try:
            # 구독 해제
            response_channel = f"{settings.SLM_RESPONSE_CHANNEL_PREFIX}{request_id}"
            if request_id in self._response_callbacks:
                # 실제 구독 해제는 database_client의 구현에 따라 다름
                # 여기서는 콜백만 제거
                del self._response_callbacks[request_id]
            
            # 대기 중인 요청에서 제거
            self.pending_requests.pop(request_id, None)
            
        except Exception as e:
            logger.warning(f"Cleanup failed for request {request_id}: {e}")
    
    async def execute_slm_task(
        self,
        slm_payload: Dict[str, Any],
        task_type: SLMTaskType = SLMTaskType.INTERACTIVE_QA,
        priority: SLMTaskPriority = SLMTaskPriority.NORMAL,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SLMTaskResponse:
        """단일 SLM 작업 실행 (발행 + 결과 대기)"""
        request_id = str(uuid.uuid4())
        
        # 작업 요청 생성
        task_request = SLMTaskRequest(
            request_id=request_id,
            task_type=task_type,
            priority=priority,
            slm_payload=slm_payload,
            metadata=metadata or {},
            timeout=timeout or settings.SLM_RESPONSE_TIMEOUT
        )
        
        # 작업 발행
        await self.publish_slm_task(task_request)
        
        # 결과 대기 및 반환
        return await self.await_slm_response(request_id, timeout)
    
    async def get_task_status(self, request_id: str) -> SLMTaskStatus:
        """작업 상태 조회"""
        if request_id in self.pending_requests:
            request_info = self.pending_requests[request_id]
            return SLMTaskStatus(
                request_id=request_id,
                status="pending",
                progress=0.0
            )
        else:
            return SLMTaskStatus(
                request_id=request_id,
                status="unknown",
                progress=None
            )
    
    async def cancel_task(self, request_id: str) -> bool:
        """작업 취소 (가능한 경우)"""
        try:
            if request_id in self.pending_requests:
                await self._cleanup_request(request_id)
                logger.info(f"Cancelled SLM task {request_id}")
                return True
            return False
        except Exception as e:
            logger.exception(f"Failed to cancel task {request_id}: {e}")
            return False
    
    async def get_pending_tasks(self) -> List[str]:
        """대기 중인 작업 목록 반환"""
        return list(self.pending_requests.keys())
    
    async def cleanup_expired_tasks(self, max_age_minutes: int = 30):
        """만료된 작업 정리"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
            expired_requests = []
            
            for request_id, request_info in self.pending_requests.items():
                if request_info['timestamp'] < cutoff_time:
                    expired_requests.append(request_id)
            
            for request_id in expired_requests:
                await self._cleanup_request(request_id)
                logger.info(f"Cleaned up expired task {request_id}")
            
            if expired_requests:
                logger.info(f"Cleaned up {len(expired_requests)} expired tasks")
                
        except Exception as e:
            logger.exception(f"Failed to cleanup expired tasks: {e}")
    
    async def close(self):
        """리소스 정리"""
        try:
            # 모든 대기 중인 요청 정리
            for request_id in list(self.pending_requests.keys()):
                await self._cleanup_request(request_id)
            
            logger.info("SLM Task Manager closed")
        except Exception as e:
            logger.exception(f"Error closing SLM Task Manager: {e}")
