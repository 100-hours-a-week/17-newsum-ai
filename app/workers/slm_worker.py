# ai/app/workers/slm_worker.py

import asyncio
import json
import signal
import sys
from typing import Dict, Any, Optional
from datetime import datetime

from app.services.database_client import DatabaseClient
from app.services.llm_service import LLMService
from app.config.settings import settings
from app.utils.logger import get_logger
from app.api.v2.schemas.slm_task_schemas import SLMTaskRequest, SLMTaskResponse

logger = get_logger(__name__)


class SLMWorker:
    """SLM 워커 프로세스
    
    Redis에서 SLM 작업을 수신하고 vLLM으로 처리한 후 결과를 반환하는 워커
    """
    
    def __init__(self, worker_id: str, request_channel: str):
        self.worker_id = worker_id
        self.request_channel = request_channel
        self.db_client = DatabaseClient()
        self.llm_service = LLMService()
        self.semaphore = asyncio.Semaphore(settings.WORKER_MAX_CONCURRENT_VLLM_REQUESTS)
        self.running = False
        self.processed_count = 0
        self.error_count = 0
        
        # 종료 신호 처리
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """종료 신호 처리"""
        logger.info(f"Worker {self.worker_id} received signal {signum}, shutting down...")
        self.running = False
    
    async def start(self):
        """워커 시작"""
        try:
            logger.info(f"SLM Worker {self.worker_id} starting on channel {self.request_channel}")
            self.running = True
            
            # Redis 구독 시작
            await self.db_client.subscribe(
                channel=self.request_channel,
                callback=self._handle_slm_task
            )
            
            # 메인 루프
            while self.running:
                await asyncio.sleep(1)
                
                # 주기적 상태 로깅
                if self.processed_count > 0 and self.processed_count % 100 == 0:
                    logger.info(f"Worker {self.worker_id} processed {self.processed_count} tasks, {self.error_count} errors")
        
        except Exception as e:
            logger.exception(f"Worker {self.worker_id} error: {e}")
        finally:
            await self._shutdown()
    
    async def _handle_slm_task(self, message_data: Dict[str, Any]):
        """SLM 작업 처리"""
        request_id = None
        start_time = datetime.now()
        
        try:
            # 메시지 파싱
            if isinstance(message_data, str):
                message_data = json.loads(message_data)
            
            task_request = SLMTaskRequest(**message_data)
            request_id = task_request.request_id
            
            logger.debug(f"Worker {self.worker_id} processing task {request_id}")
            
            # vLLM 동시 요청 제어
            async with self.semaphore:
                # LLM 서비스 호출
                result = await self.llm_service.generate_text(
                    **task_request.slm_payload
                )
                
                # 처리 시간 계산
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # 응답 구성
                if 'choices' in result and result['choices']:
                    generated_text = result['choices'][0].get('message', {}).get('content') or result['choices'][0].get('text', '')
                    response = SLMTaskResponse(
                        request_id=request_id,
                        generated_text=generated_text,
                        raw_response=result,
                        processing_time=processing_time,
                        worker_id=self.worker_id,
                        token_usage=result.get('usage', {}),
                        quality_score=self._calculate_quality_score(generated_text)
                    )
                else:
                    response = SLMTaskResponse(
                        request_id=request_id,
                        error="No valid response from LLM",
                        raw_response=result,
                        processing_time=processing_time,
                        worker_id=self.worker_id
                    )
            
            # 결과 발행
            response_channel = f"{settings.SLM_RESPONSE_CHANNEL_PREFIX}{request_id}"
            await self.db_client.publish(response_channel, response.dict())
            
            self.processed_count += 1
            logger.debug(f"Worker {self.worker_id} completed task {request_id} in {processing_time:.2f}s")
            
        except Exception as e:
            self.error_count += 1
            logger.exception(f"Worker {self.worker_id} task processing error: {e}")
            
            # 오류 응답 발행
            if request_id:
                error_response = SLMTaskResponse(
                    request_id=request_id,
                    error=str(e),
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    worker_id=self.worker_id
                )
                
                try:
                    response_channel = f"{settings.SLM_RESPONSE_CHANNEL_PREFIX}{request_id}"
                    await self.db_client.publish(response_channel, error_response.dict())
                except Exception as publish_error:
                    logger.error(f"Failed to publish error response: {publish_error}")
    
    def _calculate_quality_score(self, generated_text: Optional[str]) -> Optional[float]:
        """응답 품질 점수 계산"""
        if not generated_text:
            return 0.0
        
        # 간단한 품질 평가 기준
        score = 1.0
        
        # 길이 체크
        if len(generated_text) < 10:
            score *= 0.3
        elif len(generated_text) < 50:
            score *= 0.7
        
        # 완성도 체크 (마지막이 완전한 문장인지)
        if not generated_text.rstrip().endswith(('.', '!', '?', '다', '요', '어요', '습니다')):
            score *= 0.8
        
        # 반복 패턴 체크
        words = generated_text.split()
        if len(words) > 10:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.5:  # 고유 단어 비율이 낮으면
                score *= 0.6
        
        return round(score, 2)
    
    async def _shutdown(self):
        """워커 종료"""
        try:
            logger.info(f"Worker {self.worker_id} shutting down...")
            logger.info(f"Final stats - Processed: {self.processed_count}, Errors: {self.error_count}")
            
            # 연결 종료
            await self.db_client.close()
            await self.llm_service.close()
            
            logger.info(f"Worker {self.worker_id} shutdown complete")
            
        except Exception as e:
            logger.exception(f"Error during worker shutdown: {e}")
