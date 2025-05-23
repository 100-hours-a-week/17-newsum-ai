# ai/app/services/turn_processing_service.py

import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.services.hema_service import HEMAService
from app.services.slm_task_manager import SLMTaskManager
from app.services.llm_service import LLMService
from app.config.settings import settings
from app.utils.logger import get_logger
from app.utils.performance_monitor import performance_monitor
from app.utils.error_handler import error_handler, HEMAError, SLMError, ErrorSeverity
from app.api.v2.schemas.slm_task_schemas import SLMTaskType, SLMTaskPriority
from app.api.v2.schemas.hema_models import InteractionEventType

logger = get_logger(__name__)


class TurnProcessingService:
    """턴 처리 서비스
    
    사용자 턴을 처리하는 핵심 워크플로우를 담당
    """
    
    def __init__(
        self, 
        hema_service: HEMAService,
        slm_task_manager: SLMTaskManager,
        llm_service: LLMService
    ):
        self.hema_service = hema_service
        self.slm_task_manager = slm_task_manager
        self.llm_service = llm_service
        
    async def process_user_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """사용자 턴 처리 메인 메서드"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Processing turn {request_id} for user {user_id}")
            
            # 1. 요청 프로파일링
            profile = await self._profile_request(user_message, metadata)
            
            # 2. HEMA 컨텍스트 구성
            try:
                hema_context = await self.hema_service.retrieve_hema_context(
                    user_id=user_id,
                    session_id=session_id,
                    user_query=user_message,
                    metadata=metadata
                )
            except Exception as e:
                error_info = error_handler.handle_error(
                    error=e,
                    context={"operation": "hema_context_retrieval", "request_id": request_id},
                    user_id=user_id,
                    session_id=session_id
                )
                logger.warning(f"HEMA context retrieval failed, using empty context: {e}")
                # 빈 컨텍스트로 계속 진행
                from app.api.v2.schemas.hema_models import HEMAContext, HEMAContextSummary
                hema_context = HEMAContext(
                    context_id=str(uuid.uuid4()),
                    user_id=user_id,
                    session_id=session_id,
                    query=user_message,
                    items=[],
                    summary=HEMAContextSummary(
                        total_items=0, total_tokens=0, items_by_type={},
                        average_relevance=0.0, context_quality_score=0.0
                    ),
                    timestamp_created=datetime.now()
                )
            
            # 3. 상호작용 로그 기록
            await self._safe_log_interaction(
                user_id=user_id,
                session_id=session_id,
                event_type=InteractionEventType.USER_QUERY_TO_SLM,
                content_summary=f"User query: {user_message[:100]}...",
                metadata={"request_id": request_id, "profile": profile}
            )
            
            # 4. SLM 프롬프트 구성
            prompt = self.hema_service.create_prompt_with_context(
                user_message=user_message,
                hema_context=hema_context,
                task_type=profile.get('task_type', 'general')
            )
            
            # 5. SLM 파라미터 결정
            slm_params = self._determine_slm_parameters(profile, prompt)
            
            # 6. SLM 작업 실행
            slm_start_time = time.time()
            slm_response = await self.slm_task_manager.execute_slm_task(
                slm_payload=slm_params,
                task_type=SLMTaskType(profile.get('slm_task_type', 'interactive_qa')),
                priority=SLMTaskPriority(profile.get('priority', 'normal')),
                timeout=profile.get('timeout', settings.SLM_RESPONSE_TIMEOUT),
                metadata={"request_id": request_id, "user_id": user_id}
            )
            slm_processing_time = time.time() - slm_start_time
            
            # 7. SLM 응답 처리
            if slm_response.error:
                logger.error(f"SLM error for request {request_id}: {slm_response.error}")
                
                # SLM 오류를 HEMAError로 변환
                slm_error = SLMError(
                    message=f"SLM processing failed: {slm_response.error}",
                    severity=ErrorSeverity.HIGH if "timeout" in slm_response.error.lower() else ErrorSeverity.MEDIUM,
                    details={"request_id": request_id, "slm_response": slm_response.dict()}
                )
                
                error_info = error_handler.handle_error(
                    error=slm_error,
                    context={"operation": "slm_processing", "request_id": request_id},
                    user_id=user_id,
                    session_id=session_id
                )
                
                response_text = error_info["user_response"]
                hema_updates = []
                
                # 성능 모니터링 (실패)
                performance_monitor.record_request(
                    service="slm_processing",
                    processing_time=slm_processing_time,
                    success=False,
                    is_timeout="timeout" in slm_response.error.lower()
                )
            else:
                response_text = slm_response.generated_text or "응답을 생성할 수 없습니다."
                hema_updates = await self._generate_hema_updates(
                    user_id, session_id, user_message, response_text, hema_context
                )
                
                # 성능 모니터링 (성공)
                performance_monitor.record_request(
                    service="slm_processing",
                    processing_time=slm_processing_time,
                    success=True,
                    quality_score=slm_response.quality_score
                )
            
            # 8. HEMA 데이터 저장
            hema_save_success = await self._safe_save_hema_updates(
                user_id=user_id,
                session_id=session_id,
                updates=hema_updates,
                request_id=request_id
            )
            
            # 9. 처리 완료 로깅
            processing_time = time.time() - start_time
            await self._safe_log_interaction(
                user_id=user_id,
                session_id=session_id,
                event_type=InteractionEventType.SLM_RESPONSE_PROCESSED,
                content_summary=f"Response generated: {response_text[:100]}...",
                metadata={
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "hema_save_success": hema_save_success
                }
            )
            
            # 전체 처리 성능 모니터링
            performance_monitor.record_request(
                service="turn_processing",
                processing_time=processing_time,
                success=True
            )
            
            # 10. 결과 반환
            return {
                "session_id": session_id,
                "response_to_user": response_text,
                "hema_update_status": "success" if hema_save_success else "failed",
                "processing_time": processing_time,
                "context_summary": {
                    "used_items": hema_context.summary.total_items,
                    "context_quality": hema_context.summary.context_quality_score,
                    "average_relevance": hema_context.summary.average_relevance
                },
                "debug_info": {
                    "request_id": request_id,
                    "profile": profile,
                    "slm_response_status": "success" if not slm_response.error else "error",
                    "slm_processing_time": slm_processing_time
                } if settings.DEBUG_MODE else None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.exception(f"Turn processing failed for request {request_id}: {e}")
            
            # 오류 처리
            error_info = error_handler.handle_error(
                error=e,
                context={"operation": "turn_processing", "request_id": request_id},
                user_id=user_id,
                session_id=session_id
            )
            
            # 실패 성능 모니터링
            performance_monitor.record_request(
                service="turn_processing",
                processing_time=processing_time,
                success=False
            )
            
            return {
                "session_id": session_id,
                "response_to_user": error_info["user_response"],
                "hema_update_status": "failed",
                "processing_time": processing_time,
                "context_summary": None,
                "debug_info": {
                    "request_id": request_id,
                    "error": error_info["error"] if settings.DEBUG_MODE else None
                } if settings.DEBUG_MODE else None
            }
    
    async def _profile_request(
        self, 
        user_message: str, 
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """요청 프로파일링"""
        profile = {
            "message_length": len(user_message),
            "estimated_tokens": len(user_message) // 4,
            "task_type": "general",
            "slm_task_type": "interactive_qa",
            "priority": "normal",
            "timeout": settings.SLM_RESPONSE_TIMEOUT
        }
        
        # 메타데이터에서 task_type 추출
        if metadata and "task_type" in metadata:
            profile["task_type"] = metadata["task_type"]
            
            # task_type에 따른 SLM 작업 타입 결정
            task_type_mapping = {
                "아이디어_생성": "idea_generation",
                "캐릭터_개발": "creative_writing",
                "스토리_구성": "creative_writing",
                "문서_요약": "long_summary"
            }
            profile["slm_task_type"] = task_type_mapping.get(
                metadata["task_type"], "interactive_qa"
            )
        
        # 긴 메시지 처리
        if profile["estimated_tokens"] > settings.MAX_PROMPT_TOKENS // 2:
            profile["slm_task_type"] = "long_summary"
            profile["timeout"] = settings.SLM_RESPONSE_TIMEOUT * 2
        
        # 우선순위 설정 (메타데이터 기반)
        if metadata and metadata.get("priority"):
            profile["priority"] = metadata["priority"]
        
        return profile
    
    def _determine_slm_parameters(
        self, 
        profile: Dict[str, Any], 
        prompt: str
    ) -> Dict[str, Any]:
        """SLM 파라미터 결정"""
        # 기본 파라미터
        params = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
            "temperature": 0.7,
            "model": settings.DEFAULT_LLM_MODEL
        }
        
        # 작업 타입별 파라미터 조정
        task_type = profile.get("task_type", "general")
        
        if task_type in ["아이디어_생성", "창작"]:
            params["temperature"] = 0.8
            params["max_tokens"] = min(800, settings.DEFAULT_MAX_TOKENS * 2)
        elif task_type == "문서_요약":
            params["temperature"] = 0.3
            params["max_tokens"] = min(1000, settings.DEFAULT_MAX_TOKENS * 2)
        elif task_type == "분석":
            params["temperature"] = 0.5
            params["max_tokens"] = min(600, settings.DEFAULT_MAX_TOKENS * 1.5)
        
        # 토큰 예산 확인
        prompt_tokens = len(prompt) // 4  # 대략적인 계산
        available_tokens = settings.MAX_PROMPT_TOKENS - prompt_tokens - 100  # 버퍼
        
        if available_tokens < params["max_tokens"]:
            params["max_tokens"] = max(200, available_tokens)  # 최소 200 토큰
        
        return params
    
    async def _generate_hema_updates(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        response_text: str,
        hema_context
    ) -> List[Dict[str, Any]]:
        """HEMA 업데이트 생성"""
        updates = []
        
        try:
            # 1. 새로운 아이디어 추출 (간단한 휴리스틱)
            if any(keyword in user_message.lower() for keyword in ["아이디어", "캐릭터", "스토리", "설정"]):
                idea_update = {
                    "action": "create",
                    "entity_type": "IdeaNode",
                    "data": {
                        "idea_id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "user_id": user_id,
                        "node_type": self._extract_idea_type(user_message),
                        "title": self._extract_idea_title(response_text),
                        "description": response_text[:500],  # 응답의 첫 500자
                        "status": "proposed",
                        "version": 1,
                        "linked_snippet_ids": [item.item_id for item in hema_context.items 
                                              if item.item_type.value == "InformationSnippet"],
                        "timestamp_created": datetime.now().isoformat(),
                        "timestamp_updated": datetime.now().isoformat()
                    }
                }
                updates.append(idea_update)
            
            # 2. 세션 요약 업데이트 (주기적으로)
            if len(user_message) > 100:  # 의미있는 대화인 경우
                summary_update = {
                    "action": "create",
                    "entity_type": "SummaryNode",
                    "data": {
                        "summary_id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "user_id": user_id,
                        "summary_type": "conversation_topic",
                        "title": f"대화 주제: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        "summary_text": f"사용자 질문: {user_message[:200]}...\n시스템 응답: {response_text[:200]}...",
                        "source_ids": [],
                        "keywords": self.hema_service._extract_keywords(user_message + " " + response_text),
                        "timestamp_generated": datetime.now().isoformat()
                    }
                }
                updates.append(summary_update)
            
        except Exception as e:
            logger.exception(f"Failed to generate HEMA updates: {e}")
        
        return updates
    
    def _extract_idea_type(self, message: str) -> str:
        """메시지에서 아이디어 타입 추출"""
        message_lower = message.lower()
        if any(word in message_lower for word in ["캐릭터", "인물", "주인공"]):
            return "character"
        elif any(word in message_lower for word in ["플롯", "스토리", "줄거리", "내용"]):
            return "plot_point"
        elif any(word in message_lower for word in ["배경", "설정", "세계관"]):
            return "setting"
        elif any(word in message_lower for word in ["주제", "테마", "메시지"]):
            return "theme"
        else:
            return "plot_point"  # 기본값
    
    def _extract_idea_title(self, response_text: str) -> str:
        """응답에서 아이디어 제목 추출"""
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:  # 적당한 길이의 첫 번째 라인
                return line
        
        # 제목을 찾지 못한 경우 첫 50자 사용
        return response_text[:50] + "..." if len(response_text) > 50 else response_text
    
    async def close(self):
        """리소스 정리"""
        try:
            await self.hema_service.close()
            await self.slm_task_manager.close()
            logger.info("Turn Processing Service closed")
        except Exception as e:
            logger.exception(f"Error closing Turn Processing Service: {e}")
    
    async def _safe_log_interaction(
        self,
        user_id: str,
        session_id: str,
        event_type: InteractionEventType,
        content_summary: str,
        linked_hema_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """안전한 상호작용 로그 기록"""
        try:
            return await self.hema_service.log_interaction(
                user_id=user_id,
                session_id=session_id,
                event_type=event_type,
                content_summary=content_summary,
                linked_hema_ids=linked_hema_ids,
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Failed to log interaction: {e}")
            return False
    
    async def _safe_save_hema_updates(
        self,
        user_id: str,
        session_id: str,
        updates: List[Dict[str, Any]],
        request_id: str
    ) -> bool:
        """안전한 HEMA 데이터 저장"""
        try:
            if not updates:
                return True
            
            save_start_time = time.time()
            success = await self.hema_service.save_hema_updates(
                user_id=user_id,
                session_id=session_id,
                updates=updates
            )
            save_time = time.time() - save_start_time
            
            # HEMA 저장 성능 모니터링
            performance_monitor.record_request(
                service="hema_data_save",
                processing_time=save_time,
                success=success
            )
            
            return success
            
        except Exception as e:
            logger.warning(f"Failed to save HEMA updates for request {request_id}: {e}")
            
            # 실패 성능 모니터링
            performance_monitor.record_request(
                service="hema_data_save",
                processing_time=0.0,
                success=False
            )
            
            return False
