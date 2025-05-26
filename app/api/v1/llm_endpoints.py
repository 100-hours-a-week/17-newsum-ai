# ai/app/api/v1/llm_endpoints.py

import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.services.database_client import DatabaseClient
from app.dependencies import DatabaseClientDep
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["LLM Queue V1"])

class LLMRequestPayload(BaseModel):
    """LLM 요청 페이로드"""
    prompt: Optional[str] = Field(None, description="단일 프롬프트")
    messages: Optional[List[Dict[str, str]]] = Field(None, description="채팅 메시지 리스트")
    system_prompt_content: Optional[str] = Field(None, description="시스템 프롬프트")
    max_tokens: int = Field(512, description="최대 토큰 수")
    temperature: float = Field(0.7, description="온도 설정")
    stop_sequences: Optional[List[str]] = Field(None, description="중지 시퀀스")
    model_name: Optional[str] = Field(None, description="모델 이름")

class LLMTaskResponse(BaseModel):
    """LLM 작업 응답"""
    task_id: str
    message: str
    queue_position: Optional[int] = None

class LLMResultResponse(BaseModel):
    """LLM 결과 응답"""
    task_id: str
    status: str  # "completed", "processing", "failed", "not_found"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# === Producer: LLM 작업 요청 ===

@router.post("/tasks", 
             response_model=LLMTaskResponse,
             status_code=status.HTTP_202_ACCEPTED,
             summary="LLM 작업 생성",
             description="LLM 텍스트 생성 작업을 큐에 추가합니다.")
async def create_llm_task(
    request: LLMRequestPayload,
    db: DatabaseClientDep
):
    """LLM 생성 작업을 Redis 큐에 추가합니다."""
    
    # 입력 검증
    if not request.prompt and not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'prompt' or 'messages' must be provided."
        )
    
    # 고유 작업 ID 생성
    task_id = str(uuid.uuid4())
    
    # 작업 데이터 구성
    task_data = {
        "task_id": task_id,
        "payload": request.model_dump(exclude_unset=True)  # None 값 제외
    }
    
    try:
        # Redis 큐에 작업 추가
        queue_len = await db.lpush("llm_request_queue", task_data)
        
        if queue_len is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add task to the queue."
            )
        
        logger.info(f"LLM 작업 생성됨: task_id={task_id}, queue_length={queue_len}")
        
        return LLMTaskResponse(
            task_id=task_id,
            message="Task accepted for processing.",
            queue_position=queue_len
        )
        
    except Exception as e:
        logger.error(f"LLM 작업 생성 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create LLM task: {str(e)}"
        )

# === Consumer 결과 조회 ===

@router.get("/tasks/{task_id}/result",
            response_model=LLMResultResponse,
            summary="LLM 작업 결과 조회",
            description="작업 ID로 LLM 처리 결과를 조회합니다.")
async def get_task_result(
    task_id: str,
    db: DatabaseClientDep
):
    """task_id를 사용하여 LLM 작업 결과를 조회합니다."""
    
    result_key = f"llm_result:{task_id}"
    
    try:
        result = await db.get(result_key)
        
        if result:
            # 결과가 있는 경우
            if "error" in result:
                return LLMResultResponse(
                    task_id=task_id,
                    status="failed",
                    error=result.get("error"),
                    result=result
                )
            else:
                return LLMResultResponse(
                    task_id=task_id,
                    status="completed",
                    result=result
                )
        else:
            # 결과가 없는 경우 - 처리 중이거나 존재하지 않음
            return LLMResultResponse(
                task_id=task_id,
                status="processing",
                result=None
            )
            
    except Exception as e:
        logger.error(f"LLM 결과 조회 실패: task_id={task_id}, error={e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task result: {str(e)}"
        )

# === 큐 상태 조회 ===

@router.get("/queue/status",
            summary="큐 상태 조회",
            description="LLM 요청 큐의 현재 상태를 조회합니다.")
async def get_queue_status(db: DatabaseClientDep):
    """LLM 요청 큐의 현재 상태를 반환합니다."""
    
    try:
        queue_length = await db.llen("llm_request_queue")
        
        return {
            "queue_name": "llm_request_queue",
            "pending_tasks": queue_length or 0,
            "status": "active"
        }
        
    except Exception as e:
        logger.error(f"큐 상태 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue status: {str(e)}"
        )
