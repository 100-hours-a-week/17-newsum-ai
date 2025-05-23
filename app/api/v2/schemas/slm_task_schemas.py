# ai/app/api/v2/schemas/slm_task_schemas.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class SLMTaskPriority(str, Enum):
    """SLM 작업 우선순위"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

class SLMTaskType(str, Enum):
    """SLM 작업 타입"""
    INTERACTIVE_QA = "interactive_qa"
    IDEA_GENERATION = "idea_generation"
    LONG_SUMMARY = "long_summary"
    CONTEXT_ANALYSIS = "context_analysis"
    CREATIVE_WRITING = "creative_writing"

class SLMTaskRequest(BaseModel):
    """SLM 작업 요청 스키마"""
    request_id: str = Field(..., description="요청 고유 ID")
    task_type: SLMTaskType = Field(default=SLMTaskType.INTERACTIVE_QA, description="작업 타입")
    priority: SLMTaskPriority = Field(default=SLMTaskPriority.NORMAL, description="작업 우선순위")
    slm_payload: Dict[str, Any] = Field(..., description="LLMService.generate_text에 전달될 파라미터")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="작업 메타데이터")
    timeout: Optional[int] = Field(default=30, description="작업 타임아웃 (초)")
    retry_count: Optional[int] = Field(default=0, description="재시도 횟수")
    timestamp_created: datetime = Field(default_factory=datetime.now, description="요청 생성 시간")
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

class SLMTaskResponse(BaseModel):
    """SLM 작업 응답 스키마"""
    request_id: str = Field(..., description="요청 ID")
    generated_text: Optional[str] = Field(None, description="생성된 텍스트")
    error: Optional[str] = Field(None, description="오류 메시지")
    raw_response: Optional[Dict[str, Any]] = Field(None, description="원본 LLM 응답 (디버깅용)")
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")
    token_usage: Optional[Dict[str, int]] = Field(None, description="토큰 사용량 정보")
    quality_score: Optional[float] = Field(None, description="응답 품질 점수")
    timestamp_completed: datetime = Field(default_factory=datetime.now, description="처리 완료 시간")
    worker_id: Optional[str] = Field(None, description="처리한 워커 ID")
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}

class SLMTaskStatus(BaseModel):
    """SLM 작업 상태 스키마"""
    request_id: str = Field(..., description="요청 ID")
    status: str = Field(..., description="작업 상태 (pending, processing, completed, failed, timeout)")
    progress: Optional[float] = Field(None, description="진행률 (0.0-1.0)")
    estimated_completion: Optional[datetime] = Field(None, description="예상 완료 시간")
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
