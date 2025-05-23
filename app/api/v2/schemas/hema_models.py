# ai/app/api/v2/schemas/hema_models.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# --- HEMA 데이터 상태 및 타입 열거형 ---

class InformationSnippetStatus(str, Enum):
    """정보 조각 상태"""
    RAW = "raw"
    SUMMARIZED = "summarized"
    VERIFIED = "verified"
    OUTDATED = "outdated"

class IdeaNodeStatus(str, Enum):
    """아이디어 노드 상태"""
    PROPOSED = "proposed"
    DISCUSSED = "discussed"
    CONFIRMED = "confirmed"
    DISCARDED = "discarded"
    MERGED = "merged"

class IdeaNodeType(str, Enum):
    """아이디어 노드 타입"""
    CHARACTER = "character"
    PLOT_POINT = "plot_point"
    SETTING = "setting"
    THEME = "theme"
    DIALOGUE = "dialogue"
    VISUAL_ELEMENT = "visual_element"

class SummaryNodeType(str, Enum):
    """요약 노드 타입"""
    CONVERSATION_TOPIC = "conversation_topic"
    DOCUMENT_COLLECTION = "document_collection"
    IDEA_CLUSTER = "idea_cluster"
    SESSION_SUMMARY = "session_summary"

class InteractionEventType(str, Enum):
    """상호작용 이벤트 타입"""
    USER_QUERY_TO_SLM = "user_query_to_slm"
    SLM_RESPONSE_PROCESSED = "slm_response_processed"
    IDEA_CONFIRMED = "idea_confirmed"
    IDEA_MODIFIED = "idea_modified"
    SNIPPET_ADDED = "snippet_added"
    SNIPPET_SUMMARIZED = "snippet_summarized"
    CONTEXT_CONSTRUCTED = "context_constructed"
    SUMMARY_GENERATED = "summary_generated"

# --- HEMA 데이터 스키마 ---

class HEMAInternalInteractionLogSchema(BaseModel):
    """HEMA 내부 상호작용 로그 스키마"""
    log_id: str = Field(..., description="로그 고유 ID")
    session_id: str = Field(..., description="세션 ID")
    user_id: str = Field(..., description="사용자 ID")
    timestamp: datetime = Field(..., description="이벤트 발생 시간")
    event_type: InteractionEventType = Field(..., description="이벤트 타입")
    content_summary: str = Field(..., description="이벤트 내용 요약")
    linked_hema_ids: List[str] = Field(default_factory=list, description="연관된 HEMA 데이터 ID 목록")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="추가 메타데이터")
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class InformationSnippetSchema(BaseModel):
    """정보 조각 스키마"""
    snippet_id: str = Field(..., description="정보 조각 고유 ID")
    session_id: str = Field(..., description="세션 ID")
    user_id: str = Field(..., description="사용자 ID")
    source_type: str = Field(..., description="정보 출처 타입 (news, web, user_provided, research_paper)")
    title: str = Field(..., description="정보 제목")
    url: Optional[str] = Field(None, description="정보 출처 URL")
    summary_text: str = Field(..., description="정보 요약 텍스트")
    keywords: List[str] = Field(default_factory=list, description="키워드 목록")
    status: InformationSnippetStatus = Field(default=InformationSnippetStatus.RAW, description="처리 상태")
    relevance_score: Optional[float] = Field(None, description="관련도 점수 (0.0-1.0)")
    timestamp_added: datetime = Field(..., description="추가 시간")
    timestamp_updated: Optional[datetime] = Field(None, description="최종 수정 시간")
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class IdeaNodeSchema(BaseModel):
    """아이디어 노드 스키마"""
    idea_id: str = Field(..., description="아이디어 고유 ID")
    session_id: str = Field(..., description="세션 ID")
    user_id: str = Field(..., description="사용자 ID")
    node_type: IdeaNodeType = Field(..., description="아이디어 노드 타입")
    title: str = Field(..., description="아이디어 제목")
    description: str = Field(..., description="아이디어 상세 설명")
    status: IdeaNodeStatus = Field(default=IdeaNodeStatus.PROPOSED, description="아이디어 상태")
    version: int = Field(default=1, description="버전 번호")
    linked_snippet_ids: List[str] = Field(default_factory=list, description="연관된 정보 조각 ID 목록")
    parent_idea_id: Optional[str] = Field(None, description="부모 아이디어 ID")
    child_idea_ids: List[str] = Field(default_factory=list, description="자식 아이디어 ID 목록")
    tags: List[str] = Field(default_factory=list, description="태그 목록")
    confidence_score: Optional[float] = Field(None, description="신뢰도 점수 (0.0-1.0)")
    timestamp_created: datetime = Field(..., description="생성 시간")
    timestamp_updated: datetime = Field(..., description="최종 수정 시간")
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class SummaryNodeSchema(BaseModel):
    """요약 노드 스키마"""
    summary_id: str = Field(..., description="요약 고유 ID")
    session_id: str = Field(..., description="세션 ID")
    user_id: str = Field(..., description="사용자 ID")
    summary_type: SummaryNodeType = Field(..., description="요약 타입")
    title: str = Field(..., description="요약 제목")
    summary_text: str = Field(..., description="요약 텍스트")
    source_ids: List[str] = Field(default_factory=list, description="요약 대상 원본 데이터 ID 목록")
    keywords: List[str] = Field(default_factory=list, description="키워드 목록")
    compression_ratio: Optional[float] = Field(None, description="압축률 (원본 대비)")
    timestamp_generated: datetime = Field(..., description="생성 시간")
    validity_until: Optional[datetime] = Field(None, description="유효 기간")
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# --- HEMA 연산 및 Bulk API 관련 스키마 ---

class HEMAOperationType(str, Enum):
    """HEMA 데이터 연산 타입"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"

class HEMAEntityType(str, Enum):
    """HEMA 엔티티 타입"""
    INTERACTION_LOG = "HEMAInternalInteractionLog"
    INFORMATION_SNIPPET = "InformationSnippet"
    IDEA_NODE = "IdeaNode"
    SUMMARY_NODE = "SummaryNode"

class HEMABulkOperation(BaseModel):
    """HEMA Bulk 연산 스키마"""
    operation_id: str = Field(..., description="연산 고유 ID")
    action: HEMAOperationType = Field(..., description="연산 타입")
    entity_type: HEMAEntityType = Field(..., description="대상 엔티티 타입")
    entity_id: Optional[str] = Field(None, description="대상 엔티티 ID (UPDATE, DELETE 시 필요)")
    data: Optional[Dict[str, Any]] = Field(None, description="연산 데이터 (CREATE, UPDATE 시 필요)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="연산 메타데이터")

class HEMABulkOperationRequest(BaseModel):
    """HEMA Bulk 연산 요청 스키마"""
    user_id: str = Field(..., description="사용자 ID")
    session_id: str = Field(..., description="세션 ID")
    operations: List[HEMABulkOperation] = Field(..., description="연산 목록")
    request_id: Optional[str] = Field(None, description="요청 ID")

class HEMABulkOperationResult(BaseModel):
    """HEMA Bulk 연산 결과 스키마"""
    operation_index: int = Field(..., description="연산 인덱스")
    operation_id: str = Field(..., description="연산 ID")
    status: str = Field(..., description="처리 상태 (success, failure, partial)")
    entity_id: Optional[str] = Field(None, description="처리된 엔티티 ID")
    error_message: Optional[str] = Field(None, description="오류 메시지")
    
class HEMABulkOperationResponse(BaseModel):
    """HEMA Bulk 연산 응답 스키마"""
    request_id: str = Field(..., description="요청 ID")
    status: str = Field(..., description="전체 처리 상태 (success, partial, failure)")
    results: List[HEMABulkOperationResult] = Field(..., description="연산 결과 목록")
    processed_count: int = Field(..., description="처리된 연산 수")
    failed_count: int = Field(..., description="실패한 연산 수")
    timestamp: datetime = Field(..., description="처리 완료 시간")
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# --- HEMA 컨텍스트 구성 관련 스키마 ---

class HEMAContextItem(BaseModel):
    """HEMA 컨텍스트 항목"""
    item_id: str = Field(..., description="항목 ID")
    item_type: HEMAEntityType = Field(..., description="항목 타입")
    content: str = Field(..., description="항목 내용")
    relevance_score: float = Field(..., description="관련도 점수")
    token_count: int = Field(..., description="토큰 수")
    priority: int = Field(default=1, description="우선순위 (1: 높음, 3: 낮음)")

class HEMAContextSummary(BaseModel):
    """HEMA 컨텍스트 요약"""
    total_items: int = Field(..., description="전체 항목 수")
    total_tokens: int = Field(..., description="전체 토큰 수")
    items_by_type: Dict[str, int] = Field(..., description="타입별 항목 수")
    average_relevance: float = Field(..., description="평균 관련도")
    context_quality_score: float = Field(..., description="컨텍스트 품질 점수")

class HEMAContext(BaseModel):
    """HEMA 컨텍스트"""
    context_id: str = Field(..., description="컨텍스트 ID")
    user_id: str = Field(..., description="사용자 ID")
    session_id: str = Field(..., description="세션 ID")
    query: str = Field(..., description="사용자 쿼리")
    items: List[HEMAContextItem] = Field(..., description="컨텍스트 항목 목록")
    summary: HEMAContextSummary = Field(..., description="컨텍스트 요약")
    timestamp_created: datetime = Field(..., description="생성 시간")
    
    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
