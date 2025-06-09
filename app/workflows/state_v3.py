# app/workflows/state_v3.py
from __future__ import annotations
import random
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

class TopicClarificationPydanticState(BaseModel):
    draft: str = ""
    question: Optional[str] = None
    answers: List[str] = Field(default_factory=list)
    is_final: bool = False
    trusted_domains: List[str] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    potential_intents: Optional[List[Dict[str, str]]] = None # LLM이 생성한 의도 후보 목록 (예: [{'id': '1', 'description': '...'}])
    chosen_intent_description: Optional[str] = None         # 사용자가 선택한 최종 의도 설명
    intent_clarification_question_outstanding: bool = False # 사용자에게 의도 선택 질문을 하고 답변 대기 중인지 여부

class ReportPlanningPydanticState(BaseModel):
    structure: Dict[str, Any] = Field(default_factory=dict)
    query_analysis: Optional[Dict] = None
    planning_question: Optional[str] = None
    planning_answer: Optional[str] = None
    outline_candidates: List[Dict] = []
    is_ready: bool = False

class SourceGroupPydantic(BaseModel):
    id: int
    urls: List[str] = Field(default_factory=list)
    rep_title: str = ""
    summary: str = ""
    structured_data: Optional[List[Dict[str, Any]]] = None

class SourceCollectPydanticState(BaseModel):
    """
        N03 Source Collect 노드의 상태를 관리합니다.
        검색된 소스의 전체 텍스트와 메타데이터를 구조화하여 저장합니다.
    """
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="""검색 및 전체 텍스트 추출 결과를 담는 딕셔너리. 
        Key는 대목차, Value는 처리된 아티클 정보 리스트.
        예시: {"대목차 제목": [{"source_url": "...", "language": "ko", "full_text": "...", "text_length": 1234}]}
        """
    )
    is_ready: bool = Field(False, description="N03 노드의 모든 작업이 성공적으로 완료되었는지 여부.")

class ReportDraftingPydanticState(BaseModel):
    """
    N04 Report Drafting 노드의 상태를 관리합니다.
    최종적으로 생성된 보고서 초안을 저장합니다.
    """
    draft: str = Field("", description="모든 섹션이 통합된 최종 보고서 초안 (마크다운 형식)")
    is_ready: bool = Field(False, description="N04 노드의 보고서 초안 작성이 완료되었는지 여부.")

class Opinion(BaseModel):
    """
    페르소나가 생성한 단일 의견 모델
    """
    persona_id: str = Field(description="의견을 생성한 페르소나의 ID")
    persona_name: str = Field(description="페르소나의 이름")
    opinion_text: str = Field(description="생성된 의견 텍스트 (한국어)")

class PersonaAnalysisState(BaseModel):
    """
    N05 페르소나 분석 노드의 상태 관리 모델
    """
    # 초기 4개의 의견 후보를 저장
    opinion_candidates: List[Opinion] = Field(default_factory=list, description="사용자에게 제시될 4개의 초기 의견 후보")
    question: Optional[str] = Field(None, description="사용자에게 보낼 질문 (의견 선택 또는 최종 확인)")
    selected_opinion: Optional[Opinion] = Field(None, description="사용자가 선택하고 수정한 최종 의견")
    is_ready: bool = Field(False, description="모든 과정이 완료되어 최종 의견이 확정되었는지 여부")

    # 에러 메시지
    error_message: Optional[str] = None


class ImageConcept(BaseModel):
    """
    단일 이미지 콘셉트에 대한 구조화된 모델
    """
    panel_id: int = Field(description="패널 번호 (1-4)")
    narrative_step: str = Field(description="콘셉트의 서사 단계 (예: 기(起): 문제 제기)")
    concept_description: str = Field(description="이미지에 표현될 구체적인 시각 요소와 장면에 대한 상세 설명")
    caption: str = Field(description="이미지에 오버레이될 짧고 임팩트 있는 문구")

class ImageConceptState(BaseModel):
    """
    N06 이미지 콘셉트 생성 노드의 상태 관리 모델 (대화형)
    """
    # 초기 4개의 콘셉트 후보를 저장
    concept_candidates: List[ImageConcept] = Field(default_factory=list, description="사용자에게 제시될 4개의 초기 이미지 콘셉트 후보")
    question: Optional[str] = Field(None, description="사용자에게 보낼 질문 (콘셉트 선택 또는 최종 확인)")
    wip_concepts: Dict[str, Any] = Field(default_factory=dict, description="사용자가 수정을 진행 중인 콘셉트 정보 (예: {'original': ImageConcept, 'feedback_history': []})")
    final_concepts: List[ImageConcept] = Field(default_factory=list, description="사용자가 최종 확정한 4개의 이미지 콘셉트")
    is_ready: bool = Field(False, description="이미지 콘셉트 생성이 모두 완료되었는지 여부")
    error_message: Optional[str] = None


class ImagePromptItemPydantic(BaseModel):
    """개별 패널에 대한 이미지 생성 프롬프트 정보"""
    panel_id: int
    prompt: str = Field(description="AI 이미지 생성기용 메인 프롬프트 (영어)")
    negative_prompt: str = Field(default="text, watermark, ugly, deformed, blurry", description="제외할 요소들에 대한 프롬프트 (영어)")
    seed: int = Field(default_factory=lambda: random.randint(1, 2 ** 32 - 1))
    width: int = Field(default=1024)
    height: int = Field(default=1024)


class ImagePromptsPydanticState(BaseModel):
    """
    N07 이미지 프롬프트 생성 노드의 상태 관리 모델 (대화형)
    """
    prompt_candidates: List[ImagePromptItemPydantic] = Field(default_factory=list)
    question: Optional[str] = None
    panels: List[ImagePromptItemPydantic] = Field(default_factory=list)
    is_ready: bool = False
    error_message: Optional[str] = None

class ImageQueueState(BaseModel):
    """
    N08 이미지 생성 큐 저장 노드의 상태 관리 모델
    """
    is_ready: bool = Field(False, description="이미지 생성 작업이 큐에 성공적으로 저장되었는지 여부")
    job_id: Optional[int] = Field(None, description="DB에 생성된 작업의 고유 ID")
    error_message: Optional[str] = None

class OverallWorkflowState(BaseModel):
    user_query: str
    work_id: Optional[str] = None

    topic_clarification: TopicClarificationPydanticState = Field(default_factory=TopicClarificationPydanticState)
    report_planning: ReportPlanningPydanticState = Field(default_factory=ReportPlanningPydanticState)
    source_collect: SourceCollectPydanticState = Field(default_factory=SourceCollectPydanticState)
    report_draft: ReportDraftingPydanticState = Field(default_factory=ReportDraftingPydanticState)
    persona_analysis: PersonaAnalysisState = Field(default_factory=PersonaAnalysisState)
    image_concept: ImageConceptState = Field(default_factory=ImageConceptState)
    image_prompts: ImagePromptsPydanticState = Field(default_factory=ImagePromptsPydanticState)
    insert_image_queue: ImageQueueState = Field(default_factory=ImageQueueState)

    current_node_name: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        # model_config Pydantic V2에서 사용, V1에서는 그냥 Config
        extra = "allow" # Pydantic V1 스타일
        validate_assignment = True # Pydantic V1 스타일
        # Pydantic V2
        # model_config = {
        #     "extra": "allow",
        #     "validate_assignment": True,
        # }

    def get_node_state_dict(self, node_key: str) -> Dict[str, Any]:
        node_state_model = getattr(self, node_key, None)
        if isinstance(node_state_model, BaseModel):
            return node_state_model.model_dump(exclude_none=True)
        elif isinstance(node_state_model, dict):
            return node_state_model
        return {}

    def update_node_state_from_dict(self, node_key: str, data: Dict[str, Any]):# 이 메서드는 Pydantic 모델의 setattr 기능을 활용하여 더 간단하게 만들 수 있으나,
        # 여기서는 명시적으로 각 타입에 따라 처리하는 기존 방식을 유지합니다.
        # 실제 사용 시에는 self.model_validate() 등을 고려할 수 있습니다 (Pydantic v2).
        node_state_field = getattr(self, node_key, None)
        if node_state_field is not None and isinstance(node_state_field, BaseModel):
            field_type = type(node_state_field)
            try:
                updated_model = field_type(**data)
                setattr(self, node_key, updated_model)
            except ValidationError as e:
                print(f"Validation error updating {node_key}: {e}")
        elif node_key == "topic_clarification":
            self.topic_clarification = TopicClarificationPydanticState(**data)
        elif node_key == "report_planning":
            self.report_planning = ReportPlanningPydanticState(**data)
        elif node_key == "source_collect":
            self.source_collect = SourceCollectPydanticState(**data)
        elif node_key == "report_draft":
            self.report_draft = ReportDraftingPydanticState(**data)
        elif node_key == "persona_analysis":
            self.persona_analysis = PersonaAnalysisState(**data)
        # elif node_key == "community_collect":
        #     self.community_collect = CommunityCollectPydanticState(**data)
        # elif node_key == "community_analysis":
        #     self.community_analysis = CommunityAnalysisPydanticState(**data)
        # elif node_key == "comic_outline":
        #     self.comic_outline = ComicOutlinePydanticState(**data)
        # elif node_key == "panel_details":
        #     self.panel_details = PanelDetailPydanticState(**data)
        # elif node_key == "image_prompts":
        #     self.image_prompts = ImagePromptsPydanticState(**data)  # [추가]
        else:
            pass