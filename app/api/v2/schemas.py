# ai/app/api/v2/schemas.py (수정 예시)
from pydantic import BaseModel, Field
from typing import Optional

# main_workflow.py에서 정의된 새로운 라우터 키 (예시, 실제 값과 일치시켜야 함)
# 이 값들은 API 클라이언트가 target_status로 전송할 수 있는 유효한 값들입니다.
VALID_TARGET_STATUSES = [
    "TOPIC_CLARIFICATION_N01", # 예시: 기존 별칭 -> 실제 노드명
    "REPORT_PLANNING_N02",
    "SEARCH_EXECUTION_N03",
    "REPORT_SYNTHESIS_N04",
    "COMMUNITY_PLANNING_N05",
    "COMMUNITY_EXECUTION_N06",
    "COMMUNITY_ANALYSIS_N07",
    "COMIC_OUTLINE_N08",
    "PANEL_DETAIL_N09",
    "IMAGE_PROMPT_N10",
    "CHAT",                        # 단순 채팅 메시지 처리 (워크플로우 진행 없음)
    "PAUSE"                        # 워크플로우 일시 중지 (특수 명령)
]

class ChatRequestPayload(BaseModel):
    request_id: str = Field(..., description="요청 고유 ID. 클라이언트에서 생성 또는 서버에서 생성 후 응답에 포함 가능.")
    room_id: str = Field(..., description="채팅방 ID. 이 ID를 기준으로 대화 상태가 관리될 수 있습니다.")
    user_id: str = Field(..., description="사용자 닉네임 또는 고유 식별자.")
    message: Optional[str] = Field(None, description="사용자가 입력한 채팅 메시지. 상태에 대한 피드백이나 다음 단계 지시 등에 사용될 수 있습니다.")
    target_status: str = Field(
        ...,
        description=f"실행을 요청할 워크플로우의 목표 상태 또는 다음 단계 식별자. 유효한 값 예시: {VALID_TARGET_STATUSES}",
        examples=VALID_TARGET_STATUSES
    )
    work_id: Optional[str] = Field(
        None,
        description="진행 중인 워크플로우의 고유 ID (state.meta.work_id에 해당). 새 워크플로우 시작 시에는 비워두거나, 클라이언트가 생성한 초기 ID를 전달할 수 있습니다."
    )

class ChatResponse(BaseModel): # API 응답 스키마 예시
    message: str = Field(..., description="처리 결과 또는 다음 안내 메시지.")
    request_id: str = Field(..., description="처리된 요청의 ID.")
    work_id: Optional[str] = Field(None, description="현재 처리 중이거나 생성된 워크플로우의 work_id (구 comic_id).")
    current_overall_status: Optional[str] = Field(None, description="워크플로우의 현재 전반적인 상태 (예: WAITING, PROCESSING, COMPLETED_STEP_X, FAILED).")
    # 필요시 추가적인 상태 정보나 다음 단계 제안 포함 가능
