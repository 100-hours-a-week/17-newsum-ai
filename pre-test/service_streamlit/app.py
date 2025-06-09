import streamlit as st
import db_utils
import os
import uuid
import time
import aiohttp
import asyncio
import json
import re
from typing import Optional, Dict, Set, Any

# --- 초기 설정 ---
st.set_page_config(page_title="SLM Chat Service", layout="wide")

# --- 모델 및 유틸리티 임포트 ---
try:
    from state_v2 import WorkflowState
    import state_view_utils

    STATE_MODULE_LOADED = True
except ImportError:
    print("WARNING: state_v2.py or state_view_utils.py not found. WorkflowState display will use dummy data/functions.")


    class WorkflowState:
        pass


    def format_workflow_state_to_markdown(state: Any) -> str:
        return "워크플로우 상태 표시 (모듈 로드 실패)"


    STATE_MODULE_LOADED = False

# --- 상수 및 설정 ---
AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://localhost:8000/api/v2/chat/workflow")

TOGGLE_ORDER = [
    'toggle_TOPIC_CLARIFICATION',
    'toggle_REPORT_PLANNING',
    'toggle_SEARCH_EXECUTION',
    'toggle_REPORT_SYNTHESIS',
    'toggle_PERSONA_ANALYSIS',
    'toggle_OPINION_TO_IMAGE_CONCEPT',
    'toggle_CONCEPT_TO_PROMPT',
    'toggle_SAVE_IN_QUEUE',
    'toggle_PANEL_DETAIL',
    'toggle_IMAGE_PROMPT']

TOGGLE_LABELS = {
    'toggle_TOPIC_CLARIFICATION': '토픽 선정',
    'toggle_REPORT_PLANNING': '토픽 조사 계획',
    'toggle_SEARCH_EXECUTION': '검색 수행',
    'toggle_REPORT_SYNTHESIS': '보고서 작성',
    'toggle_PERSONA_ANALYSIS': '페르소나 랜덤선택&의견 생성',
    'toggle_OPINION_TO_IMAGE_CONCEPT': '이미지 콘셉 생성',
    'toggle_CONCEPT_TO_PROMPT': '이미지 프롬프트로 변환',
    'toggle_SAVE_IN_QUEUE': '이미지 생성 큐 저장',

    'toggle_PANEL_DETAIL': '이미지 생성',
    'toggle_IMAGE_PROMPT': '게시글 작성',
}

CHAT_OPTION_LABEL = "💬 일반 채팅 (선택 없음)"
RADIO_OPTIONS_LABELS = [CHAT_OPTION_LABEL] + [TOGGLE_LABELS[key] for key in TOGGLE_ORDER]
LABEL_TO_TOGGLE_KEY = {v: k for k, v in TOGGLE_LABELS.items()}

TOGGLE_KEY_TO_TARGET_STATUS = {
    'toggle_TOPIC_CLARIFICATION': 'TOPIC_CLARIFICATION_N01',
    'toggle_REPORT_PLANNING': 'REPORT_PLANNING_N02',
    'toggle_SEARCH_EXECUTION': 'SEARCH_EXECUTION_N03',
    'toggle_REPORT_SYNTHESIS': 'REPORT_SYNTHESIS_N04',
    'toggle_PERSONA_ANALYSIS': 'PERSONA_ANALYSIS_N05',
    'toggle_OPINION_TO_IMAGE_CONCEPT': 'OPINION_TO_IMAGE_CONCEPT_N06',
    'toggle_CONCEPT_TO_PROMPT': 'CONCEPT_TO_PROMPT_N07',
    'toggle_SAVE_IN_QUEUE': 'SAVE_IN_QUEUE_N08',

    'toggle_PANEL_DETAIL': 'PANEL_DETAIL_N09',
    'toggle_IMAGE_PROMPT': 'IMAGE_PROMPT_N10',
}
TARGET_STATUS_TO_TOGGLE_KEY = {v: k for k, v in TOGGLE_KEY_TO_TARGET_STATUS.items()}


# --- 세션 상태 초기화 ---
def init_session_state():
    defaults = {
        'logged_in': False, 'user_id': None, 'nickname': "",
        'current_room_id': None, 'current_room_name': "",
        'current_work_id': None,
        'is_loading': False, 'messages': [],
        'room_to_delete': None, 'room_name_to_delete': None,
        'warned_missing_state_for_toggles': set(),
        # st.radio 위젯의 상태는 해당 위젯의 key로 st.session_state에 자동 저장됨.
        # 예: st.session_state[f"workflow_radio_{work_id}"]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# --- 유틸리티 함수 ---

def determine_overall_status_from_dict(status_dict: Optional[Dict[str, str]]) -> str:
    if not status_dict or not isinstance(status_dict, dict):
        return 'STARTED'
    # 중요: status_criteria는 실제 워크플로우 단계 정의에 따라 정확하게 설정되어야 합니다.
    status_criteria: Dict[str, Set[str]] = {
        'IMAGE_SCHEDULING_N09': {"1", "2", "3", "4", "5", "6", "7", "8", "9"},
        'PROMPT_REFINEMENT_N08A': {"1", "2", "3", "4", "5", "6", "7", "8"},
        'SCENARIO_GENERATION_N08': {"1", "2", "3", "4", "5", "6", "7", "8"},
        'COMIC_IDEATION_TARGET': {"1", "2", "3", "4", "5", "6", "7"},
        'SATIRE_PREPARATION_UPTO_N06C': {"1", "2", "3", "4", "5", "6"},
        'SEARCH_AND_REPORT_UPTO_N05': {"1", "2", "3", "4", "5"},
        'INITIAL_PROCESSING_UPTO_N03': {"1", "2", "3"},
    }
    ordered_statuses = [
        'IMAGE_SCHEDULING_N09', 'PROMPT_REFINEMENT_N08A', 'SCENARIO_GENERATION_N08',
        'COMIC_IDEATION_TARGET', 'SATIRE_PREPARATION_UPTO_N06C',
        'SEARCH_AND_REPORT_UPTO_N05', 'INITIAL_PROCESSING_UPTO_N03',
    ]
    for status_name in ordered_statuses:
        required_steps = status_criteria[status_name]
        if all(status_dict.get(step_key) == "COMPLETED" for step_key in required_steps):
            return status_name
    return 'STARTED'


def parse_think_answer(content: str) -> tuple[Optional[str], str]:
    think_content, answer_content = None, content
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        answer_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    answer_match = re.search(r"\[ANSWER\](.*)", answer_content, re.DOTALL)
    if answer_match: answer_content = answer_match.group(1).strip()
    if not answer_content and think_content:
        answer_content = "(답변 내용이 없습니다.)"
    elif not answer_content and not think_content:
        answer_content = content
    return think_content, answer_content


async def call_ai_server(user_message: str, room_id: int, user_id: int, request_id_uuid: uuid.UUID, target_status: str,
                         work_id: Optional[uuid.UUID]):
    payload = {
        "room_id": str(room_id), "user_id": str(user_id), "request_id": str(request_id_uuid),
        "message": user_message, "target_status": target_status, "work_id": str(work_id) if work_id else None
    }
    try:
        print(f"Attempting to send payload to AI server: {payload}")
        json_data = json.dumps(payload)
        headers = {'Content-Type': 'application/json'}
        async with aiohttp.ClientSession() as session:
            async with session.post(AI_SERVER_URL, data=json_data, headers=headers) as response:
                if response.status == 202:
                    st.toast("✅ AI 작업이 예약되었습니다.")
                else:
                    response_text = await response.text()
                    print(f"Failed to send request to AI server. Status: {response.status}, Response: {response_text}")
                    st.error(f"❌ AI 서버 오류 ({response.status}): {response_text}")
                    db_utils.update_loading_message(request_id_uuid, f"AI 서버 호출 실패: {response.status}")
    except Exception as e:
        print(f"Error calling AI server: {e}")
        st.error(f"❌ AI 서버 연결 오류: {str(e)}")
        db_utils.update_loading_message(request_id_uuid, f"AI 서버 연결 오류: {e}")


def handle_confirm_delete():
    if st.session_state.room_to_delete and st.session_state.user_id:
        deleted = db_utils.delete_chatroom(st.session_state.room_to_delete, st.session_state.user_id)
        if deleted:
            st.sidebar.success(f"'{st.session_state.room_name_to_delete}' 채팅방 삭제 완료.")
            if st.session_state.current_room_id == st.session_state.room_to_delete:
                st.session_state.current_room_id = None
                st.session_state.current_room_name = ""
                st.session_state.current_work_id = None
                st.session_state.messages = []
                # 만약 st.radio 위젯 키가 work_id를 포함한다면, 해당 work_id가 사라지므로
                # 다음 방 선택 시 새 키로 위젯이 생성되어 별도 정리 불필요.
        else:
            st.sidebar.error("채팅방 삭제 실패.")
    st.session_state.room_to_delete = None
    st.session_state.room_name_to_delete = None
    st.rerun()


def handle_cancel_delete():
    st.session_state.room_to_delete = None
    st.session_state.room_name_to_delete = None
    st.rerun()


def initialize_radio_button_state(work_id: uuid.UUID):
    """
    DB 상태를 기반으로 현재 work_id의 st.radio 위젯 기본 선택값을 설정합니다.
    """
    current_state_data = db_utils.get_state(work_id)
    db_wf_status = 'STARTED'
    if current_state_data:  # DB에 해당 work_id에 대한 상태 데이터가 존재하는 경우
        if STATE_MODULE_LOADED:
            try:
                temp_state = WorkflowState(**current_state_data)
                if hasattr(temp_state, 'meta') and hasattr(temp_state.meta, 'workflow_status'):
                    if isinstance(temp_state.meta.workflow_status, dict):
                        db_wf_status = determine_overall_status_from_dict(temp_state.meta.workflow_status)
                    elif isinstance(temp_state.meta.workflow_status, str) and temp_state.meta.workflow_status:
                        db_wf_status = temp_state.meta.workflow_status
            except Exception as e:
                print(f"ERROR: initialize_radio - {e}")  # 간략화
        if work_id in st.session_state.get('warned_missing_state_for_toggles', set()):
            st.session_state.warned_missing_state_for_toggles.remove(work_id)
    else:  # DB에 데이터 없는 경우
        db_wf_status = 'STARTED'
        if work_id not in st.session_state.get('warned_missing_state_for_toggles', set()):
            st.session_state.warned_missing_state_for_toggles.add(work_id)
            print(f"INFO: No state data for {work_id} in initialize_radio. Defaulting.")

    active_toggle_key_from_db = TARGET_STATUS_TO_TOGGLE_KEY.get(db_wf_status)
    selected_radio_option = CHAT_OPTION_LABEL
    if active_toggle_key_from_db:
        selected_radio_option = TOGGLE_LABELS.get(active_toggle_key_from_db, CHAT_OPTION_LABEL)

    # radio_widget_key = f"workflow_radio_{work_id}" # 이 키는 위젯 생성 시 사용
    # st.session_state[radio_widget_key] = selected_radio_option # <--- 직접 할당 금지!

    # 대신, 다음번 st.radio 렌더링 시 사용할 "기본값"을 다른 세션 상태에 저장
    if 'intended_radio_options' not in st.session_state:
        st.session_state.intended_radio_options = {}
    st.session_state.intended_radio_options[work_id] = selected_radio_option
    # print(f"DEBUG initialize_radio_button_state: Intended radio option for work_id {work_id} set to '{selected_radio_option}'")\


def handle_radio_selection_change():
    work_id = st.session_state.current_work_id
    if not work_id: return
    radio_widget_key = f"workflow_radio_{work_id}"
    selected_option = st.session_state.get(radio_widget_key)
    # print(f"DEBUG handle_radio_selection_change: work_id '{work_id}', radio '{radio_widget_key}' selection changed to '{selected_option}'")
    # 현재는 특별한 추가 작업 없음. st.radio가 상태를 직접 관리.


# --- 렌더링 함수 ---
def render_login():
    st.title("🚀 SLM Chat Service")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("닉네임으로 로그인")
        nickname = st.text_input("닉네임을 입력하세요:", key="nickname_input_login", placeholder="예: 홍길동")
        if st.button("🔐 로그인", use_container_width=True):
            if nickname:
                user_id, nick = db_utils.get_or_create_user(nickname)
                if user_id:
                    st.session_state.logged_in, st.session_state.user_id, st.session_state.nickname = True, user_id, nick
                    st.success(f"환영합니다, {nick}님!")
                    time.sleep(1);
                    st.rerun()
                else:
                    st.error("로그인 실패.")
            else:
                st.warning("닉네임을 입력해주세요.")


def render_chat_management():
    st.sidebar.title(f"👋 {st.session_state.nickname}님!")
    st.sidebar.markdown("---")
    with st.sidebar.expander("➕ 새 채팅방 만들기", expanded=False):
        new_room_name = st.text_input("채팅방 이름:", key="new_room_name_input", placeholder="예: 새 프로젝트")
        if st.button("생성", use_container_width=True, key="create_room_btn"):
            if new_room_name:
                room_id = db_utils.create_chatroom(st.session_state.user_id, new_room_name)
                if room_id:
                    st.sidebar.success(f"'{new_room_name}' 생성 완료."); st.rerun()
                else:
                    st.sidebar.error("생성 실패.")
            else:
                st.sidebar.warning("이름 입력.")

    st.sidebar.subheader("💬 내 채팅방 목록")
    chatrooms = db_utils.get_user_chatrooms(st.session_state.user_id)
    if st.session_state.room_to_delete:
        st.sidebar.warning(f"'{st.session_state.room_name_to_delete}' 삭제하시겠습니까?")
        c1, c2 = st.sidebar.columns(2)
        c1.button("예", on_click=handle_confirm_delete, key="confirm_del_yes", type="primary", use_container_width=True)
        c2.button("아니요", on_click=handle_cancel_delete, key="confirm_del_no", use_container_width=True)

    for room in chatrooms:
        c1, c2 = st.sidebar.columns([3, 1])
        is_current = st.session_state.current_room_id == room['room_id']
        button_text = f"{'🟢 ' if is_current else '⚪ '}{room['room_name']}"
        if c1.button(button_text, key=f"room_btn_{room['room_id']}", use_container_width=True):
            if not st.session_state.room_to_delete and not is_current:
                st.session_state.current_room_id = room['room_id']
                st.session_state.current_room_name = room['room_name']
                st.session_state.current_work_id = db_utils.get_or_create_workflow(room['room_id'])
                st.session_state.messages = db_utils.get_messages(room['room_id'])
                if st.session_state.current_work_id:
                    initialize_radio_button_state(st.session_state.current_work_id)  # 라디오 버튼 상태 초기화
                    st.session_state.is_loading = db_utils.check_ai_processing(room['room_id'])
                else:
                    st.error("워크플로우 ID 처리 중 오류 발생.");
                    st.session_state.is_loading = False
                st.rerun()
        if c2.button("🗑️", key=f"del_btn_{room['room_id']}", use_container_width=True, help="채팅방 삭제"):
            if not st.session_state.room_to_delete:
                st.session_state.room_to_delete = room['room_id']
                st.session_state.room_name_to_delete = room['room_name']
                st.rerun()
    if not chatrooms: st.sidebar.info("📝 아직 생성된 채팅방이 없습니다.")
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 로그아웃", key="logout_btn", use_container_width=True):
        for key_to_del in list(st.session_state.keys()):
            del st.session_state[key_to_del]
        init_session_state()
        st.rerun()


def render_chat_interface():
    room_id = st.session_state.current_room_id
    work_id = st.session_state.current_work_id

    if not room_id or not work_id:
        st.warning("워크플로우가 초기화되지 않았습니다. 채팅방을 다시 선택해주세요.")
        return

    current_state_data = db_utils.get_state(work_id)
    current_state_obj = None
    db_workflow_status_str = 'STARTED'
    if current_state_data:
        if STATE_MODULE_LOADED:
            try:
                current_state_obj = WorkflowState(**current_state_data)
                if hasattr(current_state_obj, 'meta') and hasattr(current_state_obj.meta, 'workflow_status'):
                    if isinstance(current_state_obj.meta.workflow_status, dict):
                        db_workflow_status_str = determine_overall_status_from_dict(
                            current_state_obj.meta.workflow_status)
                    elif isinstance(current_state_obj.meta.workflow_status,
                                    str) and current_state_obj.meta.workflow_status:
                        db_workflow_status_str = current_state_obj.meta.workflow_status
            except Exception as e:
                st.error(f"WorkflowState 로드 실패 (work_id: {work_id}): {e}")

    radio_widget_key = f"workflow_radio_{work_id}"
    if radio_widget_key not in st.session_state:
        initialize_radio_button_state(work_id)

    col_title, col_status, col_menu_btn = st.columns([12, 4, 1])
    with col_title:
        st.title(f"💬 {st.session_state.current_room_name}")
    with col_status:
        st.markdown(
            f"<div style='text-align: right; padding-top: 25px;'>DB 상태: <strong>{db_workflow_status_str}</strong></div>",
            unsafe_allow_html=True)
    with col_menu_btn:
        st.write("")
        with st.popover("⚙️", use_container_width=False):
            st.markdown("**워크플로우 단계 선택**")
            options = RADIO_OPTIONS_LABELS
            current_selection_for_radio = st.session_state.get(radio_widget_key, CHAT_OPTION_LABEL)
            try:
                if current_selection_for_radio not in options:
                    current_selection_for_radio = CHAT_OPTION_LABEL
                    st.session_state[radio_widget_key] = current_selection_for_radio
                default_radio_index = options.index(current_selection_for_radio)
            except ValueError:
                default_radio_index = 0
                st.session_state[radio_widget_key] = options[0]

            st.radio(
                "작업 선택:",
                options=options,
                index=default_radio_index,
                key=radio_widget_key,
                on_change=handle_radio_selection_change,
                label_visibility="collapsed"
            )
            selected_option_for_description = st.session_state.get(radio_widget_key, CHAT_OPTION_LABEL)
            if selected_option_for_description == CHAT_OPTION_LABEL:
                st.caption(f"ℹ️ {CHAT_OPTION_LABEL}에서는 SLM과 자유롭게 대화하며, 대화 내용에 따라 실제 워크플로우 상태(State) 값들이 변경될 수 있습니다.")

    chat_col, right_col = st.columns([1, 1])
    with chat_col:
        st.subheader("대화 내용")
        msg_container = st.container(height=800, border=False)
        # ... (메시지 표시 로직은 이전과 동일) ...
        with msg_container:
            messages_to_display = st.session_state.get('messages', [])
            for msg in messages_to_display:
                role = "assistant" if msg['message_type'] in ['final_ai', 'system', 'loading'] else "user"
                avatar = "🤖" if role == "assistant" else ("🧑‍💻" if msg['message_type'] == 'user' else "ℹ️")
                with st.chat_message(role, avatar=avatar):
                    if msg['message_type'] == 'loading':
                        with st.expander("AI 응답 생성 중...", expanded=True):
                            st.write(msg['content']); st.spinner()
                    elif msg['message_type'] == 'final_ai':
                        think, answer = parse_think_answer(msg['content'])
                        if think:
                            with st.expander("생각 과정 🧠"): st.markdown(think)
                        st.markdown(answer)
                    else:
                        st.markdown(f"{msg['content']}")

    # with right_col:
    #     st.subheader("🛠️ 워크플로우 상태 (상세)")
    #     if current_state_obj and STATE_MODULE_LOADED:
    #         state_md = state_view_utils.format_workflow_state_to_markdown(current_state_obj)
    #         with st.container(height=800, border=True):
    #             st.markdown(state_md, unsafe_allow_html=True)
    #     elif STATE_MODULE_LOADED:
    #         st.info(f"상태 정보 없음 (DB에 work_id: {work_id} 데이터가 없거나 로드 실패).")
    #     else:
    #         st.warning("워크플로우 모듈(state_v2.py) 로드 실패. 상태 표시 불가.")
    with right_col:
        st.subheader("🛠️ 워크플로우 상태 (상세)")
        if current_state_data and STATE_MODULE_LOADED:
            # ① JSON 뷰어를 바로 사용
            with st.container(height=800, border=True):
                st.json(current_state_data)
        elif STATE_MODULE_LOADED:
            st.info(f"상태 정보 없음 (DB에 work_id: {work_id} 데이터가 없거나 로드 실패).")
        else:
            st.warning("워크플로우 모듈(state_v2.py) 로드 실패. 상태 표시 불가.")

    selected_option_for_placeholder = st.session_state.get(radio_widget_key, CHAT_OPTION_LABEL)
    if selected_option_for_placeholder != CHAT_OPTION_LABEL:
        toggle_key_for_placeholder = LABEL_TO_TOGGLE_KEY.get(selected_option_for_placeholder)
        if toggle_key_for_placeholder:
            target_status_of_selected_step = TOGGLE_KEY_TO_TARGET_STATUS.get(toggle_key_for_placeholder)
            if target_status_of_selected_step == db_workflow_status_str:
                chat_input_placeholder = f"'{selected_option_for_placeholder}' 단계 재실행 / (일반 채팅은 '{CHAT_OPTION_LABEL}' 선택)"
            else:
                chat_input_placeholder = f"'{selected_option_for_placeholder}' 단계 진행 / (일반 채팅은 '{CHAT_OPTION_LABEL}' 선택)"
        else:
            chat_input_placeholder = f"선택된 단계({selected_option_for_placeholder}) 진행 / (일반 채팅은 '{CHAT_OPTION_LABEL}' 선택)"
    else:
        chat_input_placeholder = f"메시지를 입력하세요 ({CHAT_OPTION_LABEL} - SLM과 대화하며 상태값 수정 가능)"

    st.session_state.is_loading = db_utils.check_ai_processing(room_id)
    prompt = st.chat_input(chat_input_placeholder, key="chat_input_main", disabled=st.session_state.is_loading)

    if prompt:
        user_id = st.session_state.user_id
        req_id = uuid.uuid4()
        selected_option_for_action = st.session_state.get(radio_widget_key, CHAT_OPTION_LABEL)

        if selected_option_for_action != CHAT_OPTION_LABEL:
            toggle_key_for_action = LABEL_TO_TOGGLE_KEY.get(selected_option_for_action)
            if toggle_key_for_action:
                target_status_for_api = TOGGLE_KEY_TO_TARGET_STATUS.get(toggle_key_for_action)
                if not target_status_for_api:
                    print(
                        f"ERROR: Action - Selected option '{selected_option_for_action}' (key: {toggle_key_for_action}) has no target status!")
                    target_status_for_api = 'CHAT'
            else:
                print(f"ERROR: Action - Cannot find toggle key for selected option '{selected_option_for_action}'!")
                target_status_for_api = 'CHAT'
        else:
            target_status_for_api = 'CHAT'

        print(
            f"DEBUG (Prompt Send): RadioSelected='{selected_option_for_action}', DB_State='{db_workflow_status_str}', Final_Target_API='{target_status_for_api}'")
        db_utils.add_message(room_id, user_id, prompt, 'user', req_id)

        if target_status_for_api == 'CHAT':
            loading_text = "AI가 답변을 준비하고 있습니다..."
            db_utils.add_message(room_id, user_id, loading_text, 'loading', req_id)
        else:
            status_label_for_loading = selected_option_for_action if selected_option_for_action != CHAT_OPTION_LABEL else "워크플로우 작업"
            loading_text = f"'{status_label_for_loading}' 단계 워크플로우 작업이 진행됩니다. \n (약 10초 ~ 1분가량 소요될수 있습니다.) \n\n 현재 시스템은 DB폴링방식으로 화면이 구현되어 별도의 새로고침이 필요할 수 있습니다."
            db_utils.add_message(room_id, user_id, loading_text, 'loading', req_id)

        st.session_state.messages = db_utils.get_messages(room_id)
        st.session_state.is_loading = True
        asyncio.run(call_ai_server(prompt, room_id, user_id, req_id, target_status_for_api, work_id))
        st.rerun()

    if st.session_state.is_loading:
        time.sleep(2)
        if not db_utils.check_ai_processing(st.session_state.current_room_id):
            st.session_state.messages = db_utils.get_messages(st.session_state.current_room_id)
            current_work_id_for_poll = st.session_state.current_work_id
            if current_work_id_for_poll:
                initialize_radio_button_state(current_work_id_for_poll)
            st.session_state.is_loading = False
            st.rerun()
        else:
            st.rerun()
    # if st.session_state.is_loading:
    #     count = st_autorefresh(interval=2000, key="loading_autorefresh")
    #     still_loading = db_utils.check_ai_processing(room_id)
    #     st.write(f"▶ 폴링 #{count}회 – check_ai_processing → {still_loading}")
    #     if not still_loading:
    #         st.write("▶ 처리 완료 감지! is_loading을 False로 전환합니다.")
    #         st.session_state.is_loading = False
    #         st.session_state.messages = db_utils.get_messages(room_id)
    #         initialize_radio_button_state(work_id)
    #         st.rerun()
    #     else:
    #         st.write("▶ 아직 로딩 중입니다...")
    # else:
    #     st.write("▶ is_loading=False – 정상 UI 코드로 진입해야 합니다.")

def main():
    if not st.session_state.logged_in:
        render_login()
    else:
        render_chat_management()
        if st.session_state.current_room_id:
            render_chat_interface()
        else:
            st.markdown("""
            <div style='text-align: center; padding: 50px;'>
                <h3>🎯 채팅을 시작하세요!</h3>
                <p>왼쪽 사이드바에서 채팅방을 선택하거나 새로 생성해주세요.</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()