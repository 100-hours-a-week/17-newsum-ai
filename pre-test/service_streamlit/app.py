import streamlit as st
import db_utils
import os
import uuid
import time
import aiohttp
import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Optional # Optional 임포트

# --- 모델 및 유틸리티 임포트 ---
# (주의: 실제 프로젝트 구조에 맞게 경로를 수정해야 합니다)
try:
    from state_v2 import WorkflowState, MetaSection, QuerySection, SearchSection, RawArticle, ReportSection
    import state_view_utils
    STATE_MODULE_LOADED = True
except ImportError:
    print("WARNING: state_v2.py or state_view_utils.py not found. WorkflowState display will use dummy data/functions.")
    # 임시 클래스/함수 정의 (오류 방지용)
    class WorkflowState: pass
    class MetaSection: pass
    class QuerySection: pass
    class SearchSection: pass
    class RawArticle: pass
    class ReportSection: pass
    def format_workflow_state_to_markdown(state): return "워크플로우 상태 표시 (모듈 로드 실패)"
    STATE_MODULE_LOADED = False


# --- 초기 설정 ---
st.set_page_config(page_title="SLM Chat Service", layout="wide")

# --- 세션 상태 초기화 ---
def init_session_state():
    defaults = {
        'logged_in': False, 'user_id': None, 'nickname': "",
        'current_room_id': None, 'current_room_name': "",
        'is_loading': False, 'messages': [],
        'room_to_delete': None, 'room_name_to_delete': None,
        'workflow_toggles': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state() # 세션 상태 초기화 호출

# --- 상수 및 설정 ---
AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://localhost:8000/api/v2/chat") # 기본값 설정
RECEPTION_API_PORT = os.getenv("RECEPTION_API_PORT", "9090")
RECEPTION_API_URL = f"http://localhost:{RECEPTION_API_PORT}/receive_response"

TOGGLE_ORDER = ['toggle_query', 'toggle_search', 'toggle_idea', 'toggle_scenario', 'toggle_image']
TOGGLE_LABELS = {
    'toggle_query': "쿼리", 'toggle_search': "검색 & 리포트", 'toggle_idea': "아이디어",
    'toggle_scenario': "시나리오", 'toggle_image': "이미지 생성 예약",
}
STATUS_TO_TOGGLES = {
    'IDLE':         {'toggle_query': False, 'toggle_search': False, 'toggle_idea': False, 'toggle_scenario': False, 'toggle_image': False},
    'QUERY_DONE':   {'toggle_query': True,  'toggle_search': False, 'toggle_idea': False, 'toggle_scenario': False, 'toggle_image': False},
    'SEARCH_DONE':  {'toggle_query': True,  'toggle_search': True,  'toggle_idea': False, 'toggle_scenario': False, 'toggle_image': False},
    'IDEA_DONE':    {'toggle_query': True,  'toggle_search': True,  'toggle_idea': True,  'toggle_scenario': False, 'toggle_image': False},
    'SCENARIO_DONE':{'toggle_query': True,  'toggle_search': True,  'toggle_idea': True,  'toggle_scenario': True,  'toggle_image': False},
    'IMAGE_DONE':   {'toggle_query': True,  'toggle_search': True,  'toggle_idea': True,  'toggle_scenario': True,  'toggle_image': True},
}
TOGGLES_TO_STATUS = {tuple(v.values()): k for k, v in STATUS_TO_TOGGLES.items()}

# --- 유틸리티 및 콜백 함수 ---

def parse_think_answer(content):
    think_content, answer_content = None, content
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        answer_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    answer_match = re.search(r"\[ANSWER\](.*)", answer_content, re.DOTALL)
    if answer_match: answer_content = answer_match.group(1).strip()
    if not answer_content and think_content: answer_content = "(답변 내용이 없습니다.)"
    elif not answer_content and not think_content: answer_content = content
    return think_content, answer_content

async def call_ai_server(user_message, room_id, user_id, request_id_uuid: uuid.UUID, target_status: Optional[str]):
    """AI 서버에 비동기적으로 요청을 보냅니다 (target_status 포함)."""
    payload = {
        "room_id": str(room_id),
        "user_id": str(user_id),
        "request_id": str(request_id_uuid),
        "message": user_message,
        "target_status": target_status # 목표 상태 추가
    }
    try:
        print(f"Attempting to send payload to AI server: {payload}")
        json_data = json.dumps(payload)
        headers = {'Content-Type': 'application/json'}
        async with aiohttp.ClientSession() as session:
            async with session.post(AI_SERVER_URL, data=json_data, headers=headers) as response:
                if response.status == 200 or response.status == 202:
                    print(f"Successfully sent request to AI server: {request_id_uuid}. Status: {response.status}")
                else:
                    response_text = await response.text()
                    print(f"Failed to send request to AI server. Status: {response.status}, Response: {response_text}")
                    db_utils.update_loading_message(request_id_uuid, f"AI 서버 호출 실패: {response.status}")
    except Exception as e:
        print(f"Error calling AI server: {e}")
        db_utils.update_loading_message(request_id_uuid, f"AI 서버 연결 오류: {e}")

def handle_confirm_delete():
    """채팅방 삭제를 확정하고 실행합니다."""
    if st.session_state.room_to_delete and st.session_state.user_id:
        deleted = db_utils.delete_chatroom(st.session_state.room_to_delete, st.session_state.user_id)
        if deleted:
            st.sidebar.success(f"'{st.session_state.room_name_to_delete}' 채팅방 삭제 완료.")
            if st.session_state.current_room_id == st.session_state.room_to_delete:
                st.session_state.current_room_id = None
                st.session_state.current_room_name = ""
                st.session_state.messages = []
        else: st.sidebar.error("채팅방 삭제 실패.")
    st.session_state.room_to_delete = None
    st.session_state.room_name_to_delete = None
    st.rerun()

def handle_cancel_delete():
    """채팅방 삭제를 취소합니다."""
    st.session_state.room_to_delete = None
    st.session_state.room_name_to_delete = None
    st.rerun()

def initialize_room_toggles(room_id):
    """특정 채팅방의 토글 상태를 초기화/로드합니다."""
    if room_id not in st.session_state.workflow_toggles:
        db_status = db_utils.get_state_status(room_id)
        st.session_state.workflow_toggles[room_id] = STATUS_TO_TOGGLES.get(db_status, STATUS_TO_TOGGLES['IDLE'])

def handle_popover_toggle_change():
    """Popover 내 토글 변경 시 콜백."""
    room_id = st.session_state.current_room_id
    if not room_id: return
    initialize_room_toggles(room_id)
    toggles = st.session_state.workflow_toggles[room_id]
    for key in TOGGLE_ORDER: toggles[key] = st.session_state.get(key, False)
    first_off_index = next((i for i, k in enumerate(TOGGLE_ORDER) if not toggles[k]), -1)
    if first_off_index != -1:
        for i in range(first_off_index + 1, len(TOGGLE_ORDER)): toggles[TOGGLE_ORDER[i]] = False
    for key, value in toggles.items(): st.session_state[key] = value

def create_sample_workflow_state():
    """테스트용 WorkflowState 객체를 생성합니다."""
    if not STATE_MODULE_LOADED: return None # 모듈 없으면 None 반환
    return WorkflowState(
        meta=MetaSection(trace_id="abc-123", current_stage="Report", workflow_status="In Progress"),
        query=QuerySection(original_query="AI 기술"),
        search=SearchSection(raw_search_results=[RawArticle(url="http://e.com/1", title="뉴스1", snippet="...", rank=1)]),
        report=ReportSection(report_content="AI는 발전중...")
    )

# --- 렌더링 함수 ---

def render_login():
    """로그인 화면 렌더링."""
    st.title("닉네임으로 로그인")
    nickname = st.text_input("닉네임을 입력하세요:", key="nickname_input_login")
    if st.button("로그인"):
        if nickname:
            user_id, nick = db_utils.get_or_create_user(nickname)
            if user_id:
                st.session_state.logged_in, st.session_state.user_id, st.session_state.nickname = True, user_id, nick
                st.rerun()
            else: st.error("로그인 실패. DB 연결 확인.")
        else: st.warning("닉네임 입력.")

def render_chat_management():
    """사이드바 렌더링."""
    st.sidebar.title(f"환영합니다, {st.session_state.nickname}님!")
    st.sidebar.subheader("새 채팅방 만들기")
    new_room_name = st.sidebar.text_input("채팅방 이름:", key="new_room_name_input")
    if st.sidebar.button("생성"):
        if new_room_name:
            room_id = db_utils.create_chatroom(st.session_state.user_id, new_room_name)

            if room_id: st.sidebar.success(f"'{new_room_name}' 생성 완료."); st.rerun()
            else: st.sidebar.error("생성 실패.")
        else: st.sidebar.warning("이름 입력.")

    st.sidebar.subheader("내 채팅방 목록")
    chatrooms = db_utils.get_user_chatrooms(st.session_state.user_id)

    if st.session_state.room_to_delete:
        st.sidebar.warning(f"'{st.session_state.room_name_to_delete}' 삭제하시겠습니까?")
        c1, c2 = st.sidebar.columns(2)
        c1.button("예", on_click=handle_confirm_delete, key="confirm_del_yes", type="primary")
        c2.button("아니요", on_click=handle_cancel_delete, key="confirm_del_no")

    for room in chatrooms:
        c1, c2 = st.sidebar.columns([3, 1])
        if c1.button(room['room_name'], key=f"room_btn_{room['room_id']}", use_container_width=True):
            if not st.session_state.room_to_delete:
                st.session_state.current_room_id = room['room_id']
                st.session_state.current_room_name = room['room_name']
                st.session_state.messages = db_utils.get_messages(room['room_id'])
                initialize_room_toggles(room['room_id']) # <<< 상태 로드/초기화
                st.session_state.is_loading = db_utils.check_ai_processing(room['room_id'])
                st.rerun()
        if c2.button("삭제", key=f"del_btn_{room['room_id']}", use_container_width=True):
            if not st.session_state.room_to_delete:
                st.session_state.room_to_delete = room['room_id']
                st.session_state.room_name_to_delete = room['room_name']
                st.rerun()

    if not chatrooms: st.sidebar.info("생성된 채팅방 없음.")
    if st.sidebar.button("로그아웃", key="logout_btn"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

def render_chat_interface():
    """메인 채팅/상태/노트패드 인터페이스 렌더링."""
    room_id = st.session_state.current_room_id
    if not room_id: return

    initialize_room_toggles(room_id)
    room_toggles = st.session_state.workflow_toggles[room_id]
    for key, value in room_toggles.items(): st.session_state[key] = value

    c_title, c_menu = st.columns([0.9, 0.1])
    with c_title: st.title(f"💬 {st.session_state.current_room_name}")
    with c_menu:
        st.write("") # Popover 위치 조절
        with st.popover("⚙️"):
            st.markdown("**워크플로우 단계**")
            prev_on = True
            for key in TOGGLE_ORDER:
                label = TOGGLE_LABELS[key]
                curr_val = st.session_state[key]
                disabled = not prev_on
                help_txt = "이전 단계를 활성화하세요." if disabled else ""
                if disabled and curr_val: st.session_state[key] = False; curr_val = False
                st.toggle(label, value=curr_val, key=key, disabled=disabled, help=help_txt, on_change=handle_popover_toggle_change)
                prev_on = st.session_state[key]

    chat_col, right_col = st.columns([2, 1])
    with chat_col:
        st.subheader("대화 내용")
        msg_container = st.container(height=800, border=False)
        with msg_container:
            for msg in st.session_state.messages:
                role = "assistant" if msg['message_type'] in ['final_ai', 'system', 'loading'] else "user"
                avatar = "🤖" if role == "assistant" else ("🧑‍💻" if msg['message_type'] == 'user' else "ℹ️")
                with st.chat_message(role, avatar=avatar):
                    if msg['message_type'] == 'loading':
                        with st.expander("AI 응답 생성 중...", expanded=True): st.write(msg['content']); st.spinner()
                    elif msg['message_type'] == 'final_ai':
                        think, answer = parse_think_answer(msg['content'])
                        if think:
                            with st.expander("생각 과정 🧠"): st.markdown(think)
                        st.markdown(answer)
                    else: st.markdown(f"**{msg['nickname']}**: {msg['content']}")

    with right_col:
        st.subheader("🛠️ 워크플로우 상태")
        current_state_data = db_utils.get_state(room_id) # DB에서 실제 상태 가져오기
        # current_state = WorkflowState(**current_state_data) if current_state_data else create_sample_workflow_state() # Pydantic 모델로 변환 또는 샘플 사용
        current_state = create_sample_workflow_state() # 지금은 샘플 사용

        if current_state and STATE_MODULE_LOADED:
            state_md = state_view_utils.format_workflow_state_to_markdown(current_state)
            with st.container(height=400, border=True): st.markdown(state_md, unsafe_allow_html=True)
        else: st.info("상태 정보 없음.")

        st.markdown("---")
        st.subheader("📝 노트패드")
        notes = db_utils.get_chatroom_canvas(room_id) or ""
        new_notes = st.text_area("노트:", value=notes, height=150, key=f"canvas_{room_id}")
        if st.button("노트 저장", key=f"save_canvas_{room_id}"):
            if db_utils.update_chatroom_canvas(room_id, new_notes): st.success("저장 완료!")
            else: st.error("저장 실패.")

    st.session_state.is_loading = db_utils.check_ai_processing(room_id)
    prompt = st.chat_input("메시지를 입력하세요...", key="chat_input_main", disabled=st.session_state.is_loading)

    if prompt:
        user_id = st.session_state.user_id
        current_toggles = st.session_state.workflow_toggles[room_id]
        current_toggles_tuple = tuple(current_toggles.values())
        ui_status = TOGGLES_TO_STATUS.get(current_toggles_tuple, 'IDLE')
        db_status = db_utils.get_state_status(room_id)
        action_needed = ui_status != db_status
        target_status = ui_status if action_needed else None

        print(f"DEBUG: UI={ui_status}, DB={db_status}, Target={target_status}")

        req_id = uuid.uuid4()
        db_utils.add_message(room_id, user_id, prompt, 'user', req_id)
        loading_text = f"{target_status} 요청 중..." if action_needed else "AI 응답 대기 중..."
        db_utils.add_message(room_id, user_id, loading_text, 'loading', req_id)

        st.session_state.messages = db_utils.get_messages(room_id)

        if action_needed: # db_utils.update_workflow_status(room_id, target_status) # 임시 DB 업데이트
            asyncio.run(call_ai_server(prompt, room_id, user_id, req_id, target_status))
        else:
            asyncio.run(call_ai_server(prompt, room_id, user_id, req_id, "CHAT"))
        st.rerun()

    if st.session_state.is_loading:
        time.sleep(2)
        if not db_utils.check_ai_processing(st.session_state.current_room_id):
            st.session_state.messages = db_utils.get_messages(st.session_state.current_room_id)
            st.rerun()
        else: st.rerun()

# --- 메인 실행 로직 ---
if not st.session_state.logged_in:
    render_login()
else:
    render_chat_management()
    if st.session_state.current_room_id:
        render_chat_interface()
    else:
        st.info("왼쪽 사이드바에서 채팅방을 선택하거나 새로 생성해주세요.")