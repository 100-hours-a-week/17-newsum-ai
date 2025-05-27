import streamlit as st
import db_utils
import os
import uuid
import time
import aiohttp
import asyncio
import json

# --- 초기 설정 ---
st.set_page_config(page_title="SLM Chat Service", layout="wide")

# --- 세션 상태 초기화 ---
def initialize_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'nickname' not in st.session_state:
        st.session_state.nickname = ""
    if 'current_room_id' not in st.session_state:
        st.session_state.current_room_id = None
    if 'current_room_name' not in st.session_state:
        st.session_state.current_room_name = ""
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'last_message_count' not in st.session_state:
        st.session_state.last_message_count = 0

# 환경 변수 설정
AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://localhost:8000/api/v2/chat")
RECEPTION_API_PORT = os.getenv("RECEPTION_API_PORT", "9090")

# --- 개선된 폴링 시스템 ---
class SmartPoller:
    def __init__(self):
        self.polling_interval = 0.5  # 500ms
        self.max_poll_count = 60     # 최대 30초
        
    def check_for_updates(self, room_id: int, initial_count: int) -> tuple[bool, list, bool]:
        """
        스마트 업데이트 확인
        Returns: (has_updates, messages, is_loading)
        """
        is_loading = db_utils.check_ai_processing(room_id)
        messages = db_utils.get_messages(room_id)
        current_count = len(messages)
        
        # 업데이트가 있는지 확인
        has_updates = (not is_loading and st.session_state.is_loading) or (current_count > initial_count)
        
        return has_updates, messages, is_loading

# 전역 폴링 객체
poller = SmartPoller()

# --- 비동기 AI 서버 호출 함수 ---
async def call_ai_server(user_message: str, room_id: int, user_id: int, request_id: uuid.UUID):
    """AI 서버에 비동기적으로 요청을 보냅니다."""
    payload = {
        "room_id": str(room_id),
        "user_id": str(user_id),
        "request_id": str(request_id),
        "message": user_message
    }
    
    try:
        json_data = json.dumps(payload)
        headers = {'Content-Type': 'application/json'}

        async with aiohttp.ClientSession() as session:
            async with session.post(AI_SERVER_URL, data=json_data, headers=headers) as response:
                if response.status in [200, 202]:
                    st.success("✅ AI 요청이 성공적으로 전송되었습니다.")
                else:
                    st.error(f"❌ AI 서버 오류 (상태: {response.status})")
                    db_utils.update_loading_message(request_id, f"AI 서버 오류 (상태: {response.status})")

    except Exception as e:
        st.error(f"❌ AI 서버 연결 오류: {str(e)}")
        db_utils.update_loading_message(request_id, f"연결 오류: {str(e)}")

# --- 화면 렌더링 함수 ---
def render_login():
    """로그인 화면을 렌더링합니다."""
    st.title("🚀 SLM Chat Service")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("닉네임으로 로그인")
        nickname = st.text_input(
            "닉네임을 입력하세요:", 
            placeholder="예: 김철수", 
            key="nickname_input_login"
        )
        
        if st.button("🔐 로그인", use_container_width=True):
            if nickname.strip():
                with st.spinner("로그인 중..."):
                    user_id, nick = db_utils.get_or_create_user(nickname.strip())
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.nickname = nick
                        st.success(f"환영합니다, {nick}님!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 로그인에 실패했습니다. DB 연결을 확인하세요.")
            else:
                st.warning("⚠️ 닉네임을 입력해주세요.")

def render_chat_management():
    """채팅방 관리 화면을 렌더링합니다."""
    st.sidebar.title(f"👋 {st.session_state.nickname}님!")
    st.sidebar.markdown("---")

    # 새 채팅방 만들기
    with st.sidebar.expander("➕ 새 채팅방 만들기", expanded=False):
        new_room_name = st.text_input(
            "채팅방 이름:", 
            placeholder="예: AI와 대화하기", 
            key="new_room_name_input"
        )
        if st.button("생성", use_container_width=True):
            if new_room_name.strip():
                with st.spinner("채팅방 생성 중..."):
                    room_id = db_utils.create_chatroom(st.session_state.user_id, new_room_name.strip())
                    if room_id:
                        st.success(f"✅ '{new_room_name}' 채팅방이 생성되었습니다.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 채팅방 생성에 실패했습니다.")
            else:
                st.warning("⚠️ 채팅방 이름을 입력해주세요.")

    # 채팅방 목록
    st.sidebar.subheader("💬 내 채팅방 목록")
    chatrooms = db_utils.get_user_chatrooms(st.session_state.user_id)
    
    if chatrooms:
        for room in chatrooms:
            # 현재 선택된 방 표시
            is_current = st.session_state.current_room_id == room['room_id']
            button_text = f"{'🟢 ' if is_current else '⚪ '}{room['room_name']}"
            
            if st.sidebar.button(button_text, key=f"room_btn_{room['room_id']}", use_container_width=True):
                if not is_current:  # 다른 방 선택 시에만 업데이트
                    st.session_state.current_room_id = room['room_id']
                    st.session_state.current_room_name = room['room_name']
                    st.session_state.messages = db_utils.get_messages(room['room_id'])
                    st.session_state.last_message_count = len(st.session_state.messages)
                    st.session_state.is_loading = db_utils.check_ai_processing(room['room_id'])
                    st.rerun()
    else:
        st.sidebar.info("📝 아직 생성된 채팅방이 없습니다.")

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 로그아웃", use_container_width=True):
        # 세션 상태 초기화
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def render_chat_interface():
    """개선된 채팅 인터페이스를 렌더링합니다."""
    # 헤더
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"💬 {st.session_state.current_room_name}")
    with col2:
        if st.session_state.is_loading:
            st.info("🤖 AI 응답 중...")

    # 메시지 컨테이너
    message_container = st.container()
    with message_container:
        messages_to_display = st.session_state.get('messages', [])
        
        for msg in messages_to_display:
            role = "assistant" if msg['message_type'] in ['final_ai', 'system', 'loading'] else "user"
            avatar_icon = "🤖" if role == "assistant" else "👤"
            
            if msg['message_type'] == 'system':
                avatar_icon = 'ℹ️'

            with st.chat_message(role, avatar=avatar_icon):
                if msg['message_type'] == 'loading':
                    with st.expander("🔄 AI가 응답을 생성하고 있습니다...", expanded=True):
                        st.write(msg['content'])
                        st.spinner("처리 중...")
                else:
                    display_name = msg['nickname'] if msg['nickname'] != 'system' else "System"
                    st.markdown(f"**{display_name}**: {msg['content']}")

    # 개선된 스마트 폴링
    if st.session_state.is_loading:
        initial_count = st.session_state.last_message_count
        has_updates, current_messages, is_loading = poller.check_for_updates(
            st.session_state.current_room_id, 
            initial_count
        )
        
        if has_updates or not is_loading:
            # 상태 업데이트
            st.session_state.messages = current_messages
            st.session_state.is_loading = is_loading
            st.session_state.last_message_count = len(current_messages)
            st.rerun()
        else:
            # 짧은 대기 후 다시 확인
            time.sleep(poller.polling_interval)
            st.rerun()

    # 채팅 입력
    prompt = st.chat_input(
        "메시지를 입력하세요...", 
        key="chat_input_main", 
        disabled=st.session_state.is_loading
    )

    if prompt:
        user_id = st.session_state.user_id
        room_id = st.session_state.current_room_id
        request_id = uuid.uuid4()

        try:
            # 1. 사용자 메시지 저장
            db_utils.add_message(room_id, user_id, prompt, 'user', request_id)

            # 2. 로딩 메시지 저장
            loading_msg_content = "AI가 응답을 준비하고 있습니다..."
            db_utils.add_message(room_id, user_id, loading_msg_content, 'loading', request_id)

            # 3. AI 서버 호출
            asyncio.run(call_ai_server(prompt, room_id, user_id, request_id))

            # 4. 상태 업데이트 및 화면 갱신
            st.session_state.messages = db_utils.get_messages(room_id)
            st.session_state.last_message_count = len(st.session_state.messages)
            st.session_state.is_loading = True
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ 메시지 전송 중 오류 발생: {str(e)}")

# --- 메인 로직 ---
def main():
    initialize_session_state()
    
    if not st.session_state.get('logged_in', False):
        render_login()
    else:
        render_chat_management()
        if st.session_state.get('current_room_id'):
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
