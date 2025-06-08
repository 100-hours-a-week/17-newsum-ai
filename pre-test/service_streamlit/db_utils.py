import psycopg2
import psycopg2.extras
import os
import json
from dotenv import load_dotenv
import uuid  # <--- UUID 모듈 추가

load_dotenv()
psycopg2.extras.register_uuid()

def get_db_connection():
    """데이터베이스 연결을 생성하고 반환합니다."""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Database connection failed: {e}")
        return None

def get_or_create_user(nickname):
    """닉네임으로 사용자를 조회하고 없으면 생성합니다."""
    conn = get_db_connection()
    if not conn: return None, None
    user_id = None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM ai_test_users WHERE nickname = %s", (nickname,))
            user = cur.fetchone()
            if user:
                user_id = user[0]
            else:
                cur.execute("INSERT INTO ai_test_users (nickname) VALUES (%s) RETURNING user_id", (nickname,))
                user_id = cur.fetchone()[0]
                conn.commit()
    except psycopg2.Error as e:
        print(f"Error getting or creating user: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()
    return user_id, nickname

def create_chatroom(user_id, room_name):
    """
    새로운 채팅방을 생성합니다. (워크플로우 상태는 별도로 생성/관리됩니다)
    성공 시 새로운 room_id, 실패 시 None을 반환합니다.
    """
    conn = get_db_connection()
    if not conn: return None
    new_room_id = None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ai_test_chatrooms (room_name, user_id) VALUES (%s, %s) RETURNING room_id",
                (room_name, user_id)
            )
            result = cur.fetchone()
            if result:
                new_room_id = result[0]
                conn.commit()
            else:
                conn.rollback()
    except psycopg2.Error as e:
        print(f"Error creating chatroom '{room_name}': {e}")
        if conn: conn.rollback()
        new_room_id = None
    finally:
        if conn: conn.close()
    return new_room_id

def get_user_chatrooms(user_id):
    """특정 사용자의 채팅방 목록을 조회합니다."""
    conn = get_db_connection()
    if not conn: return []
    chatrooms = []
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT room_id, room_name FROM ai_test_chatrooms WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
            chatrooms = cur.fetchall()
    except psycopg2.Error as e:
        print(f"Error getting chatrooms: {e}")
    finally:
        if conn: conn.close()
    return chatrooms

def get_messages(room_id):
    """특정 채팅방의 메시지 목록을 조회합니다."""
    conn = get_db_connection()
    if not conn: return []
    messages = []
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT m.message_id, m.content, m.message_type, u.nickname, m.sent_at, m.request_id
                FROM ai_test_messages m
                JOIN ai_test_users u ON m.user_id = u.user_id
                WHERE m.room_id = %s
                ORDER BY m.sent_at ASC
            """, (room_id,))
            messages = cur.fetchall()
    except psycopg2.Error as e:
        print(f"Error getting messages: {e}")
    finally:
        if conn: conn.close()
    return messages

def add_message(room_id, user_id, content, message_type, request_id=None):
    """새로운 메시지를 추가합니다."""
    conn = get_db_connection()
    if not conn: return None
    message_id = None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ai_test_messages (room_id, user_id, content, message_type, request_id)
                VALUES (%s, %s, %s, %s, %s) RETURNING message_id
            """, (room_id, user_id, content, message_type, request_id))
            message_id = cur.fetchone()[0]
            conn.commit()
    except psycopg2.Error as e:
        print(f"Error adding message: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()
    return message_id

def update_loading_message(request_id, content):
    """'loading' 메시지를 'final_ai' 메시지로 업데이트합니다."""
    conn = get_db_connection()
    if not conn: return False
    updated = False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE ai_test_messages
                SET content = %s, message_type = 'final_ai', sent_at = CURRENT_TIMESTAMP
                WHERE request_id = %s AND message_type = 'loading'
            """, (content, request_id))
            conn.commit()
            updated = cur.rowcount > 0
    except psycopg2.Error as e:
        print(f"Error updating loading message: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()
    return updated

def check_ai_processing(room_id):
    """현재 AI가 응답을 생성 중인지 확인합니다."""
    conn = get_db_connection()
    if not conn: return False
    processing = False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM ai_test_messages WHERE room_id = %s AND message_type = 'loading'", (room_id,))
            processing = cur.fetchone() is not None
    except psycopg2.Error as e:
        print(f"Error checking AI processing: {e}")
    finally:
        if conn: conn.close()
    return processing

def delete_chatroom(room_id, user_id):
    """특정 사용자가 소유한 채팅방을 삭제합니다."""
    conn = get_db_connection()
    if not conn: return False
    deleted = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM ai_test_chatrooms WHERE room_id = %s AND user_id = %s",
                (room_id, user_id)
            )
            if cur.rowcount > 0:
                conn.commit()
                deleted = True
            else:
                conn.rollback()
    except psycopg2.Error as e:
        print(f"Error deleting chatroom {room_id}: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()
    return deleted

def get_chatroom_canvas(room_id):
    """특정 채팅방의 캔버스(노트패드) 내용을 조회합니다. (주의: 스키마에 canvas_content 컬럼 필요)"""
    conn = get_db_connection()
    if not conn: return None
    content = None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT canvas_content FROM ai_test_chatrooms WHERE room_id = %s", (room_id,))
            result = cur.fetchone()
            if result:
                content = result[0]
    except psycopg2.Error as e:
        print(f"Error getting chatroom canvas (check if 'canvas_content' column exists): {e}")
    finally:
        if conn: conn.close()
    return content

def update_chatroom_canvas(room_id, content):
    """특정 채팅방의 캔버스(노트패드) 내용을 업데이트합니다. (주의: 스키마에 canvas_content 컬럼 필요)"""
    conn = get_db_connection()
    if not conn: return False
    updated = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ai_test_chatrooms SET canvas_content = %s WHERE room_id = %s",
                (content, room_id)
            )
            conn.commit()
            updated = cur.rowcount > 0
    except psycopg2.Error as e:
        print(f"Error updating chatroom canvas (check if 'canvas_content' column exists): {e}")
        conn.rollback()
    finally:
        if conn: conn.close()
    return updated

# --- New/Modified Workflow Functions ---

def get_or_create_workflow(room_id):
    """
    지정된 room_id에 대한 가장 최근의 work_id를 가져오거나,
    없으면 새로 생성하여 반환합니다.
    """
    conn = get_db_connection()
    if not conn: return None
    work_id = None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT work_id FROM ai_test_room_workflow_status
                WHERE room_id = %s ORDER BY created_at DESC LIMIT 1
            """, (room_id,))
            result = cur.fetchone()

            if result:
                work_id = result[0]
            else:
                work_id = uuid.uuid4()
                cur.execute("""
                    INSERT INTO ai_test_room_workflow_status (work_id, room_id, status)
                    VALUES (%s, %s, %s)
                """, (work_id, room_id, 'STARTED'))
                conn.commit()
                print(f"Created new workflow {work_id} for room {room_id}.")
    except psycopg2.Error as e:
        print(f"Error getting or creating workflow for room {room_id}: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()
    return work_id

def get_state_status(work_id):
    """특정 워크플로우(work_id)의 'status' 값을 가져옵니다."""
    conn = get_db_connection()
    if not conn or not work_id: return 'UNKNOWN'
    status = 'UNKNOWN'
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM ai_test_room_workflow_status WHERE work_id = %s", (work_id,))
            result = cur.fetchone()
            if result:
                status = result[0]
    except psycopg2.Error as e:
        print(f"Error getting state status for comic {work_id}: {e}")
        status = 'ERROR'
    finally:
        if conn: conn.close()
    return status

def get_state(work_id):
    """특정 워크플로우(work_id)의 'task_details'(JSONB) 값을 Python 딕셔너리로 가져옵니다."""
    conn = get_db_connection()
    if not conn or not work_id: return None
    state_data = None
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            psycopg2.extras.register_json(cur)
            cur.execute("SELECT task_details FROM ai_test_room_workflow_status WHERE work_id = %s", (work_id,))
            result = cur.fetchone()
            if result and result['task_details']:
                state_data = result['task_details']
    except psycopg2.Error as e:
        print(f"Error getting state details for comic {work_id}: {e}")
    finally:
        if conn: conn.close()
    return state_data

def update_workflow_status(work_id, new_status, task_details_dict=None):
    """특정 워크플로우(work_id)의 'status'와 'task_details'를 업데이트합니다."""
    conn = get_db_connection()
    if not conn or not work_id: return False
    updated = False
    try:
        with conn.cursor() as cur:
            if task_details_dict is not None:
                cur.execute(
                    "UPDATE ai_test_room_workflow_status SET status = %s, task_details = %s WHERE work_id = %s",
                    (new_status, json.dumps(task_details_dict), work_id)
                )
            else:
                cur.execute(
                    "UPDATE ai_test_room_workflow_status SET status = %s WHERE work_id = %s",
                    (new_status, work_id)
                )
            conn.commit()
            updated = cur.rowcount > 0
    except psycopg2.Error as e:
        print(f"Error updating workflow status for comic {work_id}: {e}")
        conn.rollback()
    finally:
        if conn: conn.close()
    return updated