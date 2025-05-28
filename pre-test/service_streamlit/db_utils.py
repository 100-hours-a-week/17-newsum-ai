import psycopg2
import psycopg2.extras # <--- Add this import
import os, json
from dotenv import load_dotenv

load_dotenv()
psycopg2.extras.register_uuid() # <--- Add this line to register UUID adapter globally


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
        conn.close()
    return user_id, nickname

def create_chatroom(user_id, room_name):
    """
    새로운 채팅방을 생성하고, 해당 채팅방의 초기 워크플로우 상태 레코드('IDLE')도 생성합니다.
    성공 시 새로운 room_id, 실패 시 None을 반환합니다.
    """
    conn = get_db_connection()
    if not conn:
        return None

    new_room_id = None
    try:
        with conn.cursor() as cur:
            # 1. 채팅방 생성
            cur.execute(
                "INSERT INTO ai_test_chatrooms (room_name, user_id) VALUES (%s, %s) RETURNING room_id",
                (room_name, user_id)
            )
            result = cur.fetchone()

            if result:
                new_room_id = result[0]
                # 2. 해당 채팅방의 초기 워크플로우 상태 레코드 생성
                #    PostgreSQL 함수 ensure_workflow_status_exists(room_id) 호출
                cur.execute("SELECT ensure_workflow_status_exists(%s);", (new_room_id,))
                # 또는, 직접 INSERT (ensure_workflow_status_exists 함수가 없다면)
                # cur.execute(
                #    "INSERT INTO ai_test_room_workflow_status (room_id, status) VALUES (%s, %s) ON CONFLICT (room_id) DO NOTHING",
                #    (new_room_id, 'IDLE') # room_workflow_id는 SERIAL이므로 자동 생성됨
                # )
                conn.commit() # 모든 변경사항 커밋
            else:
                # 채팅방 생성 실패 시 롤백 (특별한 오류가 없으면 이 부분은 실행되지 않을 수 있음)
                conn.rollback()
                new_room_id = None # 명시적으로 None 처리

    except psycopg2.Error as e:
        print(f"Error creating chatroom or initial workflow status for room '{room_name}': {e}")
        if conn:
            conn.rollback() # 오류 발생 시 롤백
        new_room_id = None # 오류 시 None 반환 보장
    finally:
        if conn:
            conn.close()

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
        conn.close()
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
        conn.close()
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
        conn.close()
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
        conn.close()
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
        conn.close()
    return processing

def delete_chatroom(room_id, user_id):
    """
    특정 사용자가 소유한 채팅방을 삭제합니다.
    성공적으로 삭제되면 True, 실패하면 False를 반환합니다.
    """
    conn = get_db_connection()
    if not conn:
        print(f"Database connection failed. Cannot delete chatroom {room_id}.")
        return False

    deleted = False
    try:
        with conn.cursor() as cur:
            # DELETE 쿼리 실행: room_id와 user_id가 모두 일치하는 경우에만 삭제
            cur.execute(
                "DELETE FROM ai_test_chatrooms WHERE room_id = %s AND user_id = %s",
                (room_id, user_id)
            )

            # cur.rowcount는 영향을 받은 행의 수를 반환합니다.
            # 1 이상이면 성공적으로 삭제된 것입니다.
            if cur.rowcount > 0:
                conn.commit()  # 변경 사항을 DB에 최종 반영
                deleted = True
                print(f"Chatroom {room_id} deleted successfully by user {user_id}.")
            else:
                # 삭제된 행이 없으면, room_id가 없거나 user_id가 일치하지 않는 경우입니다.
                conn.rollback() # 혹시 모를 변경 사항 롤백
                print(f"Failed to delete chatroom {room_id}: Not found or not owned by user {user_id}.")
                deleted = False

    except psycopg2.Error as e:
        print(f"Error deleting chatroom {room_id}: {e}")
        conn.rollback() # 오류 발생 시 롤백
        deleted = False
    finally:
        conn.close() # DB 연결 종료

    return deleted

def get_chatroom_canvas(room_id):
    """특정 채팅방의 캔버스(노트패드) 내용을 조회합니다."""
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
        print(f"Error getting chatroom canvas: {e}")
    finally:
        conn.close()
    return content

def update_chatroom_canvas(room_id, content):
    """특정 채팅방의 캔버스(노트패드) 내용을 업데이트합니다."""
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
        print(f"Error updating chatroom canvas: {e}")
        conn.rollback()
    finally:
        conn.close()
    return updated
def ensure_workflow_status_exists(room_id):
    """ai_test_room_workflow_status 테이블에 해당 room_id의 행이 있는지 확인하고 없으면 생성합니다."""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            # PostgreSQL 함수 호출
            cur.execute("SELECT ensure_workflow_status_exists(%s);", (room_id,))
            conn.commit()
    except psycopg2.Error as e:
        print(f"Error ensuring workflow status exists for room {room_id}: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_state_status(room_id):
    """특정 채팅방의 'status' 값을 가져옵니다."""
    ensure_workflow_status_exists(room_id) # 먼저 행 존재 확인/생성
    conn = get_db_connection()
    if not conn: return 'IDLE'
    status = 'IDLE' # 기본값
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM ai_test_room_workflow_status WHERE room_id = %s", (room_id,))
            result = cur.fetchone()
            if result:
                status = result[0]
    except psycopg2.Error as e:
        print(f"Error getting state status for room {room_id}: {e}")
        status = 'ERROR' # 오류 발생 시 상태
    finally:
        conn.close()
    return status

def get_state(room_id):
    """특정 채팅방의 'task_details'(JSONB) 값을 Python 딕셔너리로 가져옵니다."""
    ensure_workflow_status_exists(room_id)
    conn = get_db_connection()
    if not conn: return None
    state_data = None
    try:
        # DictCursor와 register_json 사용
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            psycopg2.extras.register_json(cur) # JSON 자동 변환 등록
            cur.execute("SELECT task_details FROM ai_test_room_workflow_status WHERE room_id = %s", (room_id,))
            result = cur.fetchone()
            if result and result['task_details']:
                state_data = result['task_details']
    except psycopg2.Error as e:
        print(f"Error getting state details for room {room_id}: {e}")
    finally:
        conn.close()
    return state_data

def update_workflow_status(room_id, new_status, task_details_dict=None):
    """특정 채팅방의 'status'와 선택적으로 'task_details'를 업데이트합니다."""
    ensure_workflow_status_exists(room_id)
    conn = get_db_connection()
    if not conn: return False
    updated = False
    try:
        with conn.cursor() as cur:
            if task_details_dict is not None:
                # Python 딕셔너리를 JSON 문자열로 변환하여 저장
                cur.execute(
                    "UPDATE ai_test_room_workflow_status SET status = %s, task_details = %s WHERE room_id = %s",
                    (new_status, json.dumps(task_details_dict), room_id)
                )
            else:
                cur.execute(
                    "UPDATE ai_test_room_workflow_status SET status = %s WHERE room_id = %s",
                    (new_status, room_id)
                )
            conn.commit()
            updated = cur.rowcount > 0
    except psycopg2.Error as e:
        print(f"Error updating workflow status for room {room_id}: {e}")
        conn.rollback()
    finally:
        conn.close()
    return updated