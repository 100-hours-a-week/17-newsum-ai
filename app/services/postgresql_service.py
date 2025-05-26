# ai/app/services/postgresql_service.py
import asyncpg
import uuid # UUID 처리를 위해 추가
from sshtunnel import SSHTunnelForwarder
from pathlib import Path
from app.config.settings import settings
from app.utils.logger import get_logger

class PostgreSQLService:
    def __init__(self, use_ssh=True):
        self.logger = get_logger(__name__)
        self.pool = None
        self.ssh_tunnel = None
        self.use_ssh = use_ssh

    async def connect(self):
        """DB 연결 (타임아웃 포함)"""
        import asyncio
        try:
            if self.use_ssh and getattr(settings, 'SSH_HOST', None):
                self.logger.info("SSH 터널링 사용")
                self.ssh_tunnel = SSHTunnelForwarder((settings.SSH_HOST, settings.SSH_PORT), ssh_username=settings.SSH_USER, ssh_pkey=str(Path(settings.SSH_KEY_PATH)), remote_bind_address=('localhost', settings.POSTGRES_PORT))
                self.ssh_tunnel.start()
                host, port = 'localhost', self.ssh_tunnel.local_bind_port
            else:
                self.logger.info("직접 연결 사용")
                host, port = settings.POSTGRES_HOST, settings.POSTGRES_PORT

            dsn = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{host}:{port}/{settings.POSTGRES_DB}"
            self.logger.info(f"연결 시도: {settings.POSTGRES_USER}@{host}:{port}/{settings.POSTGRES_DB}")

            self.pool = await asyncio.wait_for(
                asyncpg.create_pool(dsn),
                timeout=10
            )
            self.logger.info(f"PostgreSQL 연결 완료: {host}:{port}")
        except asyncio.TimeoutError:
            self.logger.error("PostgreSQL 연결 타임아웃")
            raise ConnectionError("PostgreSQL 연결 타임아웃 (10초)")
        except Exception as e:
            self.logger.error(f"연결 실패: {e}")
            raise

    async def close(self):
        if self.pool: await self.pool.close()
        if self.ssh_tunnel: self.ssh_tunnel.close()

    async def execute(self, query, *args):
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch_one(self, query, *args):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetch_all(self, query, *args):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    # ===== 스키마에 맞춘 채팅 기능 메서드들 =====

    async def get_or_create_user(self, nickname: str) -> int:
        """닉네임으로 사용자를 찾아 ID를 반환하거나, 없으면 생성하고 ID를 반환합니다."""
        async with self.pool.acquire() as conn:
            # 먼저 사용자를 찾습니다.
            user_id = await conn.fetchval(
                "SELECT user_id FROM ai_test_users WHERE nickname = $1",
                nickname
            )
            if user_id:
                return user_id
            else:
                # 사용자가 없으면 생성합니다. ON CONFLICT를 사용하여 동시성 문제를 방지합니다.
                try:
                    user_id = await conn.fetchval(
                        """
                        INSERT INTO ai_test_users (nickname) VALUES ($1)
                        ON CONFLICT (nickname) DO UPDATE SET nickname = EXCLUDED.nickname
                        RETURNING user_id
                        """,
                        nickname
                    )
                    # ON CONFLICT 시 RETURNING이 값을 반환하지 않을 수 있으므로, 다시 조회합니다.
                    if not user_id:
                         user_id = await conn.fetchval(
                            "SELECT user_id FROM ai_test_users WHERE nickname = $1",
                            nickname
                        )
                    return user_id
                except Exception as e:
                    self.logger.error(f"사용자 생성/조회 실패 (nickname: {nickname}): {e}")
                    raise

    async def add_message(self, room_id: str, user_id: int, content: str, message_type: str, request_id: str) -> dict:
        """새 메시지를 ai_test_messages 테이블에 저장합니다."""
        # room_id를 int로 변환합니다.
        try:
            room_id_int = int(room_id)
        except ValueError:
            self.logger.error(f"유효하지 않은 room_id: {room_id}. 숫자로 변환할 수 없습니다.")
            raise ValueError(f"Invalid room_id: {room_id}")

        # request_id를 UUID로 변환합니다.
        try:
            request_uuid = uuid.UUID(request_id)
        except ValueError:
            self.logger.warning(f"유효하지 않은 UUID 형식 request_id: {request_id}. NULL로 저장합니다.")
            request_uuid = None # 또는 에러 발생

        query = """
        INSERT INTO ai_test_messages (room_id, user_id, content, message_type, request_id)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING message_id, room_id, user_id, content, message_type, sent_at
        """
        async with self.pool.acquire() as conn:
            # 참고: ai_test_chatrooms 테이블에 room_id가 존재하는지 확인하는 로직이 필요할 수 있으나,
            # 현재 스키마에서는 FOREIGN KEY 제약 조건이 이를 처리합니다.
            # 만약 방이 없다면 여기서 에러가 발생하며, 이는 워커/핸들러 수준에서 처리해야 합니다.
            # `ensure_room_exists`는 스키마 변경으로 인해 제거되었습니다.
            row = await conn.fetchrow(query, room_id_int, user_id, content, message_type, request_uuid)
            return dict(row) if row else None

    async def get_summary(self, room_id: str) -> str:
        """현재 요약본 조회"""
        query = "SELECT summary FROM ai_test_chatrooms WHERE room_id = $1"
        result = await self.fetch_one(query, int(room_id))
        return result['summary'] if result and result['summary'] else ''

    async def get_recent_history(self, room_id: str, limit: int = 10) -> list:
        """최근 N개 대화 조회 (시간순 정렬) 및 role 매핑"""
        query = """
        SELECT m.message_id, m.room_id, u.nickname, m.content, m.message_type, m.sent_at
        FROM ai_test_messages m
        JOIN ai_test_users u ON m.user_id = u.user_id
        WHERE m.room_id = $1
        ORDER BY m.sent_at DESC
        LIMIT $2
        """
        messages = await self.fetch_all(query, int(room_id), limit)
        # LLM 컨텍스트에 맞게 role 매핑 및 시간순 재정렬
        history = []
        for msg in reversed(messages):
            role = 'assistant' if msg['message_type'] == 'final_ai' else \
                   'user' if msg['message_type'] == 'user' else \
                   'system' # 다른 타입은 system 또는 무시
            if role in ['user', 'assistant']: # LLM에는 user와 assistant만 전달
                history.append({
                    "role": role,
                    "content": msg['content'],
                    # 필요시 다른 정보도 포함 가능
                    "nickname": msg['nickname'],
                    "sent_at": msg['sent_at']
                })
        return history

    async def get_full_history(self, room_id: str) -> list:
        """전체 대화 조회 및 role 매핑"""
        query = """
        SELECT m.message_id, m.room_id, u.nickname, m.content, m.message_type, m.sent_at
        FROM ai_test_messages m
        JOIN ai_test_users u ON m.user_id = u.user_id
        WHERE m.room_id = $1
        ORDER BY m.sent_at ASC
        """
        messages = await self.fetch_all(query, int(room_id))
        # 요약용으로 role 매핑
        history = []
        for msg in messages:
            # 요약 프롬프트에는 닉네임이나 명시적 role을 사용하는 것이 좋음
            role = 'AI' if msg['message_type'] == 'final_ai' else msg['nickname']
            history.append({
                "role": role,
                "content": msg['content'],
                "message_type": msg['message_type'], # 원본 타입 유지
                "sent_at": msg['sent_at']
            })
        return history

    async def update_summary(self, room_id: str, summary_text: str) -> bool:
        """요약 갱신 및 마지막 요약 시간 업데이트"""
        query = """
        UPDATE ai_test_chatrooms
        SET summary = $1, last_summary_at = CURRENT_TIMESTAMP
        WHERE room_id = $2
        """
        result = await self.execute(query, summary_text, int(room_id))
        return result == "UPDATE 1"

    async def get_context(self, room_id: str, recent_limit: int = 10) -> dict:
        """SLM에 전달할 문맥 생성 (요약 + 최근 대화)"""
        summary = await self.get_summary(room_id)
        recent_history = await self.get_recent_history(room_id, recent_limit)

        return {
            "room_id": room_id,
            "summary": summary,
            "recent_history": recent_history,
            "message_count": len(recent_history)
        }

    # 스키마에 맞게 다른 메서드들 (bulk_insert, multi_table_select 등)도
    # 필요하다면 테이블/컬럼 이름을 확인하고 수정해야 합니다.
    # 여기서는 채팅 기능 관련 메서드만 집중적으로 수정했습니다.