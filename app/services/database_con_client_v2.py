# app/services/database_client_v2.py

import redis.asyncio as redis
import json
import asyncio
from app.config.settings import settings # 변경: 중앙 설정 객체 임포트
from typing import Optional, Any, Dict, Union, Callable, Coroutine
from app.utils.logger import get_logger

# 로거 설정
logger = get_logger(__name__)

class DatabaseClientV2:
    """
    비동기 Redis 데이터베이스 클라이언트 (V2)
    기본적인 Get/Set 및 Pub/Sub 기능을 포함하며 JSON 직렬화를 사용합니다.
    환경 변수 또는 인자를 통해 설정을 구성합니다.
    """
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        decode_responses: bool = True, # JSON 사용 시 True가 편리함
        logger_name: str = "DatabaseClientV2"
    ):
        """
        Redis 클라이언트 및 Pub/Sub 객체 초기화.
        인자가 제공되지 않으면 환경 변수에서 값을 찾습니다.
        (REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD)

        Args:
            host (Optional[str]): Redis 서버 호스트. 기본값: 환경변수 REDIS_HOST 또는 'localhost'.
            port (Optional[int]): Redis 서버 포트. 기본값: 환경변수 REDIS_PORT 또는 6379.
            db (Optional[int]): Redis 데이터베이스 번호. 기본값: 환경변수 REDIS_DB 또는 0.
            password (Optional[str]): Redis 비밀번호. 기본값: 환경변수 REDIS_PASSWORD 또는 None.
            decode_responses (bool): 응답을 자동으로 디코딩할지 여부. 기본값: True.
            logger_name (str): 로거 인스턴스 이름.
        """
        # 로거 설정
        self.logger = get_logger(logger_name)
        try:
            # 생성자 인수가 제공되면 해당 값을 사용하고, 그렇지 않으면 settings 객체의 값을 사용
            redis_host = host or settings.REDIS_HOST
            redis_port = port or settings.REDIS_PORT  # settings.REDIS_PORT는 이미 int 형이므로 int() 변환 불필요
            redis_db = db or settings.REDIS_DB  # settings.REDIS_DB는 이미 int 형이므로 int() 변환 불필요
            redis_password = password or settings.REDIS_PASSWORD

            self.logger.info(f"Redis 클라이언트 초기화 중: host={redis_host}, port={redis_port}, db={redis_db}")
            # 단일 클라이언트 인스턴스 생성 (Connection Pool 사용 안 함)
            self.client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=decode_responses
            )
            # Pub/Sub 객체 초기화
            self.pubsub = self.client.pubsub(ignore_subscribe_messages=True) # 구독 확인 메시지 무시
            self.logger.info("Redis 클라이언트 및 Pub/Sub 객체가 성공적으로 초기화되었습니다.")

        except ValueError as e:
            # settings.py 에서 int 변환 실패 시 또는 여기서 타입 검사 실패 시 여기로 올 수 있음
            self.logger.error(f"Redis 설정 값 오류: {e}", exc_info=True)
            raise ValueError(f"Invalid Redis configuration value: {e}") from e
        except Exception as e:
            self.logger.error(f"Redis 클라이언트 초기화 실패: {e}", exc_info=True)
            self.client = None
            self.pubsub = None
            raise ConnectionError(f"Failed to initialize Redis client: {e}") from e

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """
        키-값 쌍을 저장합니다 (JSON 직렬화 및 선택적 만료 시간 포함).

        Args:
            key (str): 저장할 키.
            value (Any): 저장할 값 (JSON 직렬화 가능해야 함).
            expire (Optional[int]): 만료 시간(초).

        Returns:
            bool: 저장 성공 여부.
        """
        if not self.client:
            logger.error("Redis 클라이언트가 초기화되지 않아 set 작업을 수행할 수 없습니다.")
            return False
        try:
            serialized_value = json.dumps(value)
            logger.debug(f"Redis SET: key='{key}', expiry={expire} seconds")
            await self.client.set(key, serialized_value, ex=expire)
            # logger.info(f"Redis SET 성공: key='{key}'") # 성공 로깅은 너무 많을 수 있어 debug로 변경하거나 제거 고려
            return True
        except TypeError as e:
            logger.error(f"Redis SET 실패: 값 JSON 직렬화 오류. key='{key}', value_type={type(value)}, error={e}", exc_info=True)
            return False
        except redis.RedisError as e:
            logger.error(f"Redis SET 실패: Redis 오류 발생. key='{key}', error={e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Redis SET 실패: 알 수 없는 오류 발생. key='{key}', error={e}", exc_info=True)
            return False

    async def get(self, key: str) -> Optional[Any]:
        """
        키를 사용하여 값을 검색합니다 (JSON 역직렬화).

        Args:
            key (str): 검색할 키.

        Returns:
            Optional[Any]: 역직렬화된 값 또는 None (키 없음 또는 오류).
        """
        if not self.client:
            logger.error("Redis 클라이언트가 초기화되지 않아 get 작업을 수행할 수 없습니다.")
            return None
        try:
            logger.debug(f"Redis GET: key='{key}'")
            serialized_value = await self.client.get(key)
            if serialized_value:
                # logger.info(f"Redis GET: 키 찾음 (cache hit). key='{key}'")
                 try:
                    return json.loads(serialized_value)
                 except json.JSONDecodeError as e:
                    logger.error(f"Redis GET 실패: JSON 역직렬화 오류. key='{key}', value='{serialized_value[:100]}...', error={e}", exc_info=True)
                    return None
            else:
                # logger.info(f"Redis GET: 키 없음 (cache miss). key='{key}'")
                return None
        except redis.RedisError as e:
            logger.error(f"Redis GET 실패: Redis 오류 발생. key='{key}', error={e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Redis GET 실패: 알 수 없는 오류 발생. key='{key}', error={e}", exc_info=True)
            return None

    async def publish(
        self,
        channel: str,
        message: Union[str, Dict, list]
    ) -> bool:
        """
        지정된 채널에 메시지를 게시합니다 (메시지는 JSON으로 직렬화됨).

        Args:
            channel (str): 게시할 채널 이름.
            message (Union[str, Dict, list]): 게시할 메시지.

        Returns:
            bool: 게시 성공 여부.
        """
        if not self.client:
            logger.error("Redis 클라이언트가 초기화되지 않아 publish 작업을 수행할 수 없습니다.")
            return False
        try:
            serialized_message = json.dumps(message)
            logger.debug(f"Redis PUBLISH: channel='{channel}'")
            await self.client.publish(channel, serialized_message)
            # logger.info(f"Redis PUBLISH 성공: channel='{channel}'")
            return True
        except TypeError as e:
            logger.error(f"Redis PUBLISH 실패: 메시지 JSON 직렬화 오류. channel='{channel}', msg_type={type(message)}, error={e}", exc_info=True)
            return False
        except redis.RedisError as e:
            logger.error(f"Redis PUBLISH 실패: Redis 오류 발생. channel='{channel}', error={e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Redis PUBLISH 실패: 알 수 없는 오류 발생. channel='{channel}', error={e}", exc_info=True)
            return False

    async def subscribe(
        self,
        channel: str,
        callback: Callable[[Any], Coroutine[Any, Any, None]] # 비동기 콜백 함수 타입 명시
    ):
        """
        채널을 구독하고 수신된 메시지를 처리하기 위한 비동기 콜백을 등록합니다.
        메시지 리스너는 백그라운드 작업으로 실행됩니다.

        Args:
            channel (str): 구독할 채널 이름.
            callback (Callable[[Any], Coroutine[Any, Any, None]]): 수신된 메시지(JSON 역직렬화됨)를
                                                                   처리하는 비동기 함수.
        """
        if not self.pubsub or not self.client: # pubsub 객체도 확인
            logger.error("Redis 클라이언트 또는 PubSub이 초기화되지 않아 subscribe 작업을 수행할 수 없습니다.")
            return

        await self.pubsub.subscribe(channel)
        logger.info(f"Subscribed to Redis channel: '{channel}'")

        async def message_handler(pubsub: redis.client.PubSub, chan: str, cb: Callable[[Any], Coroutine[Any, Any, None]]):
            """백그라운드에서 메시지를 듣고 처리하는 내부 함수"""
            logger.info(f"Starting message listener task for channel: '{chan}'")
            while True:
                try:
                    # listen()은 블러킹 호출처럼 보이지만 asyncio 환경에서 비동기적으로 작동
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=None) # 타임아웃 없이 대기
                    if message is not None and message['type'] == 'message':
                        logger.debug(f"Received message on channel '{message['channel']}': {message['data'][:100]}...")
                        try:
                            # decode_responses=True 이므로 data는 이미 문자열
                            parsed_message = json.loads(message['data'])
                            # 콜백 함수를 비동기적으로 실행
                            await cb(parsed_message)
                        except json.JSONDecodeError as e:
                            logger.error(f"Message processing error: JSON decode failed on channel '{chan}'. Data: '{message['data'][:100]}...'. Error: {e}", exc_info=True)
                        except Exception as e:
                            logger.error(f"Message processing error: Callback execution failed on channel '{chan}'. Error: {e}", exc_info=True)
                except redis.ConnectionError as e:
                     logger.error(f"Redis connection error in listener for channel '{chan}': {e}. Stopping listener.", exc_info=True)
                     break # 연결 오류 시 리스너 중지
                except Exception as e:
                    logger.error(f"Unexpected error in message listener for channel '{chan}': {e}. Stopping listener.", exc_info=True)
                    break # 예상치 못한 오류 시 리스너 중지
            logger.warning(f"Message listener task for channel '{chan}' stopped.")


        # 백그라운드에서 메시지 핸들러 실행
        # 중요: create_task의 결과를 저장하지 않으면 가비지 컬렉션될 수 있음 (상황에 따라 다름)
        #      클래스 인스턴스 등에 저장하여 참조를 유지하는 것이 더 안전할 수 있음
        #      여기서는 예시로 간단히 실행
        asyncio.create_task(message_handler(self.pubsub, channel, callback))


    async def close(self):
        """
        Redis 클라이언트 및 Pub/Sub 연결을 닫습니다.
        """
        # PubSub 닫기
        if self.pubsub:
            try:
                await self.pubsub.unsubscribe() # 모든 채널 구독 취소
                await self.pubsub.close()
                logger.info("Redis PubSub connection closed.")
                self.pubsub = None
            except Exception as e:
                logger.error(f"Error closing Redis PubSub connection: {e}", exc_info=True)

        # 클라이언트 닫기 (Connection Pool이 없으므로 직접 닫음)
        if self.client:
            try:
                await self.client.close()
                await self.client.wait_closed() # 닫힐 때까지 대기 (redis-py 4.1 이상)
                logger.info("Redis client connection closed.")
                self.client = None
            except Exception as e:
                logger.error(f"Error closing Redis client connection: {e}", exc_info=True)

# --- 사용 예시 ---
# async def my_callback(message: Any):
#     """메시지 수신 시 호출될 비동기 콜백 함수"""
#     print(f"Callback received: {message}")
#     # 여기서 수신된 메시지로 비동기 작업 수행 가능
#     await asyncio.sleep(0.1)

# async def main():
#     try:
#         # 환경 변수 또는 기본값으로 클라이언트 초기화
#         redis_client_v2 = DatabaseClientV2()
#     except (ValueError, ConnectionError) as e:
#         logger.critical(f"Failed to initialize DatabaseClientV2: {e}")
#         return

#     # Pub/Sub 구독 설정 (백그라운드 리스너 시작)
#     await redis_client_v2.subscribe("my_channel", my_callback)

#     # 잠시 대기 후 메시지 게시
#     await asyncio.sleep(1)
#     await redis_client_v2.publish("my_channel", {"user": "Alice", "action": "login"})
#     await redis_client_v2.publish("my_channel", {"user": "Bob", "action": "post", "content_id": 123})

#     # Get/Set 사용 예시
#     await redis_client_v2.set("user:alice", {"email": "alice@example.com", "status": "active"})
#     alice_data = await redis_client_v2.get("user:alice")
#     print(f"Alice's data: {alice_data}")

#     # 잠시 더 대기하여 메시지 처리 시간 확보
#     await asyncio.sleep(2)

#     # 애플리케이션 종료 시 연결 닫기
#     await redis_client_v2.close()

# if __name__ == "__main__":
#      asyncio.run(main())