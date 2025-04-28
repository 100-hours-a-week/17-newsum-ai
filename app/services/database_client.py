# app/services/database_client.py 수정 제안
import redis.asyncio as redis
import json # json 라이브러리 import
from typing import Optional, Any

class DatabaseClient:
    def __init__(self, host='localhost', port=6379, db=0):
        # 실제 연결 풀 설정 등은 더 견고하게 구현 필요
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True) # decode_responses=True 권장

    async def set(self, key: str, value: Any, expire: Optional[int] = None):
        """Redis에 키-값을 저장합니다. 값은 JSON으로 직렬화됩니다."""
        try:
            # --- 수정: 값을 JSON 문자열로 변환 ---
            serialized_value = json.dumps(value)
            await self.client.set(key, serialized_value, ex=expire)
        except Exception as e:
            # 로깅 추가 권장
            print(f"Error setting key {key} in Redis: {e}")
            raise

    async def get(self, key: str) -> Optional[Any]:
        """Redis에서 키에 해당하는 값을 가져옵니다. 값은 JSON으로 역직렬화됩니다."""
        try:
            serialized_value = await self.client.get(key)
            if serialized_value:
                # --- 수정: JSON 문자열을 파이썬 객체(dict 등)로 변환 ---
                return json.loads(serialized_value)
            return None
        except Exception as e:
             # 로깅 추가 권장
            print(f"Error getting key {key} from Redis: {e}")
            raise # 또는 None 반환 등 오류 처리 정책 결정