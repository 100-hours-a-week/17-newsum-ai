# app/services/database_client.py

import redis.asyncio as redis

class DatabaseClient:
    """
    Redis를 이용한 상태 관리 클라이언트 (간단한 key-value 저장)
    """

    def __init__(self, redis_url="redis://localhost:6379/0"):
        self.client = redis.from_url(redis_url)

    async def set(self, key: str, value: str, expire: int = 3600):
        await self.client.set(key, value, ex=expire)

    async def get(self, key: str) -> str:
        value = await self.client.get(key)
        return value.decode("utf-8") if value else None

    async def delete(self, key: str):
        await self.client.delete(key)
