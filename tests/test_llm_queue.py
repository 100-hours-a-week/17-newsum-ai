# tests/test_llm_queue.py

import pytest
import asyncio
import uuid
from app.services.database_client import DatabaseClient
from app.services.llm_service import LLMService

class TestLLMQueue:
    """LLM 큐 시스템 테스트"""
    
    @pytest.mark.asyncio
    async def test_queue_operations(self):
        """큐 기본 동작 테스트"""
        db = DatabaseClient()
        
        try:
            # 테스트 작업 데이터
            task_data = {
                "task_id": str(uuid.uuid4()),
                "payload": {
                    "prompt": "테스트 프롬프트",
                    "max_tokens": 100
                }
            }
            
            # 큐에 작업 추가
            queue_len = await db.lpush("test_llm_queue", task_data)
            assert queue_len is not None
            assert queue_len > 0
            
            # 큐에서 작업 가져오기
            retrieved_task = await db.brpop("test_llm_queue", timeout=5)
            assert retrieved_task is not None
            assert retrieved_task["task_id"] == task_data["task_id"]
            
            print("✅ 큐 기본 동작 테스트 통과")
            
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_queue_timeout(self):
        """큐 타임아웃 테스트"""
        db = DatabaseClient()
        
        try:
            # 빈 큐에서 타임아웃 테스트
            result = await db.brpop("empty_test_queue", timeout=2)
            assert result is None
            
            print("✅ 큐 타임아웃 테스트 통과")
            
        finally:
            await db.close()
    
    @pytest.mark.asyncio
    async def test_result_storage(self):
        """결과 저장 테스트"""
        db = DatabaseClient()
        
        try:
            task_id = str(uuid.uuid4())
            result_key = f"llm_result:{task_id}"
            
            test_result = {
                "generated_text": "테스트 결과",
                "task_id": task_id
            }
            print(test_result)
            # 결과 저장
            await db.set(result_key, test_result, expire=60)
            
            # 결과 조회
            retrieved_result = await db.get(result_key)
            assert retrieved_result is not None
            assert retrieved_result["task_id"] == task_id
            
            print("✅ 결과 저장 테스트 통과")
            
        finally:
            await db.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
