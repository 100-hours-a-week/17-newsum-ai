# tests/test_performance.py

import pytest
import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.api.v1.schemas import AsyncComicRequest, RequestDataPayload


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """성능 벤치마크 테스트"""

    def test_schema_validation_performance(self, benchmark):
        """스키마 유효성 검사 성능 테스트"""
        def create_request_schema():
            return AsyncComicRequest(
                writer_id="benchmark_writer",
                data=RequestDataPayload(
                    query="성능 테스트를 위한 긴 쿼리 문자열 " * 10
                )
            )
        
        result = benchmark(create_request_schema)
        assert result.writer_id == "benchmark_writer"

    def test_schema_serialization_performance(self, benchmark):
        """스키마 직렬화 성능 테스트"""
        request = AsyncComicRequest(
            writer_id="benchmark_writer",
            data=RequestDataPayload(
                query="성능 테스트 쿼리"
            )
        )
        
        def serialize_schema():
            return request.model_dump()
        
        result = benchmark(serialize_schema)
        assert "writer_id" in result

    @pytest.mark.asyncio
    async def test_background_task_trigger_performance(self, benchmark):
        """백그라운드 태스크 트리거 성능 테스트"""
        from app.api.v1.background_tasks import trigger_workflow_task
        
        mock_background_tasks = MagicMock()
        mock_compiled_app = AsyncMock()
        mock_db_client = AsyncMock()
        mock_db_client.set.return_value = True
        
        async def trigger_task():
            return await trigger_workflow_task(
                query="성능 테스트 쿼리",
                config={"writer_id": "benchmark_writer"},
                background_tasks=mock_background_tasks,
                compiled_app=mock_compiled_app,
                db_client=mock_db_client
            )
        
        result = await benchmark(trigger_task)
        assert isinstance(result, str)

    def test_uuid_generation_performance(self, benchmark):
        """UUID 생성 성능 테스트"""
        def generate_uuid():
            return str(uuid.uuid4())
        
        result = benchmark(generate_uuid)
        assert len(result) == 36

    def test_helper_function_performance(self, benchmark):
        """헬퍼 함수 성능 테스트"""
        from app.api.v1.background_tasks import _g
        
        test_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": "target_value"
                    }
                }
            }
        }
        
        def call_helper():
            return _g(test_data, "level1", "level2", "level3", "level4")
        
        result = benchmark(call_helper)
        assert result == "target_value"


@pytest.mark.slow
class TestStressTests:
    """스트레스 테스트"""

    @pytest.mark.asyncio
    async def test_concurrent_schema_validation(self):
        """동시 스키마 유효성 검사 스트레스 테스트"""
        async def create_schema(i):
            return AsyncComicRequest(
                writer_id=f"stress_writer_{i}",
                data=RequestDataPayload(
                    query=f"스트레스 테스트 쿼리 {i}"
                )
            )
        
        # 100개의 동시 스키마 생성
        tasks = [create_schema(i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 100
        assert all(r.data.query.startswith("스트레스 테스트") for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_background_tasks(self):
        """동시 백그라운드 태스크 스트레스 테스트"""
        from app.api.v1.background_tasks import trigger_workflow_task
        
        mock_background_tasks = MagicMock()
        mock_compiled_app = AsyncMock()
        mock_db_client = AsyncMock()
        mock_db_client.set.return_value = True
        
        async def trigger_task(i):
            return await trigger_workflow_task(
                query=f"스트레스 테스트 쿼리 {i}",
                config={"writer_id": f"stress_writer_{i}"},
                background_tasks=mock_background_tasks,
                compiled_app=mock_compiled_app,
                db_client=mock_db_client
            )
        
        # 50개의 동시 태스크 트리거
        tasks = [trigger_task(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 50
        assert all(isinstance(r, str) for r in results)
        assert len(set(results)) == 50  # 모든 결과가 고유해야 함

    def test_large_data_schema_validation(self):
        """대용량 데이터 스키마 유효성 검사 테스트"""
        # 큰 쿼리 생성 (1MB 정도)
        large_query = "A" * (1024 * 1024)
        
        start_time = time.time()
        request = AsyncComicRequest(
            writer_id="large_data_writer",
            data=RequestDataPayload(query=large_query)
        )
        end_time = time.time()
        
        # 1초 이내에 처리되어야 함
        assert end_time - start_time < 1.0
        assert len(request.data.query) == 1024 * 1024

    def test_memory_usage_schema_creation(self):
        """스키마 생성 시 메모리 사용량 테스트"""
        import gc
        import sys
        
        # 가비지 컬렉션 실행
        gc.collect()
        
        # 초기 메모리 측정
        initial_objects = len(gc.get_objects())
        
        # 많은 스키마 객체 생성
        schemas = []
        for i in range(1000):
            schema = AsyncComicRequest(
                writer_id=f"memory_test_writer_{i}",
                data=RequestDataPayload(
                    query=f"메모리 테스트 쿼리 {i}"
                )
            )
            schemas.append(schema)
        
        # 메모리 사용량 확인
        final_objects = len(gc.get_objects())
        objects_created = final_objects - initial_objects
        
        # 생성된 객체 수가 예상 범위 내에 있는지 확인
        assert objects_created > 1000  # 최소한 생성한 스키마 수보다는 많아야 함
        assert objects_created < 10000  # 너무 많은 객체가 생성되지 않았는지 확인


@pytest.mark.performance
class TestMemoryLeakTests:
    """메모리 누수 테스트"""

    @pytest.mark.asyncio
    async def test_no_memory_leak_in_background_tasks(self):
        """백그라운드 태스크에서 메모리 누수 없음을 확인"""
        import gc
        from app.api.v1.background_tasks import trigger_workflow_task
        
        mock_background_tasks = MagicMock()
        mock_compiled_app = AsyncMock()
        mock_db_client = AsyncMock()
        mock_db_client.set.return_value = True
        
        # 초기 메모리 상태
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # 반복적으로 태스크 실행
        for i in range(100):
            await trigger_workflow_task(
                query=f"메모리 누수 테스트 {i}",
                config={"writer_id": f"leak_test_{i}"},
                background_tasks=mock_background_tasks,
                compiled_app=mock_compiled_app,
                db_client=mock_db_client
            )
            
            # 주기적으로 가비지 컬렉션
            if i % 10 == 0:
                gc.collect()
        
        # 최종 메모리 상태
        gc.collect()
        final_objects = len(gc.get_objects())
        objects_diff = final_objects - initial_objects
        
        # 메모리 누수가 없다면 객체 수 증가가 크지 않아야 함
        assert objects_diff < 1000, f"잠재적 메모리 누수 감지: {objects_diff}개 객체 증가"


# 테스트 실행을 위한 pytest 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
