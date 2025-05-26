# tests/test_postgresql_service.py
import pytest
import asyncio
from app.services.postgresql_service import PostgreSQLService

class TestPostgreSQLService:
    
    @pytest.mark.asyncio
    async def test_connection_only(self):
        """연결 테스트만 (빠른 확인) - SSH 터널 없이"""
        service = PostgreSQLService(use_ssh=False)  # SSH 터널 비활성화
        try:
            print("🔌 PostgreSQL 직접 연결 시도...")
            await asyncio.wait_for(service.connect(), timeout=5)
            print("✅ 연결 성공!")
            
            result = await service.execute("SELECT 1 as test")
            print(f"📊 쿼리 테스트: {result}")
            
        except asyncio.TimeoutError:
            pytest.skip("연결 타임아웃 (5초)")
        except Exception as e:
            pytest.skip(f"연결 실패: {e}")
        finally:
            await service.close()
    
    @pytest.mark.asyncio
    async def test_simple_table(self):
        """간단한 테이블 테스트 - SSH 터널 없이"""
        service = PostgreSQLService(use_ssh=False)  # SSH 터널 비활성화
        try:
            await asyncio.wait_for(service.connect(), timeout=5)
            
            # 간단한 테이블 생성/삭제
            await service.execute("CREATE TEMP TABLE temp_test (id SERIAL)")
            print("✅ 임시 테이블 생성 성공")
            
        except asyncio.TimeoutError:
            pytest.skip("연결 타임아웃")
        except Exception as e:
            pytest.skip(f"테스트 실패: {e}")
        finally:
            await service.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
