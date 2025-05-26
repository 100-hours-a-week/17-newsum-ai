# tests/connection_test.py
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.postgresql_service import PostgreSQLService

async def test_connection():
    print("🔌 PostgreSQL 직접 연결 테스트 (SSH 터널 없이)...")
    
    service = PostgreSQLService(use_ssh=False)  # SSH 터널 비활성화
    try:
        print("⏳ 직접 연결 시도 중...")
        await asyncio.wait_for(service.connect(), timeout=5)
        print("✅ 연결 성공!")
        
        result = await service.execute("SELECT 1")
        print(f"✅ 쿼리 성공: {result}")
        
    except asyncio.TimeoutError:
        print("❌ 연결 타임아웃 (5초)")
        return False
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return False
    finally:
        await service.close()
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_connection())
        print("🎉 테스트 완료" if success else "💥 테스트 실패")
    except KeyboardInterrupt:
        print("⚠️ 중단됨")
