# ai/check_llm_system.py

"""
LLM 큐 시스템 상태 확인 스크립트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.services.database_client import DatabaseClient
from app.services.llm_service import LLMService

async def check_redis():
    """Redis 연결 확인"""
    print("🔍 Redis 연결 확인...")
    try:
        db = DatabaseClient()
        await db.set("test_key", "test_value", expire=10)
        result = await db.get("test_key")
        await db.close()
        
        if result == "test_value":
            print("   ✅ Redis 연결 정상")
            return True
        else:
            print("   ❌ Redis 데이터 불일치")
            return False
    except Exception as e:
        print(f"   ❌ Redis 연결 실패: {e}")
        return False

async def check_llm_service():
    """LLM 서비스 확인"""
    print("🔍 LLM 서비스 확인...")
    try:
        llm = LLMService()
        # 간단한 테스트 요청
        result = await llm.generate_text(
            prompt="안녕하세요",
            max_tokens=10
        )
        await llm.close()
        
        if "generated_text" in result or "error" in result:
            print("   ✅ LLM 서비스 응답 확인")
            return True
        else:
            print("   ❌ LLM 서비스 응답 형식 오류")
            return False
    except Exception as e:
        print(f"   ❌ LLM 서비스 연결 실패: {e}")
        return False

async def check_queue_operations():
    """큐 동작 확인"""
    print("🔍 큐 동작 확인...")
    try:
        db = DatabaseClient()
        
        # 테스트 작업 추가
        test_task = {"test": "data"}
        queue_len = await db.lpush("test_queue", test_task)
        
        # 작업 가져오기
        retrieved = await db.brpop("test_queue", timeout=5)
        
        await db.close()
        
        if retrieved and retrieved.get("test") == "data":
            print("   ✅ 큐 동작 정상")
            return True
        else:
            print("   ❌ 큐 동작 오류")
            return False
    except Exception as e:
        print(f"   ❌ 큐 동작 확인 실패: {e}")
        return False

async def main():
    """메인 체크 함수"""
    print("🚀 LLM 큐 시스템 상태 확인\n")
    
    results = []
    
    # 각 구성 요소 확인
    results.append(await check_redis())
    results.append(await check_llm_service())
    results.append(await check_queue_operations())
    
    print("\n📊 결과 요약:")
    if all(results):
        print("✅ 모든 구성 요소가 정상 작동합니다!")
        print("\n🎉 시스템 사용 준비 완료:")
        print("   1. python main.py (서버 실행)")
        print("   2. python run_llm_worker.py (워커 실행)")
        print("   3. python test_llm_api.py (API 테스트)")
    else:
        print("❌ 일부 구성 요소에 문제가 있습니다.")
        print("\n🔧 확인 사항:")
        print("   - Redis 서버 실행 상태")
        print("   - .env 파일 설정")
        print("   - LLM 서버 연결")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ 확인이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 확인 중 오류: {e}")
