# ai/test_llm_api.py

"""
LLM 큐 API 빠른 테스트 스크립트
"""

import asyncio
import httpx
import time

API_BASE = "http://localhost:8000/api/v1/llm"

async def test_llm_queue_api():
    """LLM 큐 API 테스트"""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("🧪 LLM 큐 API 테스트 시작...\n")
        
        # 1. 큐 상태 확인
        print("1️⃣ 큐 상태 확인...")
        try:
            response = await client.get(f"{API_BASE}/queue/status")
            print(f"   응답: {response.status_code}")
            print(f"   데이터: {response.json()}\n")
        except Exception as e:
            print(f"   ❌ 오류: {e}\n")
        
        # 2. LLM 작업 생성
        print("2️⃣ LLM 작업 생성...")
        task_payload = {
            "prompt": "Python으로 Hello World를 출력하는 코드를 작성해주세요.",
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        try:
            response = await client.post(f"{API_BASE}/tasks", json=task_payload)
            print(f"   응답: {response.status_code}")
            task_data = response.json()
            print(f"   데이터: {task_data}")
            
            if response.status_code == 202:
                task_id = task_data["task_id"]
                print(f"   ✅ 작업 생성됨: {task_id}\n")
                
                # 3. 결과 대기 및 조회
                print("3️⃣ 결과 대기 중...")
                max_attempts = 30  # 최대 30초 대기
                
                for attempt in range(max_attempts):
                    try:
                        result_response = await client.get(f"{API_BASE}/tasks/{task_id}/result")
                        result_data = result_response.json()
                        
                        if result_data["status"] == "completed":
                            print(f"   ✅ 완료! (시도: {attempt + 1})")
                            print(f"   결과: {result_data['result']['generated_text'][:200]}...")
                            break
                        elif result_data["status"] == "failed":
                            print(f"   ❌ 실패: {result_data.get('error')}")
                            break
                        else:
                            print(f"   ⏳ 처리 중... (시도: {attempt + 1})")
                            await asyncio.sleep(1)
                            
                    except Exception as e:
                        print(f"   ❌ 결과 조회 오류: {e}")
                        break
                else:
                    print("   ⚠️ 타임아웃 - 워커가 실행 중인지 확인하세요.")
            
        except Exception as e:
            print(f"   ❌ 작업 생성 오류: {e}\n")

if __name__ == "__main__":
    print("📡 LLM 큐 API 테스트")
    print("⚠️ 서버와 워커가 실행 중이어야 합니다.\n")
    
    try:
        asyncio.run(test_llm_queue_api())
    except KeyboardInterrupt:
        print("\n⚠️ 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 오류: {e}")
