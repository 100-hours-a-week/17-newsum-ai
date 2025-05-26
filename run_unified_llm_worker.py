# ai/run_unified_llm_worker.py

"""
통합 LLM 워커 실행 스크립트
모든 유형의 LLM 작업을 처리하는 통합 워커를 실행합니다.
- llm_request_queue: 일반 LLM 작업 (기존 워크플로우)
- llm_chat_queue: 채팅 작업  
- llm_summary_queue: 대화 요약 작업
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.workers.llm_worker import main

if __name__ == "__main__":
    print("🚀 Unified LLM Worker 시작...")
    print("중지하려면 Ctrl+C를 누르세요.")
    print("처리 대상 큐:")
    print("  - llm_chat_queue (채팅 작업)")
    print("  - llm_summary_queue (요약 작업)")  
    print("  - llm_request_queue (일반 LLM 작업)")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ 통합 LLM 워커가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 워커 실행 중 오류: {e}")
        sys.exit(1)
