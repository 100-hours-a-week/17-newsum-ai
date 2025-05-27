# ai/run_llm_worker.py

"""
통합 LLM 워커 실행 스크립트 (기본)
모든 유형의 LLM 작업을 처리하는 통합 워커를 실행합니다.

주의: 이 파일은 run_unified_llm_worker.py와 동일합니다.
      통일성을 위해 run_unified_llm_worker.py 사용을 권장합니다.
"""

import asyncio
import sys
import os
from pathlib import Path
from app.utils.logger import setup_logging

# 🚀 로깅 설정을 가장 먼저 호출합니다! 🚀
setup_logging()

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.workers.chat_worker import worker_loop

if __name__ == "__main__":
    print("🚀 LLM Worker 시작...")
    print("중지하려면 Ctrl+C를 누르세요.")

    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        print("\n⚠️ 워커가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 워커 실행 중 오류: {e}")
        sys.exit(1)
