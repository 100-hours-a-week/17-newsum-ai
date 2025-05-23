# ai/app/workers/run_slm_worker.py

import asyncio
import argparse
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from app.workers.slm_worker import SLMWorker
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """SLM 워커 메인 실행 함수"""
    parser = argparse.ArgumentParser(description="SLM Worker Process")
    parser.add_argument(
        "--worker-id", 
        required=True, 
        help="Worker ID (e.g., worker-1)"
    )
    parser.add_argument(
        "--request-channel", 
        required=True, 
        help="Redis request channel to subscribe"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting SLM Worker with config:")
    logger.info(f"  Worker ID: {args.worker_id}")
    logger.info(f"  Request Channel: {args.request_channel}")
    logger.info(f"  Log Level: {args.log_level}")
    logger.info(f"  vLLM Concurrent Requests: {settings.WORKER_MAX_CONCURRENT_VLLM_REQUESTS}")
    
    # 워커 인스턴스 생성 및 시작
    worker = SLMWorker(
        worker_id=args.worker_id,
        request_channel=args.request_channel
    )
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.exception(f"Worker failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
