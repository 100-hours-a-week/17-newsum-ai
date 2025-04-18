# run.py
import asyncio
import schedule
import time
import os
from src.workflow import workflow
from src.core.utils import logger

# Windows에서 asyncio 관련 설정 (선택적)
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        logger.info("Windows asyncio policy set.")
    except Exception as e:
        logger.warning(f"Could not set Windows asyncio policy: {e}")

async def run_job():
    """워크플로우 작업을 실행하는 비동기 함수"""
    logger.info("Starting scheduled workflow job...")
    try:
        # 워크플로우 실행 (초기 상태는 비어 있음)
        async for event in workflow.astream({}):
            # 각 단계별 결과 로깅 (선택적, 상세 로깅)
            # logger.debug(f"Workflow Event: {event}")
            pass
        logger.info("Workflow job finished.")
    except Exception as e:
        logger.exception("Unhandled error during workflow execution:") # 스택 트레이스 포함 로깅

def job_wrapper():
    """비동기 함수를 동기적으로 실행하기 위한 래퍼"""
    try:
        asyncio.run(run_job())
    except RuntimeError as e:
        # 이벤트 루프가 이미 실행 중인 경우 (예: Jupyter 노트북 환경)
        if "cannot run event loop while another loop is running" in str(e):
            logger.warning("Event loop already running. Using existing loop.")
            # 현재 루프에서 실행 시도 (환경에 따라 조정 필요)
            loop = asyncio.get_event_loop()
            loop.create_task(run_job())
        else:
            raise e

if __name__ == "__main__":
    logger.info("Scheduler started. Waiting for the next scheduled run...")
    # 매 시간 5분에 작업 실행 스케줄링
    schedule.every().hour.at(":05").do(job_wrapper)
    # schedule.every(10).minutes.do(job_wrapper) # 테스트용: 10분마다 실행

    # 즉시 1회 실행 (테스트용)
    logger.info("Running initial job immediately...")
    job_wrapper()

    while True:
        schedule.run_pending()
        time.sleep(30) # 30초마다 스케줄 확인