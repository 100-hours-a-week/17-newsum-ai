# app/tools/trends/google_trends.py
import asyncio
import time # 배치 딜레이용
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger("GoogleTrendsTool")

PYTRENDS_AVAILABLE = False
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
    logger.info("pytrends library found.")
except ImportError:
    logger.warning("pytrends library not installed. Google Trends analysis disabled.")

class GoogleTrendsTool:
    """Google Trends 데이터를 가져오는 도구 (내부 재시도 포함)"""
    def __init__(self):
        # 설정값 로드
        self.timeframe = settings.PYTRENDS_TIMEFRAME
        self.geo = settings.PYTRENDS_GEO
        self.hl = settings.PYTRENDS_HL
        self.batch_delay_sec = settings.PYTRENDS_BATCH_DELAY_SEC
        # pytrends 객체는 호출 시점에 생성 (쓰레드 문제 방지)
        logger.info(f"GoogleTrendsTool initialized. Timeframe='{self.timeframe}', Geo='{self.geo}', HL='{self.hl}'")

    @retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES), # 재시도 횟수 설정 사용
        wait=wait_exponential(multiplier=1, min=3, max=15),
        retry=retry_if_exception_type(Exception), # pytrends 관련 예외 포함
        reraise=True
    )
    async def get_trends(self, keywords: List[str], trace_id: Optional[str] = None) -> Dict[str, float]:
        """키워드 목록에 대한 Google Trends 점수(평균 관심도)를 비동기적으로 반환"""
        if not PYTRENDS_AVAILABLE or not keywords:
            return {kw: 0.0 for kw in keywords}

        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.info(f"{log_prefix} Google Trends Tool: Starting analysis for {len(keywords)} keywords...")

        # 동기 라이브러리이므로 run_in_executor 사용
        loop = asyncio.get_running_loop()
        try:
            # 각 호출마다 pytrends 객체 생성 시도 (더 안정적일 수 있음)
            scores = await loop.run_in_executor(None, self._run_sync, keywords, trace_id)
            logger.info(f"{log_prefix} Google Trends Tool: Analysis complete.")
            return scores
        except Exception as e:
            logger.exception(f"{log_prefix} Google Trends Tool: Analysis failed after retries: {e}")
            return {kw: 0.0 for kw in keywords} # 실패 시 0점 반환

    def _run_sync(self, keywords: List[str], trace_id: Optional[str]) -> Dict[str, float]:
        """pytrends 요청을 동기적으로 실행하는 헬퍼"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not PYTRENDS_AVAILABLE: return {kw: 0.0 for kw in keywords}

        scores = {kw: 0.0 for kw in keywords}
        try:
            # pytrends 객체 생성 (요청마다 새로 생성)
            pytrends = TrendReq(hl=self.hl, tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)
        except Exception as e:
            logger.error(f"{log_prefix} Failed to initialize Pytrends: {e}. Returning 0 scores.")
            return scores

        batch_size = 5
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i + batch_size]
            logger.debug(f"{log_prefix} Requesting Google Trends for batch: {batch}")
            try:
                pytrends.build_payload(batch, cat=0, timeframe=self.timeframe, geo=self.geo, gprop='')
                interest_df = pytrends.interest_over_time()

                if not interest_df.empty:
                     # isPartial 처리 - 경고만 로깅하고 평균값 사용
                     if 'isPartial' in interest_df.columns and interest_df['isPartial'].any():
                          logger.warning(f"{log_prefix} Google Trends returned partial data for batch {batch}.")

                     for keyword in batch:
                          if keyword in interest_df.columns:
                               # 평균 관심도를 점수로 사용
                               keyword_score = float(interest_df[keyword].mean())
                               scores[keyword] = keyword_score
                else:
                    logger.warning(f"{log_prefix} No Google Trends data for batch: {batch}")

                # 배치 간 딜레이
                if i + batch_size < len(keywords):
                    time.sleep(self.batch_delay_sec)

            except Exception as e:
                logger.error(f"{log_prefix} Error processing Google Trends batch {batch}: {e}")
                # 해당 배치 키워드 점수는 0 유지

        return scores