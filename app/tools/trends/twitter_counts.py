# app/tools/trends/twitter_counts.py
import aiohttp
import asyncio
from datetime import datetime, timedelta, timezone
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Dict, Optional, Tuple
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger("TwitterCountsTool")

class TwitterCountsTool:
    """Twitter Recent Tweet Counts API를 사용하여 카운트를 가져오는 도구"""
    def __init__(self):
        self.bearer_token = settings.TWITTER_BEARER_TOKEN
        self.api_url = "https://api.twitter.com/2/tweets/counts/recent"
        self.call_delay_sec = settings.TWITTER_COUNTS_DELAY_SEC
        if not self.bearer_token:
            logger.error("TWITTER_BEARER_TOKEN is missing in settings. Twitter counts will be disabled.")

    @retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES), # 설정 재시도 횟수 사용
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # 429 (Rate Limit) 에러 시 재시도하지 않도록 설정 가능 (또는 더 긴 대기)
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True
    )
    async def get_recent_count(self, keyword: str, session: aiohttp.ClientSession, trace_id: Optional[str] = None) -> float:
        """단일 키워드에 대한 최근 7일 트윗 카운트 조회"""
        if not self.bearer_token: return 0.0
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Twitter Tool: Getting count for '{keyword}'")

        # API 파라미터 설정 (최근 7일)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=7)
        params = {
            "query": keyword,
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        headers = {"Authorization": f"Bearer {self.bearer_token}"}

        try:
            async with session.get(self.api_url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 429:
                    logger.warning(f"{log_prefix} Twitter API rate limit hit for '{keyword}'. Returning 0.")
                    # 재시도 대신 0 반환 (또는 더 긴 sleep 후 재시도 필요)
                    return 0.0
                response.raise_for_status() # 429 외 다른 HTTP 오류 시 예외 발생

                data = await response.json()
                count = data.get('meta', {}).get('total_tweet_count', 0)
                logger.debug(f"{log_prefix} Twitter count for '{keyword}': {count}")
                return float(count)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
             logger.error(f"{log_prefix} Twitter Tool error for '{keyword}': {e}")
             raise # tenacity 재시도 위해 예외 다시 발생
        except Exception as e:
             logger.exception(f"{log_prefix} Twitter Tool unexpected error for '{keyword}': {e}")
             return 0.0 # 예상치 못한 오류 시 0 반환

    async def get_counts_for_keywords(self, keywords: List[str], trace_id: Optional[str] = None) -> Dict[str, float]:
        """키워드 리스트에 대한 트윗 카운트를 비동기적으로 조회"""
        if not self.bearer_token or not keywords:
             return {kw: 0.0 for kw in keywords}

        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.info(f"{log_prefix} Twitter Tool: Starting counts analysis for {len(keywords)} keywords...")
        scores = {kw: 0.0 for kw in keywords}

        # 외부에서 세션 관리하도록 변경 (NewsCollector와 일관성)
        # async with aiohttp.ClientSession() as session: # 또는 세션을 인자로 받기
        session = aiohttp.ClientSession() # 임시. 외부 세션 사용 권장.
        tasks = []
        try:
            for keyword in keywords:
                # 각 키워드에 대한 비동기 작업 생성
                tasks.append(self.get_recent_count(keyword, session, trace_id))
                # API 호출 간 딜레이 적용 (Rate Limit 방지)
                await asyncio.sleep(self.call_delay_sec)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 처리
            for i, result in enumerate(results):
                keyword = keywords[i]
                if isinstance(result, (int, float)):
                    scores[keyword] = result
                elif isinstance(result, Exception):
                    logger.error(f"{log_prefix} Twitter count task failed for '{keyword}': {result}")
                    # 키워드 점수는 0 유지
        finally:
             await session.close() # 세션 닫기

        logger.info(f"{log_prefix} Twitter Tool: Counts analysis complete.")
        return scores