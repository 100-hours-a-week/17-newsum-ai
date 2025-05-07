# app/tools/social/twitter.py (수정된 최종 버전)

import asyncio
import time  # 시간 추적을 위해 임포트
import re    # 쿼리 생성을 위해 임포트
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.settings import settings # 설정 임포트
from app.utils.logger import get_logger # 로거 임포트

# tweepy 동적 임포트 시도
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    tweepy = None # 사용 불가 시 None 할당

logger = get_logger(__name__) # 로거 초기화

class TwitterTool:
    """
    Tweepy를 사용하여 트위터 API v2와 상호작용하는 도구입니다.
    Rate Limit 발생 시 긴 대기를 피하고 오류를 반환하도록 수정되었습니다.
    """

    def __init__(self):
        """TwitterTool 초기화"""
        self.client: Optional[tweepy.Client] = None # 클라이언트 변수 초기화
        self.bearer_token = settings.TWITTER_BEARER_TOKEN # 설정에서 Bearer 토큰 로드

        # Rate Limit 관리를 위한 변수 추가
        self.last_api_call_time: float = 0.0
        # 설정에서 최소 요청 간격 로드 (기본값 1초, 현재 5.0초로 설정됨)
        self.min_request_interval: float = 5.0 #getattr(settings, 'TWITTER_MIN_REQUEST_INTERVAL_SEC', 5.0)
        self.rate_limit_lock = asyncio.Lock() # 동시 접근 제어를 위한 Lock

        # 라이브러리 또는 토큰 부재 시 경고 로깅 및 초기화 중단
        if not TWEEPY_AVAILABLE:
            logger.warning("tweepy 라이브러리가 설치되지 않았습니다. TwitterTool이 작동하지 않습니다.")
            return
        if not self.bearer_token:
            logger.warning("settings에 TWITTER_BEARER_TOKEN이 설정되지 않았습니다. TwitterTool이 작동하지 않습니다.")
            return

        try:
            logger.info("Twitter 클라이언트 초기화 중...")
            # --- *** 중요 변경: wait_on_rate_limit=False 설정 *** ---
            # Tweepy가 자동으로 긴 시간 대기하는 것을 방지하고 대신 예외를 발생시키도록 함
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                wait_on_rate_limit=False # <<<--- 여기를 False로 변경!
            )
            logger.info(f"Twitter 클라이언트 초기화 완료. wait_on_rate_limit=False, 최소 요청 간격: {self.min_request_interval}초")
        except Exception as e:
            logger.error(f"Twitter 클라이언트 초기화 실패: {e}", exc_info=True)
            self.client = None # 실패 시 클라이언트 None 설정

    async def _wait_if_needed(self, trace_id: Optional[str] = None):
        """
        API 호출 속도를 조절하기 위해 필요한 경우 대기합니다. (자체 Throttling)
        """
        async with self.rate_limit_lock:
            current_time = time.monotonic()
            time_since_last_call = current_time - self.last_api_call_time
            extra_log_data = {'trace_id': trace_id}

            if time_since_last_call < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last_call
                logger.debug(f"Throttling Twitter API call. Waiting for {wait_time:.3f} seconds.", extra=extra_log_data)
                await asyncio.sleep(wait_time)
                self.last_api_call_time = time.monotonic()
            else:
                self.last_api_call_time = current_time

    @retry(
        stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS),
        # --- MODIFIED: max 대기 시간 조정 (예: 60초) ---
        wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def search_recent_tweets(self, keyword: str, max_results: int, trace_id: str) -> List[Dict[str, Any]]:
        """
        주어진 키워드로 최근 트윗을 검색합니다. Rate Limit 시 예외를 발생시킵니다.
        """
        if not self.client:
            logger.warning("Twitter 클라이언트를 사용할 수 없어 검색을 건너<0xEB><0x9C><0x95>니다.", extra={'trace_id': trace_id})
            return []

        extra_log_data = {'trace_id': trace_id, 'keyword': keyword}

        # 쿼리 생성 (기존과 동일)
        safe_keyword = keyword.strip().replace('"', '')
        if not safe_keyword:
            logger.warning("키워드가 비어 있어 Twitter 검색을 건너<0xEB><0x9C><0x95>니다.", extra=extra_log_data)
            return []
        quoted_keyword = f'"{safe_keyword}"' if ' ' in safe_keyword else safe_keyword
        language_filter = "(lang:en)" # 또는 설정에서 가져오기
        base_query = f"{quoted_keyword} -is:retweet {language_filter}"
        if len(base_query) > 500:
            logger.warning(f"Generated Twitter query is long ({len(base_query)} chars), potential issues.", extra=extra_log_data)
        query = base_query

        logger.info(f"'{keyword}'에 대한 최근 트윗 검색 중 (최대: {max_results}, Query: '{query}')", extra=extra_log_data)
        api_max_results = max(10, min(max_results, 100))

        try:
            # 자체 Throttling 대기
            await self._wait_if_needed(trace_id)

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.search_recent_tweets(
                    query=query,
                    max_results=api_max_results,
                    tweet_fields=["created_at", "public_metrics", "conversation_id", "lang"]
                )
            )

            results = []
            if response.data:
                for tweet in response.data:
                    tweet_url = f"https://twitter.com/anyuser/status/{tweet.id}"
                    results.append({
                        "url": tweet_url,
                        "text": tweet.text,
                        "tweet_id": str(tweet.id),
                        "source": "Twitter",
                        "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                        "lang": tweet.lang,
                    })
                logger.info(f"{len(results)}개의 트윗을 찾았습니다.", extra=extra_log_data)
            else:
                logger.info("쿼리에 해당하는 최근 트윗을 찾을 수 없습니다.", extra=extra_log_data)
            return results

        # --- *** 중요 변경: TooManyRequests 예외 처리 추가 *** ---
        except tweepy.errors.TooManyRequests as e:
            # Rate Limit (429) 발생 시 경고 로깅 후 빈 리스트 반환 (재시도는 tenacity가 담당)
            logger.warning(f"Twitter API rate limit hit (429) during search for '{keyword}'. Returning empty list for this attempt. Details: {e}", extra=extra_log_data)
            return [] # 현재 시도 실패, 빈 리스트 반환
        # --------------------------------------------------------
        except tweepy.errors.TweepyException as e:
             api_codes = getattr(e, 'api_codes', [])
             api_messages = getattr(e, 'api_messages', [])
             logger.error(f"검색 중 Tweepy API 오류 발생: {e} (Codes: {api_codes}, Messages: {api_messages})", extra=extra_log_data)
             if isinstance(e, tweepy.errors.BadRequest):
                  logger.error(f"BadRequest 발생 시 사용된 쿼리: '{query}'", extra=extra_log_data)
             return [] # 다른 Tweepy 오류 시에도 빈 리스트 반환
        except Exception as e:
            logger.error(f"Twitter 검색 중 예상치 못한 오류 발생: {e}", exc_info=True, extra=extra_log_data)
            raise # tenacity 재시도를 위해 예외 다시 발생

    @retry(
        stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS),
        # --- MODIFIED: max 대기 시간 조정 (예: 60초) ---
        wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN, max=60), # 일관성 및 효과 위해 max값 조정
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def get_tweet_details(self, tweet_id: str, trace_id: str) -> Optional[Dict[str, Any]]:
        """주어진 ID의 트윗 상세 정보를 가져옵니다. Rate Limit 시 예외를 발생시킵니다."""
        if not self.client:
            logger.warning("Twitter 클라이언트를 사용할 수 없어 상세 정보 조회를 건너<0xEB><0x9C><0x95>니다.", extra={'trace_id': trace_id})
            return None

        extra_log_data = {'trace_id': trace_id, 'tweet_id': tweet_id}
        logger.debug("API를 통해 트윗 상세 정보 가져오는 중...", extra=extra_log_data)
        try:
            # 자체 Throttling 대기
            await self._wait_if_needed(trace_id)

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self.client.get_tweet(
                tweet_id,
                tweet_fields=["created_at", "public_metrics", "author_id", "conversation_id", "lang"],
                expansions=["author_id"],
                user_fields=["username", "name"]
            ))
            if response.data:
                tweet = response.data
                author = next((user for user in response.includes.get('users', []) if user.id == tweet.author_id), None)
                author_name = author.username if author else str(tweet.author_id)

                return {
                    "text": tweet.text,
                    "author": author_name,
                    "timestamp": tweet.created_at.isoformat() if tweet.created_at else None,
                    "likes": tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                    "raw_data": tweet.data,
                }
            else:
                logger.warning("API를 통해 트윗을 찾을 수 없음", extra=extra_log_data)
                return None
        except tweepy.errors.NotFound:
            logger.warning("API를 통해 트윗을 찾을 수 없음 (404)", extra=extra_log_data)
            return None
        # --- *** 중요 변경: TooManyRequests 예외 처리 추가 *** ---
        except tweepy.errors.TooManyRequests as e:
            # Rate Limit (429) 발생 시 경고 로깅 후 None 반환 (재시도는 tenacity가 담당)
            logger.warning(f"Twitter API rate limit hit (429) during get_tweet_details for ID '{tweet_id}'. Returning None for this attempt. Details: {e}", extra=extra_log_data)
            return None # 현재 시도 실패, None 반환
        # --------------------------------------------------------
        except tweepy.errors.TweepyException as e:
             api_codes = getattr(e, 'api_codes', [])
             api_messages = getattr(e, 'api_messages', [])
             logger.error(f"API 통해 트윗 가져오기 오류: {e} (Codes: {api_codes}, Messages: {api_messages})", extra=extra_log_data)
             return None # 다른 Tweepy 오류 시에도 None 반환
        except Exception as e:
            logger.error(f"API 통해 트윗 가져오기 중 예상치 못한 오류: {e}", exc_info=True, extra=extra_log_data)
            raise # tenacity 재시도를 위해 예외 발생

    # tweepy는 보통 명시적인 close 메서드가 필요 없음
    # async def close(self):
    #     pass