# app/tools/social/twitter.py
import asyncio
from typing import List, Dict, Optional
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
    """Tweepy를 사용하여 트위터 API v2와 상호작용하는 도구입니다."""

    def __init__(self):
        """TwitterTool 초기화"""
        self.client: Optional[tweepy.Client] = None # 클라이언트 변수 초기화
        self.bearer_token = settings.TWITTER_BEARER_TOKEN # 설정에서 Bearer 토큰 로드

        # 라이브러리 또는 토큰 부재 시 경고 로깅 및 초기화 중단
        if not TWEEPY_AVAILABLE:
            logger.warning("tweepy 라이브러리가 설치되지 않았습니다. TwitterTool이 작동하지 않습니다.")
            return
        if not self.bearer_token:
            logger.warning("settings에 TWITTER_BEARER_TOKEN이 설정되지 않았습니다. TwitterTool이 작동하지 않습니다.")
            return

        try:
            logger.info("Twitter 클라이언트 초기화 중...")
            # tweepy 클라이언트 생성, Rate Limit 시 자동 대기 활성화
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                wait_on_rate_limit=True
            )
            logger.info("Twitter 클라이언트 초기화 완료.")
        except Exception as e:
            logger.error(f"Twitter 클라이언트 초기화 실패: {e}", exc_info=True)
            self.client = None # 실패 시 클라이언트 None 설정

    @retry(
        stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS), # 설정된 재시도 횟수 사용
        wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN, max=settings.TOOL_RETRY_WAIT_MAX), # 지수적 대기 시간 적용
        retry=retry_if_exception_type(Exception), # 모든 예외 발생 시 재시도
        reraise=True # 모든 재시도 실패 시 예외 다시 발생
    )
    async def search_recent_tweets(self, keyword: str, max_results: int, trace_id: str) -> List[Dict[str, str]]:
        """
        주어진 키워드로 최근 트윗을 검색합니다.

        Args:
            keyword (str): 검색어.
            max_results (int): 반환할 최대 트윗 수 (API 호출당 최대 100개).
            trace_id (str): 로깅을 위한 추적 ID.

        Returns:
            List[Dict[str, str]]: 트윗 정보(url, text, tweet_id 등)를 담은 사전 목록.
        """
        # 클라이언트 사용 불가 시 빈 리스트 반환
        if not self.client:
            logger.warning("Twitter 클라이언트를 사용할 수 없어 검색을 건너<0xEB><0x9C><0x95>니다.", extra={'trace_id': trace_id})
            return []

        extra_log_data = {'trace_id': trace_id, 'keyword': keyword} # 로깅용 추가 데이터
        logger.info(f"'{keyword}'에 대한 최근 트윗 검색 중 (최대: {max_results})", extra=extra_log_data)

        # API 제한(최소 10, 최대 100)에 맞게 max_results 조정
        api_max_results = max(10, min(max_results, 100))
        # 검색 쿼리: 리트윗 제외, 한국어 또는 영어 트윗 대상
        query = f"{keyword} -is:retweet lang:ko OR lang:en"

        try:
            loop = asyncio.get_running_loop() # 현재 이벤트 루프 가져오기
            # tweepy 클라이언트 메서드는 동기적이므로 run_in_executor 사용
            response = await loop.run_in_executor(
                None, # 기본 실행기 사용
                lambda: self.client.search_recent_tweets(
                    query=query,
                    max_results=api_max_results,
                    tweet_fields=["created_at", "public_metrics", "conversation_id", "lang"] # 필요한 필드 지정
                )
            )

            results = []
            if response.data: # 응답 데이터가 있는 경우
                for tweet in response.data:
                    tweet_url = f"https://twitter.com/anyuser/status/{tweet.id}" # 트윗 URL 생성
                    # 결과 목록에 필요한 정보 추가
                    results.append({
                        "url": tweet_url,
                        "text": tweet.text,
                        "tweet_id": str(tweet.id),
                        "source": "Twitter", # 출처 명시
                        "created_at": tweet.created_at.isoformat() if tweet.created_at else None, # 생성 시간 (ISO 형식)
                        "lang": tweet.lang, # 언어 정보
                    })
                logger.info(f"{len(results)}개의 트윗을 찾았습니다.", extra=extra_log_data)
            else: # 응답 데이터가 없는 경우
                logger.info("쿼리에 해당하는 최근 트윗을 찾을 수 없습니다.", extra=extra_log_data)
            return results

        except tweepy.errors.TweepyException as e:
             # 특정 tweepy 오류 처리 (예: Rate Limit, 인증 오류 등)
             logger.error(f"검색 중 Tweepy API 오류 발생: {e}", extra=extra_log_data)
             return [] # 현재 시도에서는 빈 리스트 반환
        except Exception as e:
            # 예상치 못한 오류 발생 시 로깅 및 예외 다시 발생 (재시도 유도)
            logger.error(f"Twitter 검색 중 예상치 못한 오류 발생: {e}", exc_info=True, extra=extra_log_data)
            raise # tenacity 재시도를 위해 예외 발생

    @retry(stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS),  # 재시도 설정
           wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN, max=settings.TOOL_RETRY_WAIT_MAX),
           retry=retry_if_exception_type(Exception), reraise=True)
    async def get_tweet_details(self, tweet_id: str, trace_id: str) -> Optional[Dict[str, any]]:
        """주어진 ID의 트윗 상세 정보를 가져옵니다."""
        if not self.client: return None
        extra_log_data = {'trace_id': trace_id, 'tweet_id': tweet_id}
        logger.debug("API를 통해 트윗 상세 정보 가져오는 중...", extra=extra_log_data)
        try:
            loop = asyncio.get_running_loop()
            # 필요한 필드 확장 요청 (작성자 정보 포함 등)
            response = await loop.run_in_executor(None, lambda: self.client.get_tweet(
                tweet_id,
                tweet_fields=["created_at", "public_metrics", "author_id", "conversation_id", "lang"],
                expansions=["author_id"],  # 작성자 정보 포함
                user_fields=["username", "name"]  # 작성자의 username, name 필드 요청
            ))
            if response.data:
                tweet = response.data
                # 포함된 사용자 정보에서 작성자 찾기
                author = next((user for user in response.includes.get('users', []) if user.id == tweet.author_id), None)
                author_name = author.username if author else str(tweet.author_id)  # 사용자명 없으면 ID 사용

                return {
                    "text": tweet.text,
                    "author": author_name,
                    "timestamp": tweet.created_at.isoformat() if tweet.created_at else None,
                    "likes": tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                    "raw_data": tweet.data  # 원본 트윗 객체 저장
                }
            else:
                logger.warning("API를 통해 트윗을 찾을 수 없음", extra=extra_log_data)
                return None
        except tweepy.errors.NotFound:  # 404 에러 처리
            logger.warning("API를 통해 트윗을 찾을 수 없음 (404)", extra=extra_log_data)
            return None
        except Exception as e:
            logger.error(f"API 통해 트윗 가져오기 오류: {e}", exc_info=True, extra=extra_log_data)
            raise  # 재시도 위해 예외 발생
    # tweepy는 보통 명시적인 close 메서드가 필요 없음
    # async def close(self):
    #     pass