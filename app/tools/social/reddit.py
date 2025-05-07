# app/tools/social/reddit.py
import asyncio
from typing import List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config.settings import settings # 설정 임포트
from app.utils.logger import get_logger # 로거 임포트
from datetime import datetime, timezone

# praw 동적 임포트 시도
try:
    import praw
    import prawcore # praw 예외 처리를 위해 임포트
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    praw = None
    prawcore = None # 사용 불가 시 None 할당

logger = get_logger(__name__) # 로거 초기화

class RedditTool:
    """PRAW를 사용하여 레딧 API와 상호작용하는 도구입니다."""

    def __init__(self):
        """RedditTool 초기화"""
        self.reddit: Optional[praw.Reddit] = None # Reddit 클라이언트 변수 초기화
        # 설정에서 Reddit 관련 정보 로드
        self.client_id = settings.REDDIT_CLIENT_ID
        self.client_secret = settings.REDDIT_CLIENT_SECRET
        self.user_agent = settings.REDDIT_USER_AGENT
        self.username = settings.REDDIT_USERNAME
        self.password = settings.REDDIT_PASSWORD
        self.target_subreddits = settings.REDDIT_TARGET_SUBREDDITS # 검색 대상 서브레딧 목록 로드

        # 라이브러리 또는 필수 설정 부재 시 경고 로깅 및 초기화 중단
        if not PRAW_AVAILABLE:
            logger.warning("praw 라이브러리가 설치되지 않았습니다. RedditTool이 작동하지 않습니다.")
            return
        if not all([self.client_id, self.client_secret, self.user_agent]):
            logger.warning("Reddit client ID, secret 또는 user agent가 설정되지 않았습니다. RedditTool이 작동하지 않습니다.")
            return
        if not self.target_subreddits:
             logger.warning("settings에 REDDIT_TARGET_SUBREDDITS가 설정되지 않았습니다. Reddit 검색이 비효율적일 수 있습니다.")

        try:
            logger.info("Reddit 클라이언트 초기화 중...")
            auth_args = {} # 인증 정보 딕셔너리
            # 사용자 이름/비밀번호가 있으면 script 앱 타입 인증 시도
            if self.username and self.password:
                 auth_args["username"] = self.username
                 auth_args["password"] = self.password
                 logger.info("Username/Password를 사용하여 Reddit 인증 시도 (script 앱 타입).")
            else:
                 logger.warning("Reddit username/password가 설정되지 않았습니다. 읽기 전용 모드로 초기화 시도 (OAuth 앱 타입).")

            # PRAW Reddit 클라이언트 생성
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                **auth_args # 인증 정보 전달
                # check_for_async=False # asyncpraw 사용 안 할 경우 praw >= 7.1 에서 필요할 수 있음
            )
            # 읽기 전용 상태 확인 및 로깅
            logger.info(f"Reddit 클라이언트 초기화 완료. 읽기 전용: {self.reddit.read_only}")
            # 필요 시 간단한 읽기 작업으로 인증 테스트 가능 (예: self.reddit.user.me())
        except prawcore.exceptions.OAuthException as e:
             # OAuth 인증 실패 시 오류 로깅
             logger.error(f"Reddit OAuth 인증 실패: {e}. 인증 정보 또는 앱 타입을 확인하세요.", exc_info=True)
             self.reddit = None # 실패 시 클라이언트 None 설정
        except Exception as e:
            # 기타 초기화 오류 처리
            logger.error(f"Reddit 클라이언트 초기화 실패: {e}", exc_info=True)
            self.reddit = None # 실패 시 클라이언트 None 설정

    @retry(
        stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS), # 설정된 재시도 횟수 사용
        wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN, max=settings.TOOL_RETRY_WAIT_MAX), # 지수적 대기 시간 적용
        retry=retry_if_exception_type(Exception), # 모든 예외 발생 시 재시도 (PRAWException 포함)
        reraise=True # 모든 재시도 실패 시 예외 다시 발생
    )
    async def search_posts(self, keyword: str, max_results: int, trace_id: str) -> List[Dict[str, Any]]:
        """
        설정된 서브레딧들에서 키워드로 게시글을 검색합니다.

        Args:
            keyword (str): 검색어.
            max_results (int): 모든 서브레딧에서 반환할 목표 최대 게시글 수.
            trace_id (str): 로깅을 위한 추적 ID.

        Returns:
            List[Dict[str, Any]]: 게시글 정보(url, title, text, source 등)를 담은 사전 목록.
        """
        # 클라이언트 또는 대상 서브레딧 부재 시 빈 리스트 반환
        if not self.reddit:
            logger.warning("Reddit 클라이언트를 사용할 수 없어 검색을 건너<0xEB><0x9C><0x95>니다.", extra={'trace_id': trace_id})
            return []
        if not self.target_subreddits:
             logger.warning("검색 대상 서브레딧이 설정되지 않아 검색을 건너<0xEB><0x9C><0x95>니다.", extra={'trace_id': trace_id})
             return []

        extra_log_data = {'trace_id': trace_id, 'keyword': keyword} # 로깅용 추가 데이터
        logger.info(f"'{keyword}'에 대한 Reddit 게시글 검색 중 ({len(self.target_subreddits)}개 서브레딧, 목표: {max_results})", extra=extra_log_data)

        # 서브레딧 당 검색 제한 계산 (목표치보다 약간 더 많이 가져옴)
        limit_per_subreddit = max(1, max_results // len(self.target_subreddits) + 2)
        loop = asyncio.get_running_loop() # 현재 이벤트 루프
        tasks = [] # 비동기 작업 목록

        def search_subreddit_sync(sub_name: str, query: str, limit: int):
            """동기 함수: 특정 서브레딧 검색"""
            try:
                subreddit = self.reddit.subreddit(sub_name) # 서브레딧 객체 가져오기
                # 관련성(relevance) 순으로 검색, 필요 시 'new' 등 다른 정렬 사용 가능
                return list(subreddit.search(query, limit=limit, sort='relevance'))
            except prawcore.exceptions.Redirect: # 서브레딧 찾을 수 없거나 리다이렉트될 경우
                logger.warning(f"서브레딧 r/{sub_name}을(를) 찾을 수 없거나 리다이렉트됩니다.", extra=extra_log_data)
                return []
            except prawcore.exceptions.Forbidden: # 접근 권한이 없는 경우
                 logger.warning(f"서브레딧 r/{sub_name} 접근 권한이 없습니다.", extra=extra_log_data)
                 return []
            except Exception as search_err:
                # 해당 서브레딧 검색 중 발생한 기타 오류 로깅
                logger.error(f"서브레딧 r/{sub_name} 검색 오류: {search_err}", exc_info=True, extra=extra_log_data)
                return [] # 오류 발생 시 해당 서브레딧 결과는 빈 리스트

        # 각 서브레딧 검색 작업을 비동기 실행 목록에 추가
        for sub_name in self.target_subreddits:
            tasks.append(loop.run_in_executor(None, search_subreddit_sync, sub_name, keyword, limit_per_subreddit))

        # 모든 서브레딧 검색 작업 병렬 실행 및 결과 취합
        all_submissions = []
        submission_lists = await asyncio.gather(*tasks, return_exceptions=True) # 예외 발생 시에도 계속 진행

        # 결과 처리
        for sub_list_result in submission_lists:
            if isinstance(sub_list_result, list): # 성공적으로 리스트 반환 시
                all_submissions.extend(sub_list_result)
            elif isinstance(sub_list_result, Exception): # 예외 객체 반환 시
                logger.error(f"Reddit 검색 작업 중 예외 발생: {sub_list_result}", exc_info=sub_list_result, extra=extra_log_data)

        # 결과 포맷팅 및 URL 기준 중복 제거
        results = []
        processed_urls = set()
        # 필요 시 관련성 점수(score) 등으로 정렬 가능
        # all_submissions.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)

        for submission in all_submissions:
             # submission 객체 유효성 검사 (필수 속성 확인)
             if not hasattr(submission, 'permalink') or not hasattr(submission, 'subreddit'):
                  logger.warning(f"처리할 수 없는 submission 객체 발견: {submission}", extra=extra_log_data)
                  continue

             permalink = f"https://www.reddit.com{submission.permalink}" # 고유 링크 생성
             if permalink not in processed_urls: # 중복 URL 체크
                # 결과 딕셔너리 생성
                results.append({
                    "url": permalink,
                    "title": getattr(submission, 'title', ''), # 제목 (없으면 빈 문자열)
                    "text": getattr(submission, 'selftext', ''), # 본문 (텍스트 게시글용)
                    "score": getattr(submission, 'score', 0), # 점수
                    "num_comments": getattr(submission, 'num_comments', 0), # 댓글 수
                    "created_utc": getattr(submission, 'created_utc', None), # 생성 시간 (UTC 타임스탬프)
                    "subreddit": getattr(submission.subreddit, 'display_name', 'Unknown'), # 서브레딧 이름
                    "source": f"Reddit (r/{getattr(submission.subreddit, 'display_name', 'Unknown')})", # 출처 명시
                    "post_id": getattr(submission, 'id', None), # 게시글 ID
                })
                processed_urls.add(permalink) # 처리된 URL 집합에 추가
             # 목표 결과 수 도달 시 중단
             if len(results) >= max_results:
                  break

        logger.info(f"검색 후 {len(results)}개의 고유한 Reddit 게시글을 찾았습니다.", extra=extra_log_data)
        return results # 최종 결과 반환 (최대 max_results 개수)

    @retry(stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS),  # 재시도 설정
           wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN, max=settings.TOOL_RETRY_WAIT_MAX),
           retry=retry_if_exception_type(Exception), reraise=True)
    async def get_submission_details(self, submission_id: str, trace_id: str) -> Optional[Dict[str, Any]]:
        """주어진 ID의 Reddit 게시글 상세 정보를 가져옵니다."""
        if not self.reddit: return None
        extra_log_data = {'trace_id': trace_id, 'submission_id': submission_id}
        logger.debug("API를 통해 Reddit 게시글 상세 정보 가져오는 중...", extra=extra_log_data)
        try:
            loop = asyncio.get_running_loop()
            submission = await loop.run_in_executor(None, self.reddit.submission, submission_id)
            if submission:
                author_obj = getattr(submission, 'author', None)
                return {
                    "text": getattr(submission, 'selftext', ''),  # 본문
                    "title": getattr(submission, 'title', ''),  # 제목 추가
                    "author": str(author_obj) if author_obj else '[삭제됨]',
                    "timestamp": datetime.fromtimestamp(submission.created_utc, timezone.utc).isoformat() if hasattr(
                        submission, 'created_utc') else None,
                    "likes": getattr(submission, 'score', 0),
                    "num_comments": getattr(submission, 'num_comments', 0),  # 댓글 수 추가
                    "raw_data": {"id": submission.id, "type": "submission"}  # 기본 정보 저장
                }
            else:
                logger.warning("API를 통해 Reddit 게시글을 찾을 수 없음", extra=extra_log_data)
                return None
        except Exception as e:
            logger.error(f"API 통해 Reddit 게시글 가져오기 오류: {e}", exc_info=True, extra=extra_log_data)
            raise  # 재시도 위해 예외 발생

    @retry(stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS),  # 재시도 설정
           wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN, max=settings.TOOL_RETRY_WAIT_MAX),
           retry=retry_if_exception_type(Exception), reraise=True)
    async def get_comment_details(self, comment_id: str, trace_id: str) -> Optional[Dict[str, Any]]:
        """주어진 ID의 Reddit 댓글 상세 정보를 가져옵니다."""
        if not self.reddit: return None
        extra_log_data = {'trace_id': trace_id, 'comment_id': comment_id}
        logger.debug("API를 통해 Reddit 댓글 상세 정보 가져오는 중...", extra=extra_log_data)
        try:
            loop = asyncio.get_running_loop()
            comment = await loop.run_in_executor(None, self.reddit.comment, comment_id)
            if comment:
                author_obj = getattr(comment, 'author', None)
                return {
                    "text": getattr(comment, 'body', ''),  # 댓글 내용
                    "author": str(author_obj) if author_obj else '[삭제됨]',
                    "timestamp": datetime.fromtimestamp(comment.created_utc, timezone.utc).isoformat() if hasattr(
                        comment, 'created_utc') else None,
                    "likes": getattr(comment, 'score', 0),
                    "raw_data": {"id": comment.id, "type": "comment"}  # 기본 정보 저장
                }
            else:
                logger.warning("API를 통해 Reddit 댓글을 찾을 수 없음", extra=extra_log_data)
                return None
        except Exception as e:
            logger.error(f"API 통해 Reddit 댓글 가져오기 오류: {e}", exc_info=True, extra=extra_log_data)
            raise  # 재시도 위해 예외 발생

    # PRAW는 보통 명시적인 close 메서드가 필요 없음
    # async def close(self):
    #     pass