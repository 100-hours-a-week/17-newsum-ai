# app/tools/search/rss.py
import asyncio
from typing import List, Dict, Optional
from urllib.parse import urlparse
from app.config.settings import settings # 설정 임포트
from app.utils.logger import get_logger # 로거 임포트

logger = get_logger("RssSearchTool") # 로거 초기화 (클래스 이름과 일치시키거나 파일 이름 사용)

# feedparser 동적 임포트 시도
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.warning("feedparser 라이브러리가 설치되지 않았습니다. RssSearchTool이 작동하지 않습니다.")

class RssSearchTool:
    """미리 정의된 RSS 피드 목록에서 키워드 관련 항목을 검색하는 도구입니다."""

    def __init__(self):
        """RssSearchTool 초기화"""
        # 라이브러리 사용 가능 시 설정에서 피드 목록 로드
        self.feed_list = settings.PREDEFINED_RSS_FEEDS if FEEDPARSER_AVAILABLE else []
        # 피드 목록 없으면 경고 로깅
        if FEEDPARSER_AVAILABLE and not self.feed_list:
             logger.warning("RSS 도구가 초기화되었지만 settings.PREDEFINED_RSS_FEEDS에 피드가 없습니다.")

    async def search(self, keyword: str, limit_per_feed: int, trace_id: Optional[str]=None) -> List[Dict[str, str]]:
        """
        모든 RSS 피드에서 키워드 관련 항목을 비동기적으로 검색합니다.
        Node 03 (News Collector)에서 주로 사용될 것으로 예상됩니다.

        Args:
            keyword (str): 검색할 키워드 (소문자로 변환하여 사용).
            limit_per_feed (int): 각 피드에서 가져올 최대 항목 수.
            trace_id (Optional[str]): 로깅을 위한 추적 ID.

        Returns:
            List[Dict[str, str]]: 검색된 항목 정보(url, source, search_keyword)의 목록 (URL 기준 중복 제거됨).
        """
        # 라이브러리 또는 피드 목록 없으면 빈 리스트 반환
        if not FEEDPARSER_AVAILABLE or not self.feed_list: return []
        extra_log_data = {'trace_id': trace_id, 'keyword': keyword} # 로깅용 추가 데이터
        keyword_lower = keyword.lower() # 검색 시 대소문자 구분 안 함
        logger.debug(f"{len(self.feed_list)}개 피드에서 '{keyword_lower}' 검색 중", extra=extra_log_data)

        # 각 피드 파싱 작업을 비동기 태스크 목록으로 생성
        # comic_id는 이 도구 레벨에서는 알 수 없으므로 None 전달 또는 제거
        tasks = [self._parse_single_feed(feed_url, keyword_lower, limit_per_feed, trace_id)
                 for feed_url in self.feed_list]
        # 모든 태스크 병렬 실행 및 결과 취합
        feed_results_list = await asyncio.gather(*tasks, return_exceptions=True)

        all_items = [] # 최종 결과 목록
        processed_urls = set() # URL 중복 제거용 집합
        # 결과 처리
        for result in feed_results_list:
            if isinstance(result, list): # 성공 시
                 for item in result:
                      url = item.get('url')
                      if url and url not in processed_urls: # 중복 체크
                           all_items.append(item)
                           processed_urls.add(url)
            elif isinstance(result, Exception): # 예외 발생 시
                logger.error(f"RSS 피드 처리 중 오류 발생: {result}", extra=extra_log_data)

        logger.debug(f"모든 피드에서 '{keyword_lower}'에 대해 {len(all_items)}개의 고유 항목을 찾았습니다.", extra=extra_log_data)
        return all_items # 최종 결과 반환

    async def _parse_single_feed(self, feed_url: str, keyword_lower: str, limit: int, trace_id: Optional[str]) -> List[Dict[str, str]]:
        """단일 RSS 피드를 비동기적으로 파싱합니다."""
        if not FEEDPARSER_AVAILABLE: return [] # 라이브러리 없으면 실행 불가
        extra_log_data = {'trace_id': trace_id, 'feed_url': feed_url} # 로깅용 데이터

        loop = asyncio.get_running_loop() # 현재 이벤트 루프
        feed_items = [] # 파싱된 항목 저장 리스트
        try:
            logger.debug(f"RSS 피드 파싱 중: {feed_url}", extra=extra_log_data)
            # feedparser.parse는 동기 함수이므로 run_in_executor 사용, 타임아웃 설정
            async with asyncio.timeout(15):
                feed_data = await loop.run_in_executor(None, feedparser.parse, feed_url)

            # --- 파싱 후 피드 상태 검사 ---
            status = feed_data.get('status') # HTTP 상태 코드 확인
            if status and (status < 200 or status >= 400):
                logger.warning(f"{feed_url} RSS 가져오기 실패 (상태 코드: {status})", extra=extra_log_data)
                return [] # 오류 시 빈 리스트 반환
            # bozo: 파싱 중 문제 발생 여부 확인 (1이면 문제 있음)
            if feed_data.get('bozo'):
                 exc = feed_data.get('bozo_exception') # 문제 원인 예외 객체
                 # XML 아닌 콘텐츠 타입은 오류 아님 (조용히 무시)
                 if isinstance(exc, feedparser.NonXMLContentType):
                      logger.debug(f"피드가 XML 형식이 아님: {feed_url}", extra=extra_log_data)
                      return []
                 # 다른 bozo 문제는 경고 로깅
                 logger.warning(f"{feed_url} RSS 파싱 문제 (bozo): {exc}", extra=extra_log_data)

            # --- 피드 항목 처리 ---
            count = 0 # 찾은 항목 수 카운터
            feed_domain = urlparse(feed_url).netloc # 피드 출처 도메인 추출
            for entry in feed_data.entries: # 각 항목(entry) 순회
                title = entry.get('title', '').lower() # 제목 (소문자 변환)
                summary = entry.get('summary', '').lower() # 요약 (소문자 변환)
                # 제목 또는 요약에 키워드가 포함되어 있는지 확인
                if keyword_lower in title or keyword_lower in summary:
                    link_url = entry.get('link', '') # 항목 링크 추출
                    if link_url: # 링크가 유효한 경우
                        feed_items.append({
                            "url": link_url,
                            "source": f"RSS ({feed_domain})", # 출처 명시 (피드 도메인 포함)
                            "search_keyword": keyword_lower, # 검색 키워드 저장 (소문자)
                            "title": entry.get('title', ''), # 원본 제목 추가
                            "published": entry.get('published', '') # 발행일 정보 추가
                        })
                        count += 1
                        if count >= limit: break # 피드당 제한 도달 시 중단
            return feed_items # 찾은 항목 리스트 반환

        except asyncio.TimeoutError: # 타임아웃 발생 시
            logger.warning(f"{feed_url} RSS 피드 처리 시간 초과", extra=extra_log_data)
            return [] # 빈 리스트 반환
        except Exception as e: # 기타 예상치 못한 오류
            logger.exception(f"{feed_url} RSS 피드 파싱 중 예상치 못한 오류 발생: {e}", extra=extra_log_data)
            return [] # 빈 리스트 반환

    # async def close(self): # feedparser는 보통 명시적 close 필요 없음
    #     pass