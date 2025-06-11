# app/tools/scraping/article_scraper.py

import asyncio
import json
from typing import Dict, Any, Optional
from urllib.parse import urlparse, urljoin
import aiohttp
from bs4 import BeautifulSoup
import traceback # 상세 오류 로깅용

# 필요한 라이브러리 동적 임포트 및 사용 가능 여부 확인
try:
    import readability
    READABILITY_AVAILABLE = True
except ImportError:
    readability = None
    READABILITY_AVAILABLE = False

try:
    from newspaper import Article, ArticleException
    NEWSPAPER_AVAILABLE = True
except ImportError:
    Article = None
    ArticleException = None
    NEWSPAPER_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None
    TRAFILATURA_AVAILABLE = False

try:
    from langdetect import detect as langdetect_detect, LangDetectException, DetectorFactory
    # 언어 감지 재현성을 위한 시드 설정 (선택 사항)
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LangDetectException = None
    LANGDETECT_AVAILABLE = False
    # langdetect 없을 시 더미 함수 정의
    def langdetect_detect(text: str) -> str: return 'und'

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger("ArticleScraperTool")

class ArticleScraperTool:
    """
    단일 URL로부터 기사 콘텐츠(텍스트, 제목, 날짜 등)를 추출하는 도구입니다.
    HTML 가져오기, AMP 처리, 다양한 라이브러리(Readability, Newspaper3k, Trafilatura)를
    이용한 콘텐츠 추출, 언어 감지 기능을 포함합니다.
    """

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """
        ArticleScraperTool 초기화.

        Args:
            session (Optional[aiohttp.ClientSession]): 외부 aiohttp 세션 (선택 사항). 없으면 내부 생성.
        """

        self.user_agent = getattr(settings, 'SCRAPER_USER_AGENT', None) or \
                          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        self.http_timeout = getattr(settings, 'SCRAPER_HTTP_TIMEOUT', 15)
        self.min_text_length = getattr(settings, 'MIN_EXTRACTED_TEXT_LENGTH', 150)
        self.min_lang_detect_length = getattr(settings, 'MIN_LANGDETECT_TEXT_LENGTH', 50)


        self._session = session
        self._created_session = False
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.http_timeout))
            self._created_session = True
            logger.info("ArticleScraperTool을 위한 새로운 aiohttp ClientSession 생성됨.")

        # 라이브러리 가용성 로깅
        if not READABILITY_AVAILABLE: logger.warning("readability-lxml 라이브러리를 찾을 수 없습니다. Readability 추출 비활성화됨.")
        if not NEWSPAPER_AVAILABLE: logger.warning("newspaper3k 라이브러리를 찾을 수 없습니다. Newspaper 추출 비활성화됨.")
        if not TRAFILATURA_AVAILABLE: logger.warning("trafilatura 라이브러리를 찾을 수 없습니다. Trafilatura 추출 비활성화됨.")
        if not LANGDETECT_AVAILABLE: logger.warning("langdetect 라이브러리를 찾을 수 없습니다. 언어 감지가 비활성화됩니다 ('und').")

    @retry(
        stop=stop_after_attempt(getattr(settings, 'TOOL_RETRY_ATTEMPTS', 3)), # 설정값 사용
        wait=wait_exponential(multiplier=1, min=getattr(settings, 'TOOL_RETRY_WAIT_MIN', 2), max=getattr(settings, 'TOOL_RETRY_WAIT_MAX', 10)), # 설정값 사용
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)), # 재시도 조건
        reraise=True # 실패 시 예외 다시 발생
    )
    async def _fetch_html(self, url: str, trace_id: str, comic_id: str) -> Optional[str]:
        """주어진 URL의 HTML 콘텐츠를 비동기적으로 가져옵니다."""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url}
        logger.debug("HTML 가져오기 시도 중...", extra=extra_log_data)

        if not self._session or self._session.closed:
            logger.error("aiohttp 세션이 닫혔거나 초기화되지 않았습니다.", extra=extra_log_data)
            return None

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5,ko-KR;q=0.3", # 영어/한국어 우선
            "Connection": "keep-alive",
            "DNT": "1", # Do Not Track
            "Upgrade-Insecure-Requests": "1"
        }
        try:
            # GET 요청 실행, 리다이렉트 허용
            async with self._session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.http_timeout), allow_redirects=True) as response:
                response_text = await response.text(encoding='utf-8', errors='ignore') # 오류 무시하고 텍스트 읽기
                # 성공 상태 코드(2xx) 확인
                if 200 <= response.status < 300:
                    logger.debug(f"HTML 가져오기 성공 (길이: {len(response_text)})", extra=extra_log_data)
                    return response_text
                else:
                    logger.error(f"HTML 가져오기 실패. 상태 코드: {response.status}", extra=extra_log_data)
                    # 오류 상태 코드에 대한 예외 발생
                    response.raise_for_status()
                    return None # 예외 발생 시 도달하지 않음
        except asyncio.TimeoutError as e:
             logger.error(f"HTML 가져오기 시간 초과 ({self.http_timeout}초)", extra=extra_log_data)
             raise e # 재시도 위해 예외 발생
        except aiohttp.ClientError as e:
             logger.error(f"HTML 가져오기 클라이언트 오류: {e}", extra=extra_log_data)
             raise e # 재시도 위해 예외 발생
        except Exception as e:
            logger.exception(f"HTML 가져오기 중 예상치 못한 오류 발생: {e}", extra=extra_log_data)
            return None # 예상치 못한 오류 시 None 반환

    def _find_amp_url(self, html: str, base_url: str, trace_id: str, comic_id: str) -> Optional[str]:
        """HTML을 파싱하여 AMP 버전 URL을 찾습니다."""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'base_url': base_url}
        if not html: return None
        try:
            soup = BeautifulSoup(html, 'html.parser') # HTML 파서 사용
            # 'link' 태그 중 rel='amphtml' 속성을 가진 것 검색
            amp_link_tag = soup.find('link', rel='amphtml', href=True)

            if amp_link_tag:
                amp_url = amp_link_tag['href'] # href 속성값 추출
                # 상대 경로일 경우 절대 경로로 변환
                absolute_amp_url = urljoin(base_url, amp_url)
                logger.debug(f"AMP 링크 발견: {absolute_amp_url}", extra=extra_log_data)
                return absolute_amp_url
            return None # AMP 링크 없으면 None 반환
        except Exception as e:
            # 파싱 오류 발생 시 경고 로깅
            logger.warning(f"AMP 링크 파싱 중 오류 발생: {e}", extra=extra_log_data)
            return None

    def _run_sync_in_executor(self, func, *args):
        """CPU-bound 동기 함수를 비동기 executor에서 실행합니다."""
        # 이 메서드는 클래스 내부 또는 별도 유틸리티로 분리할 수 있습니다.
        # 현재 구현에서는 직접 loop.run_in_executor를 사용합니다.
        # 필요하다면 이 헬퍼를 완성하여 사용할 수 있습니다.
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, func, *args)

    def _extract_content_readability(self, html: str, url: str, trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """python-readability 라이브러리를 사용하여 콘텐츠를 추출합니다."""
        if not READABILITY_AVAILABLE or not html: return None
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, 'method': 'readability'}
        try:
            # Readability 문서는 CPU bound 작업일 수 있음
            doc = readability.Document(html)
            title = doc.title()
            # 요약 HTML에서 텍스트 추출
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, 'html.parser')
            text = soup.get_text(separator='\n', strip=True) # 줄바꿈 유지하며 텍스트 추출

            # 최소 길이 확인
            if text and len(text) >= self.min_text_length:
                 logger.debug("Readability 추출 성공", extra=extra_log_data)
                 # 발행일 검색 시도
                 publish_date = self._find_publish_date(html, trace_id, comic_id)
                 return {'title': title, 'text': text, 'publish_date': publish_date, 'method': 'readability'}
            else:
                 logger.debug(f"Readability 추출 결과가 너무 짧음 (길이: {len(text)})", extra=extra_log_data)
                 return None
        except Exception as e:
            logger.error(f"Readability 추출 실패: {e}", exc_info=True, extra=extra_log_data)
            return None

    async def _extract_content_newspaper(self, html: str, url: str, trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """newspaper3k 라이브러리를 사용하여 콘텐츠를 추출합니다."""
        if not NEWSPAPER_AVAILABLE or not html: return None
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, 'method': 'newspaper3k'}
        try:
            article = Article(url)
            article.set_html(html) # 미리 가져온 HTML 설정

            # newspaper3k의 parse()는 잠재적으로 블로킹 I/O를 수행할 수 있음
            # run_in_executor 사용하여 이벤트 루프 블로킹 방지
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, article.parse)

            text = article.text
            title = article.title
            publish_date_dt = article.publish_date

            # 최소 길이 확인
            if text and len(text) >= self.min_text_length:
                 logger.debug("Newspaper3k 추출 성공", extra=extra_log_data)
                 publish_date = publish_date_dt.isoformat() if publish_date_dt else None
                 # newspaper가 날짜 못 찾으면 메타 태그에서 재시도
                 if not publish_date:
                     publish_date = self._find_publish_date(html, trace_id, comic_id)
                 return {'title': title, 'text': text, 'publish_date': publish_date, 'method': 'newspaper3k'}
            else:
                 logger.debug(f"Newspaper3k 추출 결과가 너무 짧음 (길이: {len(text)})", extra=extra_log_data)
                 return None
        except ArticleException as e: # Newspaper3k 특정 예외 처리
             logger.error(f"Newspaper3k ArticleException 발생: {e}", extra=extra_log_data)
             return None
        except Exception as e: # 기타 예외 처리
            logger.error(f"Newspaper3k 추출 중 예상치 못한 오류 발생: {e}", exc_info=True, extra=extra_log_data)
            return None

    async def _extract_content_trafilatura(self, html: str, url: str, trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """trafilatura 라이브러리를 사용하여 콘텐츠를 추출합니다."""
        if not TRAFILATURA_AVAILABLE or not html: return None
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, 'method': 'trafilatura'}
        try:
            # trafilatura는 CPU-bound 작업일 수 있음
            loop = asyncio.get_running_loop()
            # JSON 형태로 결과 추출 (주석 제외)
            extracted_json = await loop.run_in_executor(
               None, trafilatura.extract, html, output_format='json', include_comments=False, url=url
            )

            if extracted_json:
                data = json.loads(extracted_json) # JSON 파싱
                text = data.get('text')
                # 최소 길이 확인
                if text and len(text) >= self.min_text_length:
                     logger.debug("Trafilatura 추출 성공", extra=extra_log_data)
                     # trafilatura가 날짜 못 찾으면 메타 태그에서 재시도
                     publish_date = data.get('date') or self._find_publish_date(html, trace_id, comic_id)
                     return {'title': data.get('title', ''), 'text': text, 'publish_date': publish_date, 'method': 'trafilatura'}
                else:
                     logger.debug(f"Trafilatura 추출 결과가 너무 짧음 (길이: {len(text)})", extra=extra_log_data)
                     return None
            return None # 추출 결과 없으면 None 반환
        except Exception as e:
            logger.error(f"Trafilatura 추출 실패: {e}", exc_info=True, extra=extra_log_data)
            return None

    def _find_publish_date(self, html: str, trace_id: str, comic_id: str) -> Optional[str]:
        """일반적인 메타 태그에서 발행일을 찾으려고 시도합니다."""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        if not html: return None
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # 발행일 관련 흔한 메타 태그 및 time 태그 셀렉터
            selectors = [
                "meta[property='article:published_time']",
                "meta[name='pubdate']",
                "meta[name='date']",
                "meta[itemprop='datePublished']",
                "meta[name='parsely-pub-date']",
                "meta[name='sailthru.date']",
                "meta[name='article.published']", # 추가
                "meta[name='cXenseParse:recs:publishtime']", # 추가
                "time[itemprop='datePublished']", # time 태그 추가
                "time[datetime]" # 일반적인 time 태그
            ]
            for selector in selectors:
                 tag = soup.select_one(selector) # CSS 셀렉터로 태그 검색
                 if tag:
                      # content 또는 datetime 속성에서 날짜 문자열 가져오기
                      date_str = tag.get('content') or tag.get('datetime')
                      if date_str:
                           # TODO: dateutil.parser 등을 사용하여 더 정확한 날짜 파싱 구현 가능
                           # 현재는 찾은 문자열 그대로 반환
                           logger.debug(f"메타 태그에서 발행일 정보 발견: '{date_str}' (셀렉터: {selector})", extra=extra_log_data)
                           return date_str.strip()
            # 루프 후에도 못 찾으면 None 반환
            return None
        except Exception as e:
            # 날짜 파싱 오류는 무시하고 None 반환
            logger.debug(f"발행일 검색 중 오류 발생 (무시됨): {e}", extra=extra_log_data)
            return None

    def _detect_language(self, text: str, trace_id: str, comic_id: str) -> str:
        """langdetect를 사용하여 텍스트의 언어를 감지합니다."""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        # 라이브러리 없거나 텍스트가 너무 짧으면 'und' (미정) 반환
        if not LANGDETECT_AVAILABLE: return 'und'
        if not text or len(text.strip()) < self.min_lang_detect_length:
            logger.debug(f"언어 감지를 위한 텍스트가 너무 짧음 (길이: {len(text.strip()) if text else 0})", extra=extra_log_data)
            return 'und'

        try:
            # 언어 감지 실행
            lang_code = langdetect_detect(text[:2000]) # 너무 긴 텍스트는 일부만 사용
            logger.debug(f"감지된 언어: {lang_code}", extra=extra_log_data)
            return lang_code
        except LangDetectException:
             # 언어 특징을 찾지 못한 경우
             logger.warning("언어 감지 실패 (특징 없음?). 'und' 반환.", extra=extra_log_data)
             return 'und'
        except Exception as e:
            # 기타 예상치 못한 오류
            logger.error(f"언어 감지 중 예상치 못한 오류 발생: {e}", exc_info=True, extra=extra_log_data)
            return 'und'

    async def scrape_article(self, url: str, trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """
        단일 URL에 대해 스크래핑 및 콘텐츠 추출 파이프라인을 실행합니다.

        Args:
            url (str): 스크랩할 원본 URL.
            trace_id (str): 로깅용 추적 ID.
            comic_id (str): 로깅용 코믹 ID.

        Returns:
            Optional[Dict[str, Any]]: 추출된 기사 데이터 (title, text, publish_date, language 등)
                                       또는 실패 시 None.
        """
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'original_url': url}
        logger.info(f"기사 스크래핑 시작: {url}", extra=extra_log_data)
        final_url_processed = url # 최종적으로 처리된 URL (AMP일 수 있음)

        try:
            # 1. 원본 HTML 가져오기
            html = await self._fetch_html(url, trace_id, comic_id)
            if not html:
                logger.warning("원본 HTML 가져오기 실패", extra=extra_log_data)
                return None # fetch 실패 시 종료

            # 2. AMP 버전 확인 및 가져오기
            amp_url = self._find_amp_url(html, url, trace_id, comic_id)
            if amp_url:
                amp_html = await self._fetch_html(amp_url, trace_id, comic_id)
                if amp_html:
                     logger.info(f"AMP 콘텐츠 사용: {amp_url}", extra=extra_log_data)
                     html = amp_html # 추출 시 AMP HTML 사용
                     final_url_processed = amp_url # 처리된 URL 업데이트
                else:
                     logger.warning(f"AMP 콘텐츠 가져오기 실패, 원본 사용: {amp_url}", extra=extra_log_data)

            # 3. 콘텐츠 추출 (Fallback 전략 사용)
            # CPU-bound 작업들이므로, 병렬화가 필요하다면 각 함수 내부에서 executor 사용 고려
            # 여기서는 순차적으로 실행
            extracted_content = None
            if READABILITY_AVAILABLE:
                extracted_content = self._extract_content_readability(html, final_url_processed, trace_id, comic_id)
            if not extracted_content and NEWSPAPER_AVAILABLE:
                extracted_content = await self._extract_content_newspaper(html, final_url_processed, trace_id, comic_id)
            if not extracted_content and TRAFILATURA_AVAILABLE:
                extracted_content = await self._extract_content_trafilatura(html, final_url_processed, trace_id, comic_id)

            # 추출 성공 여부 확인
            if not extracted_content:
                 logger.warning(f"모든 방법으로 유의미한 콘텐츠 추출 실패: {final_url_processed}", extra=extra_log_data)
                 return None

            # 4. 언어 감지
            language = self._detect_language(extracted_content['text'], trace_id, comic_id)

            # 5. 최종 결과 데이터 구성
            source_domain = urlparse(final_url_processed).netloc # 실제 스크랩된 URL의 도메인
            article_data = {
                'url': final_url_processed, # 실제 스크랩된 URL
                'original_url': url, # 노드에 입력된 원본 URL
                'title': extracted_content.get('title', ''),
                'text': extracted_content.get('text', ''),
                'publish_date': extracted_content.get('publish_date'), # 파싱된 날짜 문자열
                'language': language, # 감지된 언어 코드
                'source_domain': source_domain,
                'extraction_method': extracted_content.get('method', 'unknown') # 사용된 추출 방법
            }
            logger.info(f"기사 처리 성공: {final_url_processed} (언어: {language}, 방법: {article_data['extraction_method']})", extra=extra_log_data)
            return article_data

        except Exception as e:
            # 전체 URL 처리 과정에서 발생한 예외 처리
            stack_trace = traceback.format_exc() # 상세 스택 트레이스 로깅
            logger.error(f"URL 처리 중 심각한 오류 발생: {url} | 오류: {e}\n{stack_trace}", extra=extra_log_data)
            return None # 오류 발생 시 None 반환

    async def close(self):
        """내부적으로 생성된 aiohttp ClientSession을 닫습니다."""
        if self._session and self._created_session and not self._session.closed:
            await self._session.close()
            logger.info("ArticleScraperTool 내부에서 생성된 aiohttp ClientSession을 닫았습니다.")
        elif self._session and not self._created_session:
             logger.debug("외부에서 주입된 aiohttp 세션은 ArticleScraperTool에서 닫지 않습니다.")