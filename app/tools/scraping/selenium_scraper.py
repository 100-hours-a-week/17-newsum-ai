# app/tools/scraping/selenium_scraper.py

import asyncio
import random
import time
import re
import traceback # 상세 오류 로깅용
from typing import Dict, Any, Optional
from urllib.parse import urlparse

# Selenium 및 관련 라이브러리 동적 임포트
SELENIUM_AVAILABLE = False
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException
    # webdriver-manager 사용 고려 시 주석 해제
    # from selenium.webdriver.chrome.service import Service as ChromeService
    # from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    webdriver = None # type: ignore
    WebDriverException = Exception # type: ignore
    TimeoutException = Exception # type: ignore
    NoSuchElementException = Exception # type: ignore
    By = None # type: ignore
    EC = None # type: ignore
    # logger는 아래에서 초기화되므로 여기서 로깅 불가

from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger("SeleniumScraperTool")

# 라이브러리 부재 시 경고
if not SELENIUM_AVAILABLE:
    logger.warning("Selenium 관련 라이브러리가 설치되지 않았습니다. Selenium 스크래핑이 비활성화됩니다.")

class SeleniumScraperTool:
    """
    Selenium WebDriver를 사용하여 웹 페이지 콘텐츠를 스크랩하는 도구입니다.
    WebDriver 초기화, 페이지 로딩, 콘텐츠 추출 및 드라이버 종료를 관리합니다.
    """
    # TODO: 사용자 에이전트 로테이션 구현 시 필요
    # USER_AGENTS = settings.USER_AGENT_LIST or [settings.SCRAPER_USER_AGENT]

    def __init__(self):
        """SeleniumScraperTool 초기화."""
        self.grid_url = settings.SELENIUM_GRID_URL
        self.is_headless = settings.SELENIUM_HEADLESS
        self.use_proxy = settings.SCRAPER_USE_PROXY
        self.proxy_url = settings.SCRAPER_PROXY_URL
        self.rotate_ua = settings.SCRAPER_ROTATE_UA
        self.default_ua = settings.SCRAPER_USER_AGENT
        self.min_delay_ms = settings.SCRAPER_MIN_DELAY_MS
        self.max_delay_ms = settings.SCRAPER_MAX_DELAY_MS
        self.min_text_length = settings.MIN_EXTRACTED_TEXT_LENGTH
        self.retry_attempts = settings.SELENIUM_RETRY_ATTEMPTS
        self.retry_wait = settings.SELENIUM_RETRY_WAIT_SECONDS

        self._driver: Optional[webdriver.Remote] = None # 드라이버 인스턴스 (Lazy init)

    def _get_driver(self, trace_id: str, comic_id: str) -> Optional[webdriver.Remote]:
        """Selenium WebDriver 인스턴스를 가져오거나 초기화합니다."""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium 사용 불가 (라이브러리 미설치)", extra=extra_log_data)
            return None

        # 이미 드라이버가 실행 중이면 반환 (재사용)
        # 주의: 단일 드라이버 재사용 시 상태 문제 발생 가능성 있음 (쿠키, 세션 등)
        # 매번 새로 생성하는 것이 더 안정적일 수 있음 (self._driver = None 후 생성)
        if self._driver:
             # 간단한 health check (선택 사항)
             try:
                  _ = self._driver.current_url # 드라이버 상태 확인 시도
                  logger.debug("기존 Selenium 드라이버 재사용", extra=extra_log_data)
                  return self._driver
             except WebDriverException:
                  logger.warning("기존 Selenium 드라이버가 응답하지 않음. 새로 생성 시도.", extra=extra_log_data)
                  self.quit_driver(trace_id, comic_id) # 기존 드라이버 종료

        logger.info("Selenium WebDriver 초기화 시도...", extra=extra_log_data)
        try:
            # TODO: 브라우저 선택 기능 추가 (예: Firefox)
            options = webdriver.ChromeOptions()
            if self.is_headless: options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            # 페이지 로딩 전략 설정 (eager: DOM 완료 시, normal: 전체 로드, none: 즉시 반환)
            options.page_load_strategy = 'normal' # 'normal' 또는 'eager'

            # TODO: 사용자 에이전트 로테이션 구현
            current_ua = self.default_ua
            if self.rotate_ua:
                # current_ua = random.choice(self.USER_AGENTS)
                pass # 실제 로테이션 로직 필요
            options.add_argument(f"user-agent={current_ua}")

            # 프록시 설정
            if self.use_proxy and self.proxy_url:
                logger.info(f"프록시 사용 설정: {self.proxy_url}", extra=extra_log_data)
                options.add_argument(f'--proxy-server={self.proxy_url}')

            # Selenium Grid 또는 로컬 드라이버 사용
            if self.grid_url:
                 logger.info(f"Selenium Grid에 연결 시도: {self.grid_url}", extra=extra_log_data)
                 self._driver = webdriver.Remote(
                      command_executor=self.grid_url, options=options )
            else:
                 logger.info("로컬 ChromeDriver 시작 시도...", extra=extra_log_data)
                 # webdriver-manager 사용 예시 (주석 처리됨)
                 # service = ChromeService(ChromeDriverManager().install())
                 # self._driver = webdriver.Chrome(service=service, options=options)
                 # PATH에서 chromedriver 찾기 가정
                 self._driver = webdriver.Chrome(options=options)

            logger.info("Selenium WebDriver 초기화 성공.", extra=extra_log_data)
            # 암시적 대기 설정 (권장되지 않으나 간단한 경우 사용 가능)
            # self._driver.implicitly_wait(5) # 5초
            return self._driver

        except WebDriverException as e:
            logger.error(f"Selenium WebDriver 초기화 실패 (WebDriverException): {e}. WebDriver가 설치 및 PATH에 설정되었는지 확인하세요.", exc_info=True, extra=extra_log_data)
            self._driver = None
            return None
        except Exception as e:
             logger.error(f"Selenium 초기화 중 예상치 못한 오류 발생: {e}", exc_info=True, extra=extra_log_data)
             self._driver = None
             return None

    def quit_driver(self, trace_id: str, comic_id: str):
        """현재 WebDriver 인스턴스를 종료합니다."""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        if self._driver:
            logger.info("Selenium WebDriver 종료 중...", extra=extra_log_data)
            try:
                self._driver.quit()
            except Exception as e:
                logger.error(f"Selenium WebDriver 종료 중 오류 발생: {e}", extra=extra_log_data)
            finally:
                self._driver = None # 종료 후 None으로 설정

    @retry(
        stop=stop_after_attempt(settings.SELENIUM_RETRY_ATTEMPTS), # Selenium 재시도 설정 사용
        wait=wait_fixed(settings.SELENIUM_RETRY_WAIT_SECONDS), # 고정 대기 사용
        retry=retry_if_exception_type(WebDriverException), # WebDriver 관련 예외 발생 시 재시도
        reraise=True # 최종 실패 시 예외 다시 발생
    )
    def _scrape_url_sync(self, driver: webdriver.Remote, url: str, platform: str, trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """
        동기 함수: Selenium을 사용하여 특정 URL 스크랩. 재시도 로직은 외부 decorator에 의해 처리됨.
        """
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, 'platform': platform}
        logger.debug(f"Selenium 스크래핑 실행: {url}", extra=extra_log_data)

        # 페이지 로드
        driver.get(url)

        # --- 플랫폼별 또는 일반적인 콘텐츠 로딩 대기 ---
        # WebDriverWait와 EC (Expected Conditions)를 사용하여 특정 요소가 나타날 때까지 대기하는 것이 좋음
        wait_time = 10 # 최대 대기 시간 (초)
        try:
            if platform == "Twitter":
                # 예시: 트윗 컨테이너 대기 (실제 셀렉터는 변경될 수 있음)
                WebDriverWait(driver, wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-testid='tweet']"))
                )
            elif platform == "Reddit":
                 # 예시: 게시글 또는 댓글 컨테이너 대기
                 WebDriverWait(driver, wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-testid='post-container'], #comment-tree"))
                 )
            elif platform == "YouTube":
                 # 예시: 비디오 정보 섹션 대기
                 WebDriverWait(driver, wait_time).until(
                      EC.presence_of_element_located((By.CSS_SELECTOR, "#description, #info"))
                 )
            else: # Blog, Community, OtherWeb
                # 일반적인 콘텐츠 영역 대기 시도 또는 고정 시간 대기
                # WebDriverWait(driver, wait_time).until(EC.presence_of_element_located((By.CSS_SELECTOR, "article, main, #content")))
                time.sleep(random.uniform(1.0, 3.0)) # 단순 대기 (비권장)

        except TimeoutException:
            logger.warning(f"콘텐츠 로딩 시간 초과 ({wait_time}초). 스크래핑 계속 진행.", extra=extra_log_data)
        except Exception as wait_err:
             logger.warning(f"콘텐츠 로딩 대기 중 오류 발생: {wait_err}", extra=extra_log_data)

        # --- 데이터 추출 ---
        # TODO: 플랫폼별 상세 추출 로직 구현 (Node 06의 _scrape_*_selenium 로직 참고 및 개선)
        # 현재는 일반적인 방식만 구현
        data = {"text": "", "author": None, "timestamp": None, "likes": 0, "title": ""}
        try: data["title"] = driver.title
        except Exception: pass

        # 텍스트 추출 (여러 셀렉터 시도)
        text_selectors = ["article", ".post-content", ".entry-content", ".td-post-content", "main", ".c-entry-content", "#content", ".content", "[role='main']"]
        extracted_text = ""
        for selector in text_selectors:
            try:
                 elements = driver.find_elements(By.CSS_SELECTOR, selector)
                 if elements:
                      candidate_text = max(elements, key=lambda e: len(e.text)).text
                      if len(candidate_text) > len(extracted_text): extracted_text = candidate_text
            except Exception: continue
        if not extracted_text or len(extracted_text) < self.min_text_length / 2 : # 본문 없으면 body 시도
             try: extracted_text = driver.find_element(By.TAG_NAME, 'body').text
             except Exception: pass
        data["text"] = extracted_text.strip() if extracted_text else ""

        # 작성자 추출
        author_selectors = [".author", ".byline", ".post-author", "a[rel='author']", "[itemprop='author']", ".writer", ".reporter"]
        for selector in author_selectors:
             try:
                  element = driver.find_element(By.CSS_SELECTOR, selector)
                  author_text = element.text or element.get_attribute('content')
                  if author_text: data["author"] = author_text.strip(); break
             except Exception: continue

        # 타임스탬프 추출
        time_selectors = ["time[datetime]", "[itemprop='datePublished']", ".published", ".post-date", ".entry-date", "span[data-testid='User-statuses'] > a > time"] # 트위터용 추가
        for selector in time_selectors:
             try:
                  element = driver.find_element(By.CSS_SELECTOR, selector)
                  time_str = element.get_attribute('datetime') or element.get_attribute('content') or element.text
                  if time_str: data["timestamp"] = time_str.strip(); break
             except Exception: continue

        # 좋아요 수 추출 (매우 부정확할 수 있음)
        like_selectors = [".like-count", ".likes", ".vote-count", ".recommend-count", "[data-testid='like']", "[data-testid='appreciationUnfilled']"] # 트위터/레딧 등 고려
        for selector in like_selectors:
             try:
                  element = driver.find_element(By.CSS_SELECTOR, selector)
                  # 여러 속성 확인 (aria-label, text 등)
                  likes_text = element.get_attribute('aria-label') or element.text
                  match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)\s*(?:likes|표|추천|개)', likes_text, re.IGNORECASE) # 숫자 및 단위 포함
                  if match: data["likes"] = int(match.group(1).replace(',', '')); break
             except Exception: continue

        # --- 검증 및 반환 ---
        # 최소 텍스트 길이 만족하는지 확인
        if len(data.get("text", "")) < self.min_text_length:
             logger.warning(f"스크랩된 텍스트 길이가 너무 짧음 ({len(data.get('text', ''))} < {self.min_text_length})", extra=extra_log_data)
             return None # 실패로 간주
        else:
             logger.info(f"Selenium 스크래핑 성공", extra=extra_log_data)
             return data

    async def scrape_url(self, url: str, platform: str, trace_id: str, comic_id: str) -> Optional[Dict[str, Any]]:
        """
        주어진 URL을 Selenium을 사용하여 스크랩합니다. WebDriver를 관리하고 동기 스크래핑 함수를 실행합니다.

        Args:
            url (str): 스크랩할 URL.
            platform (str): URL의 플랫폼 유형 (예: 'Blog', 'Twitter').
            trace_id (str): 로깅용 추적 ID.
            comic_id (str): 로깅용 코믹 ID.

        Returns:
            Optional[Dict[str, Any]]: 추출된 데이터 또는 실패 시 None.
        """
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url, 'platform': platform}

        # 랜덤 지연 적용 (Anti-scraping)
        delay = random.uniform(self.min_delay_ms, self.max_delay_ms) / 1000.0
        logger.debug(f"스크래핑 전 랜덤 지연 적용: {delay:.2f}초", extra=extra_log_data)
        await asyncio.sleep(delay)

        driver = self._get_driver(trace_id, comic_id) # 드라이버 가져오기 또는 생성
        if not driver:
            logger.error("Selenium 드라이버를 사용할 수 없어 스크래핑 불가", extra=extra_log_data)
            return None

        scraped_data = None
        try:
            loop = asyncio.get_running_loop()
            # 동기 스크래핑 함수를 executor에서 실행 (재시도 로직 포함)
            scraped_data = await loop.run_in_executor(
                None, self._scrape_url_sync, driver, url, platform, trace_id, comic_id
            )
        except WebDriverException as e: # 재시도 실패 후 최종 WebDriver 오류
             logger.error(f"Selenium 최종 실패 (WebDriverException): {e}", exc_info=True, extra=extra_log_data)
             # 드라이버가 불안정할 수 있으므로 리셋 고려
             self.quit_driver(trace_id, comic_id)
        except Exception as e: # 기타 예상치 못한 오류
             logger.error(f"Selenium 스크래핑 실행 중 예상치 못한 오류 발생: {e}", exc_info=True, extra=extra_log_data)

        # 참고: 드라이버 종료 시점 결정 필요
        # 매번 종료 vs 노드 실행 끝날 때 한번만 종료 vs 외부에서 관리
        # 여기서는 매번 종료하지 않고, 노드 레벨에서 마지막에 quit_driver 호출 가정
        # self.quit_driver(trace_id, comic_id) # 필요 시 여기서 매번 종료

        return scraped_data

    async def close(self, trace_id: str, comic_id: str):
        """도구 사용 완료 후 WebDriver 인스턴스를 종료합니다."""
        self.quit_driver(trace_id, comic_id)