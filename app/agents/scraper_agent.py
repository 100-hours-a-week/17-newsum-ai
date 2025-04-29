# app/agents/scraper_agent.py
import logging
import httpx # 비동기 HTTP 요청
from bs4 import BeautifulSoup # HTML 파싱
from app.workflows.state import ComicState # 상태 객체
from typing import Dict, Optional, Any # 타입 힌팅

logger = logging.getLogger(__name__)
# 로깅 레벨 및 형식 설정 (main.py 또는 별도 로깅 설정 파일에서 관리하는 것이 더 좋음)
# logging.basicConfig(level=logging.INFO) # 필요시 주석 해제 또는 수정

class ScraperAgent:
    """
    웹사이트 URL을 입력받아 해당 페이지의 본문 텍스트를 스크래핑하는 에이전트.
    httpx와 BeautifulSoup를 사용합니다.
    """
    async def run(self, state: ComicState) -> Dict[str, Optional[Any]]:
        """
        선택된 뉴스 URL로부터 본문 텍스트를 스크래핑하고,
        업데이트 내용을 담은 딕셔너리를 반환합니다.
        """
        logger.info("--- Scraper Agent 실행 시작 ---")
        # 업데이트할 내용을 담을 딕셔너리
        updates: Dict[str, Optional[Any]] = {}
        selected_url = state.selected_url # 이전 노드에서 전달된 URL

        if not selected_url:
            logger.warning("스크랩할 URL이 없습니다.")
            updates["error_message"] = "No URL provided for scraping."
            return updates # 업데이트 딕셔너리 반환

        # 웹사이트가 봇을 차단하는 것을 피하기 위해 사용자 에이전트 헤더 추가
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        }

        try:
            # httpx 클라이언트를 사용하여 비동기적으로 페이지 내용 가져오기
            # follow_redirects=True: 리다이렉션 따라가기
            # timeout=30.0: 타임아웃 30초 설정
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=30.0) as client:
                logger.info(f"Requesting URL: {selected_url}")
                response = await client.get(selected_url)
                # HTTP 오류 상태 코드(4xx, 5xx) 발생 시 예외 발생
                response.raise_for_status()

                logger.info(f"Successfully fetched URL. Status code: {response.status_code}")

                # BeautifulSoup 객체 생성 (lxml 파서 사용)
                # response.text 대신 response.content를 사용하면 인코딩 문제 방지에 도움될 수 있음
                soup = BeautifulSoup(response.content, 'lxml')

                # --- 중요: 실제 뉴스 본문 추출 로직 ---
                # 이 부분은 스크래핑 대상 웹사이트의 HTML 구조에 따라 맞춤 설정해야 합니다.
                # 브라우저 개발자 도구(F12)를 사용하여 적절한 선택자를 찾아야 합니다.

                article_body_element = None
                scraped_content = None

                # 일반적인 뉴스 기사 본문 선택자 예시 (우선순위대로 시도)
                # 실제 대상 사이트에 맞게 이 선택자 목록을 수정/추가/삭제해야 합니다.
                selectors_to_try = [
                    'article',                     # <article> 태그
                    '.article-body',               # class="article-body"
                    '#article_content',            # id="article_content"
                    '.news_view',                  # class="news_view" (언론사별 확인)
                    '.content',                    # class="content"
                    '#content',                    # id="content"
                    '#main-content',               # id="main-content"
                    '.post-content',               # class="post-content"
                    'div[itemprop="articleBody"]'  # itemprop 속성 사용
                ]

                for selector in selectors_to_try:
                    element = soup.select_one(selector)
                    if element:
                        article_body_element = element
                        logger.info(f"Found potential article body using selector: '{selector}'")
                        break # 첫 번째로 찾은 요소를 사용

                # 특정 컨테이너를 찾지 못한 경우, 메인 영역의 <p> 태그 수집 시도 (덜 정확함)
                if not article_body_element:
                    logger.warning(f"Could not find specific container using selectors: {selectors_to_try}. Trying to gather <p> tags from main area.")
                    # main 태그나 특정 ID/Class를 가진 div 영역을 먼저 찾아 그 안의 p 태그만 추출 시도
                    main_area = soup.find('main') or soup.select_one('#main') or soup.select_one('.main') or soup.body
                    if main_area:
                        paragraphs = main_area.find_all('p')
                        if paragraphs:
                             paragraph_texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
                             # 너무 짧은 문단 제외 등 추가 필터링 가능
                             scraped_content = "\n\n".join(paragraph_texts) # 문단 간 줄바꿈 추가
                             logger.info(f"Gathered {len(paragraphs)} <p> tags.")

                # 선택된 요소에서 텍스트 추출
                if article_body_element and not scraped_content: # 컨테이너 요소를 찾았고, p태그 방식 사용 안했을 때
                    # 불필요한 요소(광고, 스크립트 등) 제거 시도
                    for unwanted in article_body_element(['script', 'style', 'aside', 'figure', 'figcaption', 'footer', '.ad', '.advertisement']):
                        unwanted.decompose()
                    # 텍스트 추출
                    scraped_content = article_body_element.get_text(separator='\n', strip=True)

                # ----------------------------------------

                if scraped_content and len(scraped_content) > 50: # 최소 길이 체크 (임계값 조정 필요)
                    logger.info(f"스크래핑 성공. 추출된 텍스트 길이: {len(scraped_content)}")
                    # 상태 업데이트 내용 추가 (ComicState 필드 타입 확인 필요)
                    # 예: 필드가 Optional[str] 인 경우:
                    updates["articles"] = scraped_content
                    # 예: 필드가 Optional[List[str]] 인 경우:
                    # updates["articles"] = [scraped_content]
                    updates["error_message"] = None # 성공 시 오류 메시지 초기화
                elif scraped_content:
                     logger.warning(f"Extracted text might be too short (length: {len(scraped_content)}). Check selectors.")
                     updates["error_message"] = "Extracted content is too short."
                else:
                    logger.error("Failed to extract meaningful article content.")
                    updates["error_message"] = "Could not extract article content from the page."

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred while scraping {selected_url}: Status {e.response.status_code}")
            updates["error_message"] = f"HTTP error: {e.response.status_code}"
        except httpx.RequestError as e:
            logger.error(f"Network error occurred while scraping {selected_url}: {type(e).__name__}")
            updates["error_message"] = f"Network error: {type(e).__name__}"
        except Exception as e:
            # 예상치 못한 오류 로깅
            logger.exception(f"Scraper Agent 실행 중 예외 발생: {e}")
            updates["error_message"] = f"An unexpected error occurred during scraping: {str(e)}"

        logger.info("--- Scraper Agent 실행 종료 ---")
        # 변경된 필드만 담은 딕셔너리 반환
        return updates