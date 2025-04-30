# app/agents/scraper_agent.py
import logging
import asyncio # asyncio.gather 사용 위해 추가
import httpx
from bs4 import BeautifulSoup
from app.workflows.state import ComicState
from typing import Dict, Optional, Any, List # List 추가

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # 로깅 설정

class ScraperAgent:
    """
    URL 리스트를 받아 각 URL의 본문 텍스트를 비동기적으로 스크래핑하는 에이전트.
    """

    async def _scrape_single_url(self, url: str) -> Optional[str]:
        """단일 URL을 스크래핑하는 내부 비동기 함수"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        try:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=20.0) as client: # 타임아웃 조정
                logger.debug(f"[Scraper] Requesting URL: {url}")
                response = await client.get(url)
                response.raise_for_status()
                logger.debug(f"[Scraper] Fetched URL: {url}, Status: {response.status_code}")
                soup = BeautifulSoup(response.content, 'lxml')

                # --- 실제 본문 추출 로직 (이전과 동일, 사용자 정의 필요) ---
                article_body_element = None
                scraped_content = None
                selectors_to_try = [
                    'article', '.article-body', '#article_content', '.news_view',
                    '.content', '#content', '#main-content', '.post-content',
                    'div[itemprop="articleBody"]'
                ]
                for selector in selectors_to_try:
                    element = soup.select_one(selector)
                    if element: article_body_element = element; break
                if not article_body_element:
                    main_area = soup.find('main') or soup.select_one('#main') or soup.body
                    if main_area:
                        paragraphs = main_area.find_all('p')
                        if paragraphs: scraped_content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                if article_body_element and not scraped_content:
                    for unwanted in article_body_element(['script', 'style', 'aside', 'figure', 'figcaption', 'footer', '.ad']): unwanted.decompose()
                    scraped_content = article_body_element.get_text(separator='\n', strip=True)
                # --- ---

                if scraped_content and len(scraped_content) > 50:
                    logger.info(f"[Scraper] Successfully scraped URL: {url}. Length: {len(scraped_content)}")
                    return scraped_content
                else:
                    logger.warning(f"[Scraper] Failed to extract sufficient content from URL: {url}. Length: {len(scraped_content or '')}")
                    return None # 내용 부족 시 None 반환

        except httpx.HTTPStatusError as e:
            logger.error(f"[Scraper] HTTP error for URL {url}: Status {e.response.status_code}")
            return None # 실패 시 None 반환
        except Exception as e:
            logger.error(f"[Scraper] Error scraping URL {url}: {e}")
            return None # 실패 시 None 반환

    async def run(self, state: ComicState) -> Dict[str, Optional[Any]]:
        """
        state.news_urls 리스트의 각 URL에 대해 비동기적으로 스크래핑을 수행하고,
        결과 리스트를 state.articles 에 저장합니다.
        """
        logger.info("--- [Scraper Agent] 실행 시작 ---")
        updates: Dict[str, Optional[Any]] = {}
        urls_to_scrape = state.news_urls # URL 리스트를 가져옴

        if not urls_to_scrape:
            logger.warning("[Scraper Agent] 스크랩할 URL 리스트가 비어있습니다.")
            updates["articles"] = [] # 빈 리스트로 업데이트
            updates["error_message"] = "No URLs provided for scraping."
            return updates

        logger.info(f"[Scraper Agent] Received {len(urls_to_scrape)} URLs to scrape: {urls_to_scrape}")

        # asyncio.gather를 사용하여 여러 URL 스크래핑을 동시에 실행
        scrape_tasks = [self._scrape_single_url(url) for url in urls_to_scrape]
        scraped_results = await asyncio.gather(*scrape_tasks)

        # 결과 처리: None이 아닌 성공한 결과만 리스트로 저장
        successful_scrapes = [content for content in scraped_results if content is not None]

        logger.info(f"[Scraper Agent] Successfully scraped {len(successful_scrapes)} out of {len(urls_to_scrape)} URLs.")

        if not successful_scrapes:
            logger.error("[Scraper Agent] Failed to scrape any content from the provided URLs.")
            updates["articles"] = []
            updates["error_message"] = "Failed to scrape content from any of the provided URLs."
        else:
            # ComicState의 articles 필드가 List[str] 임을 가정
            updates["articles"] = successful_scrapes
            updates["error_message"] = None # 하나라도 성공하면 전체 오류는 없는 것으로 처리 (정책 변경 가능)

        # selected_url 필드는 이제 사용하지 않거나, 첫 번째 성공 URL 등으로 업데이트 가능 (선택 사항)
        # updates["selected_url"] = urls_to_scrape[scraped_results.index(successful_scrapes[0])] if successful_scrapes else None

        logger.info("--- [Scraper Agent] 실행 종료 ---")
        return updates