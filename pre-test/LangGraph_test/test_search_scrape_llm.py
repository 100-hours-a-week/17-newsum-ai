# test_search_scrape_llm.py
import asyncio
import csv
import uuid
from datetime import datetime
from pathlib import Path
import time

# 검색 라이브러리 임포트
from duckduckgo_search import DDGS

# 뉴스 스크래핑 라이브러리 임포트
from newspaper import Article, ArticleException

# 기존 모듈 임포트
from src.core.analyzers import analyze_article
from src.core.humorators import make_joke
from src.core.schemas import NewsItem, AnalysisResult, HumorResult
from src.core.utils import logger, generate_id_from_url
from src import settings

# --- 설정 ---
SEARCH_QUERY = "top news today Korea" # 검색어 설정 (영어로 검색하는 것이 LLM 처리에 유리)
MAX_SEARCH_RESULTS = 5 # 가져올 검색 결과 수
# 결과를 저장할 CSV 파일 경로
output_csv_path = settings.OUTPUT_DIR / "search_llm_test_results.csv"

async def scrape_article_content(url: str) -> dict | None:
    """newspaper3k를 사용하여 기사 내용 스크래핑 (비동기 실행용 래퍼)"""
    logger.info(f"Attempting to scrape content from: {url}")
    try:
        article = Article(url)
        # newspaper3k의 download/parse는 동기 함수이므로 executor 사용
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, article.download)
        # 짧은 딜레이 추가 (선택적, 과도한 요청 방지)
        await asyncio.sleep(0.5)
        await loop.run_in_executor(None, article.parse)

        if article.text and len(article.text) > 100: # 너무 짧은 내용은 제외
            logger.info(f"Successfully scraped content from: {url} (Length: {len(article.text)})")
            return {
                "title": article.title if article.title else "Title not found",
                "content": article.text,
                "publish_date": article.publish_date
            }
        else:
            logger.warning(f"Failed to scrape sufficient content from: {url} (Content length: {len(article.text)})")
            return None
    except ArticleException as e:
        logger.warning(f"newspaper3k failed for URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {e}", exc_info=False)
        return None

async def process_single_scraped_news(scraped_data: dict, url: str) -> dict:
    """스크랩된 뉴스 데이터를 분석하고 유머를 생성"""
    news_item = NewsItem(
        id=generate_id_from_url(url),
        title=scraped_data.get("title", "Unknown Title"),
        url=url,
        content=scraped_data.get("content", ""),
        source=url.split('/')[2], # 도메인을 소스로 사용
        published_time=scraped_data.get("publish_date")
    )

    result_row = {
        "news_id": news_item.id,
        "news_url": url,
        "news_title": news_item.title,
        "scraped_content_length": len(news_item.content),
        "analysis_summary": "N/A",
        "humor_text": "N/A",
        "error": ""
    }

    analysis_result: AnalysisResult | None = None
    try:
        logger.info(f"Analyzing scraped news: {news_item.id} - {news_item.title[:30]}...")
        analysis_result = await analyze_article(news_item)
        if analysis_result and analysis_result.summary:
             result_row["analysis_summary"] = analysis_result.summary
        else:
             result_row["error"] += "Analysis failed or returned no summary. "
        logger.info(f"Analysis complete for {news_item.id}")

    except Exception as e:
        logger.error(f"Error during analysis for {news_item.id}: {e}", exc_info=False)
        result_row["error"] += f"Analysis Error: {e}. "
        analysis_result = None

    if analysis_result:
        try:
            logger.info(f"Generating humor for {news_item.id}...")
            humor_result = await make_joke(analysis_result)
            if humor_result:
                result_row["humor_text"] = humor_result.humor_text
            else:
                result_row["error"] += "Humor generation returned None. "
            logger.info(f"Humor generation complete for {news_item.id}")

        except Exception as e:
            logger.error(f"Error during humor generation for {news_item.id}: {e}", exc_info=False)
            result_row["error"] += f"Humor Gen Error: {e}. "

    return result_row

async def main():
    """웹 검색 -> 뉴스 스크래핑 -> LLM 처리 -> 결과 저장"""
    logger.info(f"--- Starting Web Search & Scrape & LLM Test ---")
    logger.info(f"Search Query: '{SEARCH_QUERY}'")
    logger.info(f"Max Results: {MAX_SEARCH_RESULTS}")
    logger.info(f"Results will be saved to: {output_csv_path}")

    results_to_write = []
    try:
        # 1. 웹 검색 실행
        logger.info("Performing web search...")
        search_results = []  # 초기화

        ddgs = DDGS()

        # 수정된 부분: text 메소드 호출 후 결과를 바로 할당
        results_generator = ddgs.text(  # text 메소드는 검색 결과 리스트를 반환할 것으로 예상
            SEARCH_QUERY,
            region='wt-wt',
            safesearch='off',
            timelimit='d',
            max_results=MAX_SEARCH_RESULTS
        )

        # 수정된 부분: async for 제거, 결과가 리스트이므로 바로 할당
        if results_generator and isinstance(results_generator, list):
            search_results = results_generator
        elif results_generator:
            # 혹시 다른 타입이면 로깅 (예: 제너레이터)
            logger.warning(f"Search returned unexpected type: {type(results_generator)}. Attempting list conversion.")
            try:
                search_results = list(results_generator)  # 동기 제너레이터일 경우 list로 변환
            except TypeError:
                logger.error("Could not convert search results to list.")
                search_results = []  # 안전하게 빈 리스트로
        else:
            logger.warning("Search results generator is None or empty.")
            search_results = []

        # async with 구문 및 async for 제거됨

        if not search_results:
            logger.warning("No search results found.")
            return

        logger.info(f"Found {len(search_results)} search results. Starting scraping and processing...")

        # 2. 검색 결과 스크래핑 및 LLM 처리 (이전과 동일)
        processing_tasks = []
        for result in search_results:
            url = result.get('href')
            if url:
                # 스크래핑 후 LLM 처리하는 태스크 생성
                async def scrape_and_process(url_to_process):
                    scraped = await scrape_article_content(url_to_process)
                    if scraped:
                        return await process_single_scraped_news(scraped, url_to_process)
                    else:
                        return {
                            "news_id": generate_id_from_url(url_to_process),
                            "news_url": url_to_process,
                            "news_title": result.get('title', 'Scraping Failed'),
                            "scraped_content_length": 0,
                            "analysis_summary": "N/A",
                            "humor_text": "N/A",
                            "error": "Scraping failed."
                        }
                processing_tasks.append(scrape_and_process(url))

        test_results = await asyncio.gather(*processing_tasks)
        results_to_write.extend(test_results)


    except Exception as e:
        logger.error(f"An error occurred during search or processing: {e}", exc_info=True)

    # 3. CSV 파일에 결과 저장
    if results_to_write:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["news_id", "news_url", "news_title", "scraped_content_length", "analysis_summary", "humor_text", "error"]
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_to_write)
            logger.info(f"Successfully saved {len(results_to_write)} results to {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to write results to CSV: {e}", exc_info=True)
    else:
        logger.warning("No results were processed to save.")

    logger.info("--- Test Finished ---")

if __name__ == "__main__":
    asyncio.run(main())