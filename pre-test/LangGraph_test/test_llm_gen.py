# test_llm_multi_save.py
import asyncio
import csv
import uuid
from datetime import datetime
from pathlib import Path

# 필요한 모듈 및 함수 임포트
from src.core.analyzers import analyze_article
from src.core.humorators import make_joke
from src.core.schemas import NewsItem, AnalysisResult, HumorResult
from src.core.utils import logger
from src import settings

# 여러 개의 테스트용 뉴스 샘플 정의
sample_news_list = [
    NewsItem(
        id=f"news_{uuid.uuid4().hex[:6]}",
        title="Local Library Sees Record Visitors After New Cafe Opens Inside",
        url="http://example.com/library_cafe",
        content="The city's main library reported a surprising surge in visitor numbers this past month. Librarians attribute the spike not to a newfound love for literature, but to the popular new artisanal coffee shop that opened in the main lobby. 'People come for the lattes, and maybe check out a book,' one librarian commented.",
        source="Local Gazette",
        published_time=datetime.now()
    ),
    NewsItem(
        id=f"news_{uuid.uuid4().hex[:6]}",
        title="Scientists Discover Singing Mice on Remote Island",
        url="http://example.com/singing_mice",
        content="A team of researchers has reported the discovery of a unique species of mouse on a previously unexplored island. These mice communicate using complex, melodious chirps resembling songs. 'It's unlike anything we've ever heard in rodents,' said lead scientist Dr. Anya Sharma. Studies are underway to decode their 'language'.",
        source="Nature Journal",
        published_time=datetime.now()
    ),
    NewsItem(
        id=f"news_{uuid.uuid4().hex[:6]}",
        title="Stock Market Hits Record High Amidst Economic Uncertainty",
        url="http://example.com/stock_market",
        content="Major stock indices reached unprecedented levels today, baffling some analysts given the ongoing global economic challenges. Tech stocks led the rally, fueled by AI optimism. 'The market seems disconnected from reality,' noted one economist, 'but investors are clearly bullish.'",
        source="Financial Times",
        published_time=datetime.now()
    ),
    # --- 여기에 더 많은 뉴스 샘플 추가 ---
    # NewsItem(id=..., title=..., url=..., content=..., source=..., published_time=...),
]

# 결과를 저장할 CSV 파일 경로
output_csv_path = settings.OUTPUT_DIR / "llm_test_results.csv"

async def process_single_news(news_item: NewsItem) -> dict:
    """단일 뉴스 아이템을 분석하고 유머를 생성하여 결과를 딕셔너리로 반환"""
    result_row = {
        "news_id": news_item.id,
        "news_title": news_item.title,
        "analysis_summary": "N/A",
        "humor_text": "N/A",
        "error": ""
    }

    analysis_result: AnalysisResult | None = None
    try:
        logger.info(f"Analyzing news: {news_item.id} - {news_item.title[:30]}...")
        analysis_result = await analyze_article(news_item)
        if analysis_result and analysis_result.summary:
             result_row["analysis_summary"] = analysis_result.summary
        else:
             result_row["error"] += "Analysis failed or returned no summary. "
        logger.info(f"Analysis complete for {news_item.id}")

    except Exception as e:
        logger.error(f"Error during analysis for {news_item.id}: {e}", exc_info=False) # 상세 스택 트레이스 제외
        result_row["error"] += f"Analysis Error: {e}. "
        analysis_result = None # 분석 실패 시 None 설정

    # 분석 성공 시 유머 생성 시도
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
    """여러 뉴스에 대해 LLM 테스트를 수행하고 결과를 CSV 파일에 저장"""
    logger.info("--- Starting Multi-News LLM Test & Save ---")
    logger.info(f"Results will be saved to: {output_csv_path}")

    # CSV 파일 쓰기 준비
    output_csv_path.parent.mkdir(parents=True, exist_ok=True) # 디렉토리 생성
    fieldnames = ["news_id", "news_title", "analysis_summary", "humor_text", "error"]

    results_to_write = []
    tasks = [process_single_news(news_item) for news_item in sample_news_list]
    # asyncio.gather를 사용하여 병렬 처리 (API 호출 부하 주의)
    test_results = await asyncio.gather(*tasks)
    results_to_write.extend(test_results)

    # CSV 파일에 결과 저장
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_to_write)
        logger.info(f"Successfully saved {len(results_to_write)} results to {output_csv_path}")
    except Exception as e:
        logger.error(f"Failed to write results to CSV: {e}", exc_info=True)

    logger.info("--- Test Finished ---")

if __name__ == "__main__":
    asyncio.run(main())