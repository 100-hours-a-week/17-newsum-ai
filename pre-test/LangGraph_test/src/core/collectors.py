# src/core/collectors.py
import asyncio
import aiohttp
import feedparser
from newspaper import Article
from datetime import datetime
from .schemas import NewsItem
from .utils import generate_id_from_url, logger
from src import settings
from typing import List

async def fetch_rss_feed(session, url):
    """비동기로 단일 RSS 피드 파싱"""
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                content = await response.text()
                feed = feedparser.parse(content)
                items = []
                for entry in feed.entries[:10]: # 최신 10개만 가져오기 (조절 가능)
                    pub_time = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else None
                    item = NewsItem(
                        id=generate_id_from_url(entry.link),
                        title=entry.title,
                        url=entry.link,
                        content=entry.summary if hasattr(entry, 'summary') else '', # 초기 콘텐츠는 요약으로
                        source=feed.feed.title if hasattr(feed.feed, 'title') else url,
                        published_time=pub_time
                    )
                    items.append(item)
                return items
            else:
                logger.warning(f"Failed to fetch RSS {url}: Status {response.status}")
                return []
    except Exception as e:
        logger.error(f"Error fetching RSS {url}: {e}")
        return []

async def fetch_article_content(session, news_item: NewsItem):
    """newspaper3k를 이용해 기사 본문 가져오기"""
    try:
        loop = asyncio.get_running_loop()
        # newspaper3k 사용 방식으로 수정
        article = await loop.run_in_executor(None, Article, news_item.url)
        await loop.run_in_executor(None, article.download)
        await loop.run_in_executor(None, article.parse)
        if article.text:
            news_item.content = article.text # 요약 대신 본문으로 업데이트
        return news_item
    except Exception as e:
        logger.warning(f"Failed to parse article {news_item.url} with newspaper3k: {e}")
        return news_item # 실패 시 기존 content 유지

async def fetch_latest_news(sources: List[str] = settings.NEWS_SOURCES) -> List[NewsItem]:
    """여러 소스에서 최신 뉴스 수집"""
    all_news = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_rss_feed(session, url) for url in sources if url.startswith('rss://') or url.startswith('http')]
        results = await asyncio.gather(*tasks)
        for news_list in results:
            all_news.extend(news_list)

    # ID 기준 중복 제거
    unique_news_dict = {}
    for item in all_news:
        if item.id not in unique_news_dict:  # 간단한 중복 처리
            unique_news_dict[item.id] = item
    unique_news = list(unique_news_dict.values())

    # (선택적) 기사 본문 상세화 - newspaper3k 사용
    async with aiohttp.ClientSession() as detail_session: # 별도 세션 또는 기존 세션 사용
        content_tasks = [fetch_article_content(detail_session, item) for item in unique_news]
        detailed_news = await asyncio.gather(*content_tasks)
        logger.info(f"Collected and detailed {len(detailed_news)} unique news items.")
        return list(detailed_news)

    logger.info(f"Collected {len(unique_news)} unique news items (content detail fetching skipped/optional).")
    return list(unique_news)

# ----------------------------------------
