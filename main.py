# app/main.py

import asyncio
import contextlib # contextmanager 사용 시 필요할 수 있음 (여기서는 직접 사용 안 함)
from contextlib import asynccontextmanager # lifespan 구현용
import logging
import uuid
import aiohttp

from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException, status

# --- 애플리케이션 구성 요소 임포트 ---
from app.config.settings import settings
from app.utils.logger import setup_logging, get_logger
from app.workflows.state import ComicState
# 노드, 서비스, 도구, 워크플로우 빌더 임포트 (경로 주의)
from app.nodes import (
    n01_initialize_node, n02_topic_analyzer_node, n03_news_collector_node, n04_opinion_collector_node,
    n05_news_scraper_node, n06_opinion_scraper_node, n07_filter_node # 필요한 노드 임포트
)
from app.services import ( llm_server_client_v2, database_con_client_v2 ) # 필요한 서비스 임포트
from app.tools.search import ( google, naver, rss )
from app.tools.social import ( twitter, reddit )
from app.tools.scraping import ( article_scraper, selenium_scraper )
from app.tools.analysis import ( language_detector, text_clusterer )
from app.services.spam_detector import SpamDetectionService # 서비스 경로 확인

from app.workflows.main_workflow import build_main_workflow # 워크플로우 빌더 임포트
# 의존성 주입 함수 임포트
from app.dependencies import get_compiled_workflow_app, get_db_client
# 백그라운드 작업 트리거 함수 임포트
from app.api.v1.background_tasks import trigger_workflow_task # 경로 확인
from langgraph.graph import StateGraph # 타입 힌트용

# --- 로깅 설정 ---
# setup_logging('logging_config.yaml') # 여기서 직접 호출하거나 main 실행 블록에서 호출
logger = get_logger("FastAPIApp")

# --- Lifespan 컨텍스트 매니저 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱의 시작 및 종료 시 리소스를 관리하는 lifespan 컨텍스트 매니저.
    """
    logger.info("애플리케이션 시작... 리소스 초기화 시작.")
    setup_logging('logging_config.yaml') # 로깅 설정 적용

    # --- 1. 공유 리소스 생성 (예: HTTP 세션, DB 클라이언트) ---
    app.state.shared_http_session = aiohttp.ClientSession()
    logger.info("공유 aiohttp 세션 생성됨.")
    app.state.db_client = database_con_client_v2.DatabaseClientV2() # DB 클라이언트 생성
    closable_resources = [app.state.shared_http_session, app.state.db_client] # 정리할 리소스 목록

    # --- 2. 서비스 및 도구 인스턴스화 ---
    logger.info("서비스/도구 인스턴스화 시작...")
    # 각 도구/서비스 생성 시 필요 시 세션/DB 클라이언트 전달
    llm_service = llm_server_client_v2.LLMService(); closable_resources.append(llm_service)
    google_tool = google.GoogleSearchTool(session=app.state.shared_http_session); closable_resources.append(google_tool)
    naver_tool = naver.NaverSearchTool() # NaverTool은 세션 직접 사용 안 함 가정
    rss_tool = rss.RssSearchTool()
    twitter_tool = twitter.TwitterTool()
    reddit_tool = reddit.RedditTool()
    article_scraper_tool = article_scraper.ArticleScraperTool(session=app.state.shared_http_session); closable_resources.append(article_scraper_tool)  # 변수명 수정
    selenium_tool = selenium_scraper.SeleniumScraperTool(); closable_resources.append(selenium_tool)
    lang_tool = language_detector.LanguageDetectionTool()
    spam_service = SpamDetectionService()
    cluster_tool = text_clusterer.TextClusteringTool()
    # TODO: 필요한 다른 서비스/도구들도 초기화 및 closable_resources에 추가
    logger.info("서비스/도구 인스턴스화 완료.")

    # --- 3. 노드 인스턴스화 (의존성 주입) ---
    logger.info("노드 인스턴스화 시작...")
    node01 = n01_initialize_node.InitializeNode()
    node02 = n02_topic_analyzer_node.TopicAnalyzerNode(llm_client=llm_service, db_client=app.state.db_client) # DB 클라이언트 주입
    node03 = n03_news_collector_node.NewsCollectorNode(Google=google_tool, naver=naver_tool, rss=rss_tool)
    node04 = n04_opinion_collector_node.OpinionCollectorNode(twitter_tool=twitter_tool, reddit_tool=reddit_tool, google_tool=google_tool)
    node05 = n05_news_scraper_node.NewsScraperNode(scraper_tool=article_scraper_tool)  # 변수명 수정
    node06 = n06_opinion_scraper_node.OpinionScraperNode(twitter_tool=twitter_tool, reddit_tool=reddit_tool, google_tool=google_tool, selenium_tool=selenium_tool)
    node07 = n07_filter_node.FilterNode(language_tool=lang_tool, spam_service=spam_service, clustering_tool=cluster_tool)
    # TODO: 필요한 모든 노드 초기화
    logger.info("노드 인스턴스화 완료.")

    # --- 4. 워크플로우 빌드 및 컴파일 ---
    # 모든 초기화된 노드를 build_main_workflow에 전달
    app.state.compiled_workflow_app = build_main_workflow(
        initialize_node=node01, topic_analyzer_node=node02, news_collector_node=node03,
        opinion_collector_node=node04, news_scraper_node=node05, opinion_scraper_node=node06,
        filter_node=node07
        # TODO: 모든 노드 인스턴스 전달
    )
    logger.info("워크플로우 그래프 빌드 및 컴파일 완료.")

    # --- 앱 실행 준비 완료 ---
    logger.info("애플리케이션 시작 준비 완료.")
    yield # 여기에서 애플리케이션이 실행됨

    # --- 앱 종료 시 리소스 정리 ---
    logger.info("애플리케이션 종료... 리소스 정리 시작.")
    for resource in reversed(closable_resources): # 생성 역순으로 정리
         try:
              if hasattr(resource, 'close') and callable(resource.close):
                   if asyncio.iscoroutinefunction(resource.close):
                        await resource.close() # 비동기 close
                        logger.info(f"리소스 정리됨 (async): {type(resource).__name__}")
                   else:
                        resource.close() # 동기 close
                        logger.info(f"리소스 정리됨 (sync): {type(resource).__name__}")
              # SeleniumScraperTool 특별 처리 (close가 비동기이며 인자 필요 가정)
              elif isinstance(resource, selenium_scraper.SeleniumScraperTool):
                  await resource.close(trace_id="shutdown", comic_id="shutdown")
                  logger.info(f"리소스 정리됨 (Selenium): {type(resource).__name__}")
         except Exception as close_err:
              logger.error(f"{type(resource).__name__} 리소스 정리 중 오류 발생: {close_err}", exc_info=True)
    logger.info("리소스 정리 완료.")


# --- FastAPI 앱 생성 ---
# title, description 등은 프로젝트에 맞게 수정
app = FastAPI(
    title="Comic Generation API",
    description="LangGraph 기반 만화 생성 워크플로우 API",
    version="1.0.0",
    lifespan=lifespan # lifespan 컨텍스트 매니저 등록
)

# --- API 라우터 정의 (예시) ---
# 실제 API 라우터는 별도 파일(예: app/api/v1/endpoints/comic.py)에서 정의하고 여기서 include 하는 것이 좋음
@app.post("/v1/comics/generate", status_code=status.HTTP_202_ACCEPTED) # 202 Accepted 반환
async def generate_comic_endpoint(
    query: str, # 요청 본문 또는 쿼리 파라미터로 받을 수 있음
    background_tasks: BackgroundTasks, # FastAPI의 백그라운드 작업 객체 주입
    # --- 의존성 주입 사용 ---
    compiled_app: StateGraph = Depends(get_compiled_workflow_app),
    db_client: database_con_client_v2.DatabaseClientV2 = Depends(get_db_client)
):
    """
    만화 생성 워크플로우를 백그라운드에서 실행하도록 요청합니다.
    """
    logger.info(f"API 요청 수신: 만화 생성 요청, query='{query}'")
    # 백그라운드 작업 트리거 함수 호출 (주입된 의존성 전달)
    comic_id = await trigger_workflow_task(query, background_tasks, compiled_app, db_client)
    logger.info(f"백그라운드 작업 시작됨. Comic ID: {comic_id}")
    # 클라이언트에게는 작업 ID와 함께 즉시 응답 반환
    return {"message": "만화 생성 작업이 시작되었습니다.", "comic_id": comic_id}

@app.get("/v1/comics/status/{comic_id}")
async def get_comic_status_endpoint(
    comic_id: str,
    db_client: database_con_client_v2.DatabaseClientV2 = Depends(get_db_client) # DB 클라이언트 주입
):
    """
    주어진 comic_id에 해당하는 작업의 상태를 조회합니다.
    """
    logger.info(f"API 요청 수신: 상태 조회, comic_id='{comic_id}'")
    status_data = await db_client.get(comic_id)
    if status_data:
        return status_data
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 ID의 만화 작업 상태를 찾을 수 없습니다.")


# --- (선택 사항) 메인 실행 블록 ---
# Uvicorn 등으로 직접 실행 시 이 블록은 필요 없음
if __name__ == "__main__":
    import uvicorn
    logger.info("Uvicorn 서버 시작...")
    # 실제 배포 시에는 Gunicorn + Uvicorn 워커 사용 권장
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")