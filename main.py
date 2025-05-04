# app/main.py (Upgraded Version)

import asyncio
from contextlib import asynccontextmanager
import aiohttp

from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException, status
from langgraph.graph import StateGraph # 타입 힌트 및 의존성 주입용

# --- 애플리케이션 구성 요소 임포트 ---
from app.config.settings import settings
from app.api.v1 import endpoints as v1_endpoints # 라우터 모듈 임포트
from app.utils.logger import setup_logging, get_logger

# --- 모든 노드 클래스 임포트 ---
from app.nodes import (
    n01_initialize_node, n02_topic_analyzer_node, n03_news_collector_node,
    n04_opinion_collector_node, n05_news_scraper_node, n06_opinion_scraper_node,
    n07_filter_node, n08_news_summarizer_node, n09_opinion_summarizer_node,
    n10_synthesis_summarizer_node, n11_evaluate_summary_node, n12_trend_analyzer_node,
    n13_progress_report_node, n14_idea_generator_node, n15_scenario_writer_node,
    n16_scenario_report_node, n17_imager_node, n18_translator_node,
    n19_postprocessor_node
)

# --- 모든 서비스 및 도구 클래스 임포트 ---
# Services
from app.services.llm_server_client_v2 import LLMService
from app.services.database_con_client_v2 import DatabaseClientV2
from app.services.image_server_client_v2 import ImageGenerationClient # 추가
from app.services.papago_translation_service import PapagoTranslationService # 추가
from app.services.storage_client_v2 import StorageClient # 추가
from app.services.spam_detector import SpamDetectionService

# Tools
from app.tools.search.google import GoogleSearchTool
from app.tools.search.naver import NaverSearchTool
from app.tools.search.rss import RssSearchTool
from app.tools.social.twitter import TwitterTool
from app.tools.social.reddit import RedditTool
from app.tools.scraping.article_scraper import ArticleScraperTool
from app.tools.scraping.selenium_scraper import SeleniumScraperTool
from app.tools.analysis.language_detector import LanguageDetectionTool
from app.tools.analysis.text_clusterer import TextClusteringTool
from app.tools.trends.google_trends import GoogleTrendsTool # 추가
from app.tools.trends.twitter_counts import TwitterCountsTool # 추가

# 워크플로우 빌더, 의존성 주입, 백그라운드 작업 함수 임포트
from app.workflows.main_workflow import build_main_workflow
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

# ------------------------------------
import os
# main.py 파일의 디렉토리 (프로젝트 루트 가정)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(PROJECT_ROOT, 'app/config/logging_config.yaml')

# 로그 디렉토리 생성 확인 (main.py 등 시작점에서 수행 권장)
LOG_DIR = os.path.join(PROJECT_ROOT, "app", "log")
os.makedirs(LOG_DIR, exist_ok=True)
# 로깅 설정
setup_logging(YAML_PATH) # 앱 시작 시 한 번만 설정 권장
logger = get_logger("FastAPIApp")

# --- Lifespan 컨텍스트 매니저 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱 시작/종료 시 리소스 관리 (모든 서비스/도구/노드 초기화).
    """
    logger.info("애플리케이션 시작... 리소스 초기화 시작.")
    app.state.compiled_workflow_app = None  # 초기 상태는 None
    app.state.initialization_error = None  # 초기화 오류 상태 초기화

    # --- 2. 서비스 및 도구 인스턴스화 (이전 코드와 동일하게 진행) ---
    logger.info("서비스/도구 인스턴스화 시작...")
    # setup_logging('logging_config.yaml') # 여기서 로깅 설정 적용
    # --- 1. 공유 리소스 생성 ---
    # 공유 aiohttp 세션 (여러 도구에서 사용 가능)
    app.state.shared_http_session = aiohttp.ClientSession()
    # DB 클라이언트 (여러 노드에서 사용 가능)
    app.state.db_client = DatabaseClientV2()
    # 정리 필요한 리소스 목록
    closable_resources = [app.state.shared_http_session, app.state.db_client]
    logger.info("공유 리소스 (HTTP 세션, DB 클라이언트) 생성됨.")

    # --- 2. 서비스 및 도구 인스턴스화 ---
    logger.info("서비스/도구 인스턴스화 시작...")
    # 각 도구/서비스 생성 시 필요 시 세션/DB 클라이언트 전달
    llm_service = LLMService() # closable_resources.append(llm_service) # close 메서드 없으면 추가 안 함
    Google_Search_tool = GoogleSearchTool(session=app.state.shared_http_session) # 세션 전달
    # google_tool의 close 메서드 유무 확인 후 추가
    # if hasattr(Google_Search_tool, 'close'): closable_resources.append(Google_Search_tool)
    naver_tool = NaverSearchTool()
    rss_tool = RssSearchTool()
    twitter_tool = TwitterTool()
    reddit_tool = RedditTool()
    article_scraper_tool = ArticleScraperTool(session=app.state.shared_http_session) # 세션 전달
    if hasattr(article_scraper_tool, 'close'): closable_resources.append(article_scraper_tool)
    selenium_tool = SeleniumScraperTool(); closable_resources.append(selenium_tool) # close 필요
    lang_tool = LanguageDetectionTool()
    spam_service = SpamDetectionService()
    cluster_tool = TextClusteringTool()
    google_trends_tool = GoogleTrendsTool() # 세션 필요 여부 확인
    twitter_counts_tool = TwitterCountsTool() # API 키 등 필요 여부 확인

    # 추가된 서비스
    image_client = ImageGenerationClient() # close 필요 여부 확인
    translator_client = PapagoTranslationService() # close 필요 여부 확인 (API 키 로딩 등)
    storage_client = StorageClient() # Optional, close 필요 여부 확인
    # if image_client and hasattr(image_client, 'close'): closable_resources.append(image_client)
    # if translator_client and hasattr(translator_client, 'close'): closable_resources.append(translator_client)
    # if storage_client and hasattr(storage_client, 'close'): closable_resources.append(storage_client)

    logger.info("서비스/도구 인스턴스화 완료.")

    # --- 3. 모든 노드 인스턴스화 (의존성 주입) ---
    logger.info("전체 노드 인스턴스화 시작...")
    try:
        node01 = n01_initialize_node.InitializeNode()
        node02 = n02_topic_analyzer_node.TopicAnalyzerNode(llm_client=llm_service, db_client=app.state.db_client)
        # *** TypeError 수정: 키워드 인자 이름 변경 ***
        node03 = n03_news_collector_node.NewsCollectorNode(
            Google_Search_tool=Google_Search_tool, # 'Google' -> 'Google_Search_tool'
            naver_search_tool=naver_tool,         # 'naver' -> 'naver_search_tool'
            rss_search_tool=rss_tool             # 'rss' -> 'rss_search_tool'
        )
        node04 = n04_opinion_collector_node.OpinionCollectorNode(
            twitter_tool=twitter_tool, reddit_tool=reddit_tool,
            Google_Search_tool=Google_Search_tool # 'google_tool' -> 'Google_Search_tool' 일관성
        )
        node05 = n05_news_scraper_node.NewsScraperNode(scraper_tool=article_scraper_tool)
        node06 = n06_opinion_scraper_node.OpinionScraperNode(
            twitter_tool=twitter_tool, reddit_tool=reddit_tool,
            Google_Search_tool=Google_Search_tool, # 'google_tool' -> 'Google_Search_tool' 일관성
            selenium_tool=selenium_tool
        )
        node07 = n07_filter_node.FilterNode(
            language_tool=lang_tool, spam_service=spam_service, clustering_tool=cluster_tool
        )
        node08 = n08_news_summarizer_node.NewsSummarizerNode(llm_client=llm_service)
        node09 = n09_opinion_summarizer_node.OpinionSummarizerNode(llm_client=llm_service)
        node10 = n10_synthesis_summarizer_node.SynthesisSummarizerNode(llm_client=llm_service)
        node11 = n11_evaluate_summary_node.EvaluateSummaryNode()
        node12 = n12_trend_analyzer_node.TrendAnalyzerNode(
            google_trends_tool=google_trends_tool, twitter_counts_tool=twitter_counts_tool
        )
        node13 = n13_progress_report_node.ProgressReportNode()
        node14 = n14_idea_generator_node.IdeaGeneratorNode(llm_client=llm_service)
        node15 = n15_scenario_writer_node.ScenarioWriterNode(llm_client=llm_service)
        node16 = n16_scenario_report_node.ScenarioReportNode(llm_client=llm_service) # LLM 평가 시 필요
        node17 = n17_imager_node.ImagerNode(image_client=image_client)
        node18 = n18_translator_node.TranslatorNode(translator_client=translator_client)
        node19 = n19_postprocessor_node.PostprocessorNode(storage_client=storage_client) # Storage 클라이언트 전달
        logger.info("전체 노드 인스턴스화 완료.")
        # 1. 그래프 구조 빌드 (컴파일되지 않은 StateGraph 객체 반환)
        logger.info("Building workflow graph structure...")
        graph_structure: StateGraph = build_main_workflow(
            node01, node02, node03, node04, node05, node06, node07, node08, node09,
            node10, node11, node12, node13, node14, node15, node16, node17, node18, node19
        )
        logger.info("Workflow graph structure built.")

        # --- 2. 체크포인터 설정 (수정됨) ---
        checkpointer_instance = None  # 컴파일 시 사용할 체크포인터 변수
        if settings.REDIS_URL:
            try:
                logger.info(f"Configuring Redis checkpointer using URL: {settings.REDIS_URL}")

                # --- async with 사용하여 AsyncRedisSaver 인스턴스 얻기 ---
                async with AsyncRedisSaver.from_conn_string(settings.REDIS_URL) as cp:
                    checkpointer_instance = cp  # 얻어진 인스턴스를 변수에 할당
                    logger.info("AsyncRedisSaver checkpointer configured via from_conn_string.")
                # async with 블록을 벗어나면 컨텍스트 관리자의 __aexit__가 호출됨 (자동 정리 시작)
                # 하지만 checkpointer_instance는 컴파일에 사용 가능

            except ImportError:
                logger.warning(
                    "AsyncRedisSaver or redis library not available. Checkpointer disabled. Install 'langgraph[redis]'.")
                checkpointer_instance = None
            except Exception as cp_err:
                logger.error(f"Failed to configure Redis checkpointer using from_conn_string: {cp_err}", exc_info=True)
                checkpointer_instance = None  # 오류 발생 시 None으로 설정
        else:
            logger.warning("REDIS_URL not set. Proceeding without checkpointer.")
        # --- 체크포인터 설정 끝 ---

        # --- 3. 그래프 컴파일 ---
        logger.info("Compiling the workflow graph...")
        # checkpointer_instance (None일 수 있음)를 compile 메서드에 전달
        compiled_graph_object = graph_structure.compile(checkpointer=checkpointer_instance)
        logger.info("Workflow graph compilation step finished.")

        # 4. 컴파일된 그래프 객체를 app.state에 저장
        app.state.compiled_workflow_app = compiled_graph_object

        # 컴파일 성공 확인 및 로깅
        if app.state.compiled_workflow_app:
            logger.info("Compiled workflow graph assigned to app.state.")
        else:
            # 컴파일 실패 시 (이론적으로 compile에서 예외 발생 가능성 높음)
            raise ValueError("Workflow compilation failed or returned None.")


        # 4. 컴파일된 그래프 객체를 app.state에 저장
        app.state.compiled_workflow_app = compiled_graph_object
        # --- 체크포인터 설정 끝 ---

        # 컴파일 성공 확인
        if app.state.compiled_workflow_app:
            logger.info("Compiled workflow graph assigned to app.state.")
        else:
            # .compile()이 실패하거나 None을 반환하는 경우 (이론상 예외 발생)
            raise ValueError("Workflow compilation failed or returned None.")

    except Exception as build_err:
        logger.critical(f"Error during workflow build/compile: {build_err}", exc_info=True)
        app.state.compiled_workflow_app = None  # 오류 시 None 설정

    # --- 앱 준비 상태 확인 ---
    # getattr 사용으로 안전하게 확인
    if getattr(app.state, 'compiled_workflow_app', None) is not None:
        logger.info("Application startup complete. Workflow ready.")
    else:
        logger.error("Application startup completed BUT workflow is unavailable due to build/compile errors.")
        if not hasattr(app.state, 'initialization_error'):
            app.state.initialization_error = "Workflow compilation failed during startup."

    yield  # 애플리케이션 실행
    # --- 앱 종료 시 리소스 정리 ---
    logger.info("애플리케이션 종료... 리소스 정리 시작.")

    for resource in reversed(closable_resources): # 생성 역순으로 정리
         try:
              resource_name = type(resource).__name__
              if hasattr(resource, 'close') and callable(resource.close):
                   if asyncio.iscoroutinefunction(resource.close):
                        logger.debug(f"Closing resource (async): {resource_name}")
                        await resource.close()
                        logger.info(f"리소스 정리됨 (async): {resource_name}")
                   else:
                        logger.debug(f"Closing resource (sync): {resource_name}")
                        resource.close()
                        logger.info(f"리소스 정리됨 (sync): {resource_name}")
              # Selenium 특별 처리
              elif isinstance(resource, SeleniumScraperTool):
                   logger.debug(f"Closing resource (Selenium): {resource_name}")
                   await resource.close(trace_id="shutdown", comic_id="shutdown")
                   logger.info(f"리소스 정리됨 (Selenium): {resource_name}")
         except Exception as close_err:
              logger.error(f"{resource_name} 리소스 정리 중 오류: {close_err}", exc_info=True)
    logger.info("리소스 정리 완료.")


# --- FastAPI 앱 생성 ---
app = FastAPI(
    title="Comic Generation API",
    description="LangGraph 기반 4컷 만화 생성 워크플로우 API",
    version="0.1.6", # 설정에서 버전 가져오기
    lifespan=lifespan
)

app.include_router(v1_endpoints.router)
logger.info("Included API router from app.api.v1.endpoints")
# # --- API 라우터 정의 ---
# @app.post("/v1/comics", status_code=status.HTTP_202_ACCEPTED, tags=["Comic Generation"])
# async def generate_comic_endpoint(
#     query: str,
#     background_tasks: BackgroundTasks,
#     # 의존성 주입으로 컴파일된 앱과 DB 클라이언트 가져오기
#     compiled_app: StateGraph = Depends(get_compiled_workflow_app),
#     db_client: DatabaseClientV2 = Depends(get_db_client)
# ):
#     """
#     만화 생성을 위한 워크플로우를 백그라운드에서 실행합니다.
#
#     - **query**: 만화의 주제가 될 사용자 입력 쿼리 문자열.
#     """
#     # 앱 초기화 오류 확인
#     if hasattr(app.state, 'initialization_error') and app.state.initialization_error:
#          raise HTTPException(
#              status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#              detail=f"Workflow service not available due to initialization error: {app.state.initialization_error}"
#          )
#     if not compiled_app:
#         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Workflow is not compiled.")
#
#     logger.info(f"API Request: Generate Comic for query='{query}'")
#     try:
#         # trigger_workflow_task 함수가 comic_id를 반환한다고 가정
#         comic_id = await trigger_workflow_task(query, background_tasks, compiled_app, db_client)
#         logger.info(f"Background task scheduled. Comic ID: {comic_id}")
#         return {"message": "Comic generation task started.", "comic_id": comic_id}
#     except Exception as e:
#         logger.exception("Failed to trigger workflow task.", exc_info=e)
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to start comic generation task.")
#
# @app.get("/v1/comics/status/{comic_id}", tags=["Comic Generation"])
# async def get_comic_status_endpoint(
#     comic_id: str,
#     db_client: DatabaseClientV2 = Depends(get_db_client)
# ):
#     """
#     주어진 comic_id에 해당하는 만화 생성 작업의 상태를 조회합니다.
#
#     - **comic_id**: 조회할 작업의 고유 ID.
#     """
#     logger.info(f"API Request: Get status for comic_id='{comic_id}'")
#     # DB에서 상태 조회 (DatabaseClientV2에 get_status 와 같은 메서드 필요 가정)
#     # status_data = await db_client.get_status(comic_id) # 예시
#     # 임시로 Redis get 사용 (실제로는 더 구조화된 데이터 필요)
#     status_data = await db_client.get(f"comic_status::{comic_id}") # 예시 키
#
#     if status_data:
#         # status_data가 JSON 문자열이면 파싱 필요할 수 있음
#         try:
#             import json
#             return json.loads(status_data) if isinstance(status_data, str) else status_data
#         except json.JSONDecodeError:
#              return {"status": "unknown", "detail": "Failed to parse status data."}
#     else:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comic generation task status not found.")
#
# # --- Root Endpoint ---
# @app.get("/", tags=["General"])
# async def read_root():
#     """API 루트 엔드포인트."""
#     return {"message": f"{settings.APP_NAME} running. Version: {settings.APP_VERSION}"}


# --- 메인 실행 블록 (로컬 개발용) ---
if __name__ == "__main__":
    import uvicorn
    # 로깅 설정은 lifespan에서 수행
    logger.info("Starting Uvicorn server for local development...")
    # host="127.0.0.1" 로 변경하여 로컬에서만 접근하도록 할 수 있음
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug", reload=True) # DEBUG 모드 설정 반영