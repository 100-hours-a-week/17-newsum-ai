# app/workflows/main_workflow.py

import asyncio
import logging
import uuid
import aiohttp # aiohttp 세션 관리를 위해 추가

# --- LangGraph 및 애플리케이션 구성 요소 임포트 ---
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver # 필요 시 체크포인터 사용

# 설정 및 로거
from app.config.settings import settings
from app.utils.logger import setup_logging, get_logger

# 상태 및 노드 클래스
from app.workflows.state import ComicState
from app.nodes.n01_initialize_node import InitializeNode
from app.nodes.n02_topic_analyzer_node import TopicAnalyzerNode
from app.nodes.n03_news_collector_node import NewsCollectorNode
from app.nodes.n04_opinion_collector_node import OpinionCollectorNode
from app.nodes.n05_news_scraper_node import NewsScraperNode
from app.nodes.n06_opinion_scraper_node import OpinionScraperNode
from app.nodes.n07_filter_node import FilterNode
# --- (이후 노드는 주석 처리) ---

# 서비스 및 도구 클래스
# Services
from app.services.llm_server_client_v2 import LLMService
from app.services.database_con_client_v2 import DatabaseClientV2
# Tools - 각 도구 파일에서 해당 클래스를 정확히 임포트해야 함
from app.tools.search.google import GoogleSearchTool
from app.tools.search.naver import NaverSearchTool
from app.tools.search.rss import RssSearchTool
from app.tools.social.twitter import TwitterTool
from app.tools.social.reddit import RedditTool
from app.tools.scraping.article_scraper import ArticleScraperTool
from app.tools.scraping.selenium_scraper import SeleniumScraperTool
from app.tools.analysis.language_detector import LanguageDetectionTool
from app.services.spam_detector import SpamDetectionService # Service 또는 Tool 분류 확인
from app.tools.analysis.text_clusterer import TextClusteringTool

# --- 로깅 설정 ---
# setup_logging('logging_config.yaml') # 앱 진입점에서 한 번만 호출 권장
logger = get_logger("MainWorkflow")

# --- 워크플로우 빌드 함수 ---
def build_main_workflow(
    # 노드 01 ~ 07 인스턴스를 인자로 받음
    initialize_node: InitializeNode,
    topic_analyzer_node: TopicAnalyzerNode,
    news_collector_node: NewsCollectorNode,
    opinion_collector_node: OpinionCollectorNode,
    news_scraper_node: NewsScraperNode,
    opinion_scraper_node: OpinionScraperNode,
    filter_node: FilterNode,
    # --- (이후 노드 인스턴스도 필요 시 인자로 추가) ---
) -> StateGraph:
    """
    노드 01부터 07까지 실행되는 LangGraph StateGraph를 빌드합니다.

    Args:
        initialize_node: InitializeNode 인스턴스.
        topic_analyzer_node: TopicAnalyzerNode 인스턴스 (의존성 주입 완료 상태).
        news_collector_node: NewsCollectorNode 인스턴스 (의존성 주입 완료 상태).
        opinion_collector_node: OpinionCollectorNode 인스턴스 (의존성 주입 완료 상태).
        news_scraper_node: NewsScraperNode 인스턴스 (의존성 주입 완료 상태).
        opinion_scraper_node: OpinionScraperNode 인스턴스 (의존성 주입 완료 상태).
        filter_node: FilterNode 인스턴스 (의존성 주입 완료 상태).

    Returns:
        StateGraph: 컴파일된 LangGraph 애플리케이션 (실행 가능 객체).
    """
    logger.info("워크플로우 그래프 정의 시작 (01-07 단계)...")
    graph = StateGraph(ComicState) # 상태 클래스로 그래프 초기화

    # 노드 추가
    graph.add_node("initialize", initialize_node.run)
    graph.add_node("topic_analyzer", topic_analyzer_node.run)
    graph.add_node("news_collector", news_collector_node.run)
    graph.add_node("opinion_collector", opinion_collector_node.execute)
    graph.add_node("news_scraper", news_scraper_node.execute)
    graph.add_node("opinion_scraper", opinion_scraper_node.execute)
    graph.add_node("filter_opinions", filter_node.execute) # 노드 07

    # 엣지(연결) 정의
    graph.set_entry_point("initialize") # 시작점 설정
    graph.add_edge("initialize", "topic_analyzer") # 01 -> 02

    # 토픽 분석 후 뉴스/의견 수집 병렬 실행
    graph.add_edge("topic_analyzer", "news_collector") # 02 -> 03
    graph.add_edge("topic_analyzer", "opinion_collector") # 02 -> 04 (병렬)

    # 각 수집 후 해당 스크래퍼 실행
    graph.add_edge("news_collector", "news_scraper") # 03 -> 05
    graph.add_edge("opinion_collector", "opinion_scraper") # 04 -> 06

    # 뉴스 스크래퍼 이후 일단 종료 (07까지만 실행)
    graph.add_edge("news_scraper", END) # 05 -> END

    # 의견 스크래퍼 이후 필터 노드 실행
    graph.add_edge("opinion_scraper", "filter_opinions") # 06 -> 07

    # 필터 노드 이후 일단 종료 (07까지만 실행)
    graph.add_edge("filter_opinions", END) # 07 -> END

    logger.info("워크플로우 그래프 정의 완료 (01-07 단계).")

    # --- 그래프 컴파일 ---
    # 체크포인터 없이 컴파일 (상태 추적 안 함 - DB 등으로 별도 관리 가정)
    compiled_graph = graph.compile()
    logger.info("워크플로우 그래프 컴파일 완료.")

    # (선택 사항) 그래프 시각화 코드
    try:
        graph_png = compiled_graph.get_graph().draw_mermaid_png()
        if graph_png:
            with open("workflow_graph_01_07.png", "wb") as f:
                f.write(graph_png)
            logger.info("워크플로우 그래프 이미지 저장됨: workflow_graph_01_07.png")
    except Exception as viz_err:
        logger.warning(f"워크플로우 그래프 이미지 생성 실패: {viz_err}")

    return compiled_graph # 컴파일된 그래프 반환


# --- 워크플로우 실행 및 리소스 관리 함수 ---
async def run_workflow(initial_query: str):
    """
    워크플로우 실행을 위한 서비스/도구/노드 초기화 및 실행, 리소스 정리를 수행합니다.
    """
    main_logger = get_logger("WorkflowRunner") # 실행용 로거
    main_logger.info(f"워크플로우 실행 요청 수신: query='{initial_query}'")

    # --- 리소스 초기화 ---
    shared_session = None
    selenium_tool_instance = None
    db_client_instance = None
    compiled_app = None
    # 관리해야 할 도구/서비스 목록 (close 호출 위해)
    closable_resources = []

    try:
        # --- 1. 공통 리소스 생성 ---
        shared_session = aiohttp.ClientSession()
        closable_resources.append(shared_session) # 정리 목록에 추가
        main_logger.info("공유 aiohttp 세션 생성됨.")

        # --- 2. 서비스 및 도구 인스턴스화 ---
        main_logger.info("서비스/도구 인스턴스화 시작...")
        # 각 도구/서비스 생성 시 필요 시 세션 전달
        llm_service = LLMService(); closable_resources.append(llm_service)
        db_client_instance = DatabaseClientV2(); closable_resources.append(db_client_instance) # DB 클라이언트 인스턴스 저장
        google_tool = GoogleSearchTool(session=shared_session); closable_resources.append(google_tool)
        naver_tool = NaverSearchTool() # 세션 사용 안 함
        rss_tool = RssSearchTool()
        twitter_tool = TwitterTool() # closable_resources.append(twitter_tool) # close 메서드 없으면 추가 안 함
        reddit_tool = RedditTool() # closable_resources.append(reddit_tool) # close 메서드 없으면 추가 안 함
        article_scraper = ArticleScraperTool(session=shared_session); closable_resources.append(article_scraper)
        selenium_tool_instance = SeleniumScraperTool(); closable_resources.append(selenium_tool_instance) # Selenium 도구 인스턴스 저장
        lang_tool = LanguageDetectionTool()
        spam_service = SpamDetectionService() # closable_resources.append(spam_service) # close 메서드 없으면 추가 안 함
        cluster_tool = TextClusteringTool()
        main_logger.info("서비스/도구 인스턴스화 완료.")

        # --- 3. 노드 인스턴스화 (의존성 주입) ---
        main_logger.info("노드 인스턴스화 시작...")
        node01 = InitializeNode()
        # DB 클라이언트 인스턴스 전달 확인
        node02 = TopicAnalyzerNode(llm_client=llm_service, db_client=db_client_instance)
        # News Collector 도구 전달 확인
        node03 = NewsCollectorNode(Google=google_tool, naver=naver_tool, rss=rss_tool)
        # Opinion Collector 도구 전달 확인 (GoogleSearchTool 전달)
        node04 = OpinionCollectorNode(twitter_tool=twitter_tool, reddit_tool=reddit_tool, google_tool=google_tool)
        # News Scraper 도구 전달 확인
        node05 = NewsScraperNode(scraper_tool=article_scraper)
        # Opinion Scraper 도구 전달 확인 (google_tool, selenium_tool_instance 전달)
        node06 = OpinionScraperNode(twitter_tool=twitter_tool, reddit_tool=reddit_tool, google_tool=google_tool, selenium_tool=selenium_tool_instance)
        # Filter Node 도구 전달 확인
        node07 = FilterNode(language_tool=lang_tool, spam_service=spam_service, clustering_tool=cluster_tool)
        main_logger.info("노드 인스턴스화 완료.")

        # --- 4. 워크플로우 빌드 ---
        compiled_app = build_main_workflow(
            initialize_node=node01, topic_analyzer_node=node02, news_collector_node=node03,
            opinion_collector_node=node04, news_scraper_node=node05, opinion_scraper_node=node06,
            filter_node=node07
        )

        # --- 5. 워크플로우 실행 ---
        main_logger.info("워크플로우 실행 시작...")
        thread_id = str(uuid.uuid4()) # 실행 고유 ID 생성
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"initial_query": initial_query} # 입력 설정
        final_state_dict = None # 최종 상태 저장 변수

        # astream_events를 사용하여 이벤트 기반 처리 (더 많은 정보 로깅 가능)
        async for event in compiled_app.astream_events(inputs, config=config, version="v1"):
            kind = event["event"]
            name = event["name"]
            # 필요한 이벤트 로깅 (예: 노드 시작/종료)
            if kind == "on_chain_start":
                main_logger.debug(f"노드 시작: {name}", extra={'trace_id': thread_id, 'comic_id': thread_id}) # comic_id는 아직 없을 수 있음
            elif kind == "on_chain_end":
                main_logger.debug(f"노드 종료: {name}", extra={'trace_id': thread_id, 'comic_id': thread_id})
                # 마지막 이벤트에서 최종 상태 추출 시도
                output = event["data"].get("output")
                if isinstance(output, dict) and "__end__" in output:
                    final_state_dict = {k: v for k, v in output.items() if k != '__end__'}
                    # LangGraph 구조에 따라 상태가 중첩될 수 있음
                    if len(final_state_dict) == 1:
                           possible_state_key = list(final_state_dict.keys())[0]
                           if isinstance(final_state_dict[possible_state_key], ComicState):
                                final_state_dict = final_state_dict[possible_state_key].model_dump() # pydantic v2
                           elif isinstance(final_state_dict[possible_state_key], dict): # dict로 반환될 수도 있음
                                final_state_dict = final_state_dict[possible_state_key]
                elif isinstance(output, ComicState):
                    final_state_dict = output.model_dump() # pydantic v2

        main_logger.info("워크플로우 실행 완료.", extra={'trace_id': thread_id})

        # 최종 상태 확인 (출력 또는 DB 저장 등)
        if final_state_dict:
            final_state_obj = ComicState(**final_state_dict) # 상태 객체로 변환
            main_logger.info(f"최종 상태 (일부): Comic ID={final_state_obj.comic_id}, Opinions Cleaned={len(final_state_obj.opinions_clean)}, Error='{final_state_obj.error_message}'", extra={'trace_id': final_state_obj.trace_id or thread_id})
            # TODO: 최종 상태를 DB에 업데이트하는 로직 추가 가능 (background_tasks.py 참고)
        else:
            main_logger.warning("최종 상태를 가져오지 못했습니다.", extra={'trace_id': thread_id})

    except Exception as e:
        # 실행 중 발생한 예외 로깅
        main_logger.critical(f"워크플로우 실행 중 오류 발생: {e}", exc_info=True)
        # TODO: 오류 발생 시 DB 상태 업데이트 로직 추가 가능
    finally:
        # --- 6. 리소스 정리 ---
        main_logger.info("리소스 정리 시작...")
        for resource in reversed(closable_resources): # 생성 역순으로 정리 시도
             try:
                  if hasattr(resource, 'close') and callable(resource.close):
                       # close가 비동기 함수인지 확인
                       if asyncio.iscoroutinefunction(resource.close):
                            await resource.close() # 비동기 close 호출
                            main_logger.info(f"리소스 정리됨 (async): {type(resource).__name__}")
                       else:
                            resource.close() # 동기 close 호출
                            main_logger.info(f"리소스 정리됨 (sync): {type(resource).__name__}")
                  # SeleniumScraperTool의 특별 처리 (trace_id, comic_id 필요 시)
                  elif isinstance(resource, SeleniumScraperTool):
                      # 임시 ID 사용하여 종료 (실제 ID는 없을 수 있음)
                      await resource.close(trace_id="cleanup", comic_id="cleanup")
                      main_logger.info(f"리소스 정리됨 (Selenium): {type(resource).__name__}")

             except Exception as close_err:
                  main_logger.error(f"{type(resource).__name__} 리소스 정리 중 오류 발생: {close_err}", exc_info=True)
        main_logger.info("리소스 정리 완료.")

# --- 스크립트 직접 실행 시 예시 ---
# if __name__ == "__main__":
#     # 로깅 설정 (앱 진입점에서 수행 권장)
#     setup_logging('logging_config.yaml')
#
#     # 테스트 쿼리
#     test_query = "최근 인공지능 기술 동향에 대해 알려줘"
#
#     # 비동기 워크플로우 실행
#     try:
#         asyncio.run(run_workflow(test_query))
#     except KeyboardInterrupt:
#         logger.info("사용자에 의해 실행 중단됨.")
#     except Exception as e:
#          logger.critical(f"메인 실행 중 치명적 오류 발생: {e}", exc_info=True)