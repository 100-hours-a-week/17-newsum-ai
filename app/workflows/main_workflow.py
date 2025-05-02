# app/workflows/main_workflow.py (진입점 및 Redis 체크포인트 적용 수정본 - 전체 코드)

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import aiohttp # aiohttp 세션 관리

# LangGraph 및 상태/노드 임포트
from langgraph.graph import StateGraph, END
# --- 수정: AsyncRedisSaver 임포트 ---
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
import redis.asyncio as redis

from app.config.settings import settings
from app.utils.logger import setup_logging, get_logger
from app.workflows.state import ComicState

# --- 모든 노드 클래스 임포트 ---
from app.nodes.n01_initialize_node import InitializeNode
from app.nodes.n02_topic_analyzer_node import TopicAnalyzerNode
from app.nodes.n03_news_collector_node import NewsCollectorNode
from app.nodes.n04_opinion_collector_node import OpinionCollectorNode
from app.nodes.n05_news_scraper_node import NewsScraperNode
from app.nodes.n06_opinion_scraper_node import OpinionScraperNode
from app.nodes.n07_filter_node import FilterNode
from app.nodes.n08_news_summarizer_node import NewsSummarizerNode
from app.nodes.n09_opinion_summarizer_node import OpinionSummarizerNode
from app.nodes.n10_synthesis_summarizer_node import SynthesisSummarizerNode
from app.nodes.n11_evaluate_summary_node import EvaluateSummaryNode
from app.nodes.n12_trend_analyzer_node import TrendAnalyzerNode
from app.nodes.n13_progress_report_node import ProgressReportNode
from app.nodes.n14_idea_generator_node import IdeaGeneratorNode
from app.nodes.n15_scenario_writer_node import ScenarioWriterNode
from app.nodes.n16_scenario_report_node import ScenarioReportNode
from app.nodes.n17_imager_node import ImagerNode
from app.nodes.n18_translator_node import TranslatorNode
from app.nodes.n19_postprocessor_node import PostprocessorNode

# --- 모든 서비스 및 도구 클래스 임포트 ---
# Services
from app.services.llm_server_client_v2 import LLMService
from app.services.database_con_client_v2 import DatabaseClientV2
from app.services.image_server_client_v2 import ImageGenerationClient
from app.services.papago_translation_service import PapagoTranslationService
from app.services.storage_client_v2 import StorageClient
from app.services.spam_detector import SpamDetectionService # 경로 및 타입 확인

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
from app.tools.trends.google_trends import GoogleTrendsTool
from app.tools.trends.twitter_counts import TwitterCountsTool

# 로깅 설정 (앱 진입점에서 호출 권장)
# setup_logging('logging_config.yaml')
logger = get_logger("MainWorkflowBuilder")

# --- 조건부 엣지 함수 ---
def should_continue_based_on_evaluation(state: ComicState) -> str:
    """
    요약 평가 결과('decision' 상태)에 따라 다음 단계를 결정합니다.
    """
    decision = state.decision
    trace_id = state.trace_id
    log_prefix = f"[{trace_id}]" if trace_id else ""

    logger.info(f"{log_prefix} Evaluating decision: '{decision}'")
    if decision == "research_again":
        logger.warning(f"{log_prefix} Branching: research_again. Workflow likely ending here.")
        # 'research_again' 시나리오는 현재 그래프에서 명시적으로 처리하기보다,
        # 워크플로우 외부에서 재실행을 유도하거나 상태를 리셋하는 방식으로 처리하는 것이 일반적일 수 있음.
        # 여기서는 일단 종료(END)로 연결하거나, 특정 재시작 노드(미구현)로 보낼 수 있음.
        return "end_workflow" # 또는 "trigger_research_node" 등
    elif decision == "refine_topic":
        logger.warning(f"{log_prefix} Branching: refine_topic. Suggests re-running topic analysis.")
        # 토픽 재정의는 현재 그래프 내에서 직접 처리하기 어려움. 외부 재실행 필요.
        return "end_workflow_with_refine_suggestion" # 또는 "trigger_refine_node" 등
    elif decision == "proceed":
        logger.info(f"{log_prefix} Branching: proceed to creative generation.")
        return "generate_ideas" # 아이디어 생성 단계로 진행
    else: # 'proceed'가 아니거나 decision이 없는 경우 (기본값 또는 오류)
        logger.warning(f"{log_prefix} Unknown or missing decision ('{decision}'). Proceeding as default.")
        return "generate_ideas" # 기본적으로 아이디어 생성으로 진행

# --- 워크플로우 빌드 함수 ---
def build_main_workflow() -> StateGraph:
    """
    전체 19개 노드를 포함하는 LangGraph StateGraph를 빌드하고 Redis 체크포인터를 설정합니다.
    """
    logger.info("Building the full comic generation workflow graph...")
    graph = StateGraph(ComicState)

    # --- 서비스/도구 인스턴스화 (실제 사용 시 DI 프레임워크 고려) ---
    # 참고: 각 서비스/도구의 초기화 방식(세션 필요 여부 등) 확인 필요
    #      여기서는 간단히 기본 생성자 호출 가정
    llm_service = LLMService()
    db_client = DatabaseClientV2()
    image_client = ImageGenerationClient()
    translator_client = PapagoTranslationService()
    storage_client = StorageClient() # Optional, S3 사용 시 필요
    spam_service = SpamDetectionService()
    Google_Search_tool = GoogleSearchTool() # session은 내부에서 관리하거나 외부 주입 필요
    naver_search_tool = NaverSearchTool()
    rss_search_tool = RssSearchTool()
    twitter_tool = TwitterTool()
    reddit_tool = RedditTool()
    article_scraper = ArticleScraperTool() # session 필요 시 주입
    selenium_tool = SeleniumScraperTool()
    lang_tool = LanguageDetectionTool()
    cluster_tool = TextClusteringTool()
    google_trends_tool = GoogleTrendsTool()
    twitter_counts_tool = TwitterCountsTool()
    logger.info("All services and tools instantiated.")

    # --- 노드 인스턴스화 (의존성 주입) ---
    node01 = InitializeNode()
    node02 = TopicAnalyzerNode(llm_client=llm_service, db_client=db_client)
    node03 = NewsCollectorNode(Google_Search_tool=Google_Search_tool, naver_search_tool=naver_search_tool, rss_search_tool=rss_search_tool)
    node04 = OpinionCollectorNode(twitter_tool=twitter_tool, reddit_tool=reddit_tool, Google_Search_tool=Google_Search_tool)
    node05 = NewsScraperNode(scraper_tool=article_scraper)
    node06 = OpinionScraperNode(twitter_tool=twitter_tool, reddit_tool=reddit_tool, Google_Search_tool=Google_Search_tool, selenium_tool=selenium_tool)
    node07 = FilterNode(language_tool=lang_tool, spam_service=spam_service, clustering_tool=cluster_tool)
    node08 = NewsSummarizerNode(llm_client=llm_service)
    node09 = OpinionSummarizerNode(llm_client=llm_service)
    node10 = SynthesisSummarizerNode(llm_client=llm_service)
    node11 = EvaluateSummaryNode() # 내부적으로 rouge/bert 사용
    node12 = TrendAnalyzerNode(google_trends_tool=google_trends_tool, twitter_counts_tool=twitter_counts_tool)
    node13 = ProgressReportNode() # Jinja2 내부 사용
    node14 = IdeaGeneratorNode(llm_client=llm_service)
    node15 = ScenarioWriterNode(llm_client=llm_service)
    node16 = ScenarioReportNode(llm_client=llm_service) # LLM 클라이언트 선택적 전달
    node17 = ImagerNode(image_client=image_client)
    node18 = TranslatorNode(translator_client=translator_client)
    node19 = PostprocessorNode(storage_client=storage_client) # Storage 클라이언트 선택적 전달
    logger.info("All nodes instantiated.")

    # --- 그래프 노드 추가 ---
    graph.add_node("initialize", node01.run)
    graph.add_node("analyze_topic", node02.run)
    graph.add_node("collect_news", node03.run)
    graph.add_node("collect_opinions", node04.run)
    graph.add_node("scrape_news", node05.run)
    graph.add_node("scrape_opinions", node06.run)
    graph.add_node("filter_opinions", node07.run)
    graph.add_node("summarize_news", node08.run)
    graph.add_node("summarize_opinions", node09.run)
    graph.add_node("synthesize_summary", node10.run)
    graph.add_node("evaluate_summary", node11.run)
    graph.add_node("analyze_trends", node12.run)
    graph.add_node("report_progress", node13.run)
    graph.add_node("generate_ideas", node14.run)
    graph.add_node("write_scenario", node15.run)
    graph.add_node("report_scenario", node16.run)
    graph.add_node("generate_images", node17.run)
    graph.add_node("translate_dialogue", node18.run)
    graph.add_node("postprocess_comic", node19.run)
    logger.info("All nodes added to the graph.")

    # --- !!! 그래프 진입점 설정 (추가) !!! ---
    graph.set_entry_point("initialize")
    logger.info("Graph entry point set to 'initialize'.")

    # --- 그래프 엣지 정의 ---
    graph.add_edge("initialize", "analyze_topic")

    # Topic Analysis 후 병렬 실행
    graph.add_edge("analyze_topic", "collect_news")
    graph.add_edge("analyze_topic", "collect_opinions")
    graph.add_edge("analyze_topic", "analyze_trends") # 트렌드 분석도 병렬 시작

    # 데이터 수집 후 스크래핑
    graph.add_edge("collect_news", "scrape_news")
    graph.add_edge("collect_opinions", "scrape_opinions")

    # 스크래핑 후 요약
    graph.add_edge("scrape_news", "summarize_news")
    graph.add_edge("scrape_opinions", "filter_opinions") # 의견은 필터링 먼저
    graph.add_edge("filter_opinions", "summarize_opinions")

    # 양쪽 요약 완료 후 통합 (주의: 병렬 작업 완료 동기화 필요)
    # LangGraph는 기본적으로 모든 입력 엣지가 충족되어야 노드 실행
    graph.add_edge("summarize_news", "synthesize_summary")
    graph.add_edge("summarize_opinions", "synthesize_summary")

    # 통합 요약 후 평가 및 진행 보고
    graph.add_edge("synthesize_summary", "evaluate_summary")
    # 트렌드 분석은 통합 요약과 병렬로 진행되다가 평가 전에 완료되어야 함
    # (평가 노드가 트렌드 점수를 사용하지 않으므로 직접 연결은 불필요)
    # 단, 보고서에는 트렌드 점수가 필요하므로, 보고서 노드는 트렌드 분석 이후 실행
    graph.add_edge("analyze_trends", "report_progress") # 트렌드 분석 후 보고서
    graph.add_edge("evaluate_summary", "report_progress") # 평가 후 보고서

    # 평가 결과에 따른 분기 (Conditional Edge)
    graph.add_conditional_edges(
        "evaluate_summary", # 평가 노드 이후 분기
        should_continue_based_on_evaluation, # 상태 기반 결정 함수
        {
            "generate_ideas": "generate_ideas", # 아이디어 생성으로 진행
            "end_workflow": END, # 워크플로우 종료 (재탐색 필요 시)
            "end_workflow_with_refine_suggestion": END, # 워크플로우 종료 (토픽 재정의 필요 시)
        }
    )
    logger.info("Conditional edge added after evaluation.")

    # 창작 단계 (generate_ideas 부터 시작)
    graph.add_edge("generate_ideas", "write_scenario")
    graph.add_edge("write_scenario", "report_scenario") # 시나리오 보고서 생성
    graph.add_edge("report_scenario", "generate_images") # 보고 후 이미지 생성
    graph.add_edge("generate_images", "translate_dialogue") # 이미지 생성 후 번역
    graph.add_edge("translate_dialogue", "postprocess_comic") # 번역 후 최종 후처리
    graph.add_edge("postprocess_comic", END) # 최종 완료

    logger.info("All edges defined.")

    # --- !!! 그래프 컴파일 (체크포인터 적용 수정) !!! ---
    checkpointer = None # 체크포인터 변수 초기화
    compiled_graph = None # 컴파일된 그래프 변수 초기화
    logger.info(f"Attempting to configure Redis checkpointer with URL: {settings.REDIS_CHECKPOINT_URL}")
    try:
        # Redis 연결 URL 설정 확인
        if not settings.REDIS_CHECKPOINT_URL:
            raise ValueError("REDIS_CHECKPOINT_URL is not configured in settings.")

        # Redis 클라이언트 생성 (연결은 실제 사용 시점에 이루어짐)
        # redis_client = redis.from_url(settings.REDIS_CHECKPOINT_URL, decode_responses=True)
        logger.info("Redis client configured for checkpointer.")

        # Redis 체크포인터 생성
        checkpointer = AsyncRedisSaver(settings.REDIS_CHECKPOINT_URL)
        logger.info("AsyncRedisSaver checkpointer created.")

        # 체크포인터와 함께 그래프 컴파일
        compiled_graph = graph.compile(checkpointer=checkpointer)
        logger.info("Workflow graph compiled successfully with Redis checkpointer.")

    except ValueError as ve: # 설정 오류 처리
         logger.error(f"Configuration error for Redis checkpointer: {ve}", exc_info=True)
         logger.warning("Compiling graph without checkpointer due to configuration error.")
         compiled_graph = graph.compile() # 체크포인터 없이 컴파일 (폴백)
    except ImportError: # AsyncRedisSaver 임포트 실패 시 (라이브러리 미설치 등)
         logger.warning("AsyncRedisSaver not available. Compiling graph without checkpointer. "
                        "Install 'langgraph[redis]' for Redis checkpointing.")
         compiled_graph = graph.compile() # 체크포인터 없이 컴파일 (폴백)
    except Exception as e: # 기타 예외 처리 (Redis 연결 실패 등은 실행 시점에 발생할 수 있음)
        logger.error(f"Error during checkpointer setup or graph compilation: {e}", exc_info=True)
        logger.warning("Compiling graph without checkpointer due to an unexpected error.")
        compiled_graph = graph.compile() # 체크포인터 없이 컴파일 (폴백)

    # 컴파일 성공 여부 최종 확인 (폴백 컴파일도 실패할 경우)
    if compiled_graph is None:
        logger.critical("Graph compilation failed even after fallback. Unable to build workflow.")
        # 여기서 에러를 발생시키거나, None을 반환하여 호출 측에서 처리하도록 할 수 있음
        raise RuntimeError("Failed to compile the workflow graph.")

    # (선택 사항) 그래프 시각화
    try:
        output_path = "full_workflow_graph.png"
        # get_graph().draw_mermaid_png() 가 비동기/블로킹 여부 확인 필요
        # compiled_graph가 None일 수 있으므로 확인 후 호출
        if compiled_graph:
            graph_png = compiled_graph.get_graph().draw_mermaid_png()
            if graph_png:
                with open(output_path, "wb") as f: f.write(graph_png)
                logger.info(f"Workflow graph image saved: {output_path}")
        else:
             logger.warning("Graph visualization skipped because graph compilation failed.")
    except ImportError as viz_err:
         logger.warning(f"Graph visualization skipped: {viz_err}. Install pygraphviz and its dependencies.")
    except Exception as viz_err:
        logger.warning(f"Failed to generate workflow graph image: {viz_err}")

    return compiled_graph

# --- 워크플로우 실행 및 리소스 관리 함수 (개선 버전) ---
async def run_workflow(initial_query: str, initial_config: Optional[Dict[str, Any]] = None):
    """
    전체 워크플로우 실행을 위한 서비스/도구/노드 초기화, 실행, 리소스 정리를 수행합니다.
    """
    main_logger = get_logger("FullWorkflowRunner")
    run_start_time = datetime.now(timezone.utc)
    main_logger.info(f"Starting full workflow run for query: '{initial_query}'")

    # --- 리소스 관리 ---
    # closable_resources 리스트를 사용하여 close() 메서드가 있는 객체 관리
    closable_resources: List[Any] = []
    # aiohttp 세션은 여러 도구에서 공유 가능
    shared_session: Optional[aiohttp.ClientSession] = None
    # Redis 클라이언트도 정리 필요 (체크포인터용) - build_main_workflow에서 생성되므로 여기서 직접 닫기는 어려움
    # -> 체크포인터 객체 자체의 close 메서드가 있는지 확인 필요 (LangGraph 문서 참조)
    # -> 또는 redis_client를 build 함수 외부에서 생성/관리

    try:
        # --- 1. 공유 리소스 생성 (aiohttp Session) ---
        # 참고: aiohttp 세션 생성 및 관리를 build_main_workflow 외부(예: lifespan)에서 하고
        #      필요한 도구에 주입하는 것이 더 효율적일 수 있음. 여기서는 run_workflow 내 생성 가정.
        shared_session = aiohttp.ClientSession()
        closable_resources.append(shared_session)
        main_logger.info("Shared aiohttp session created.")

        # --- 2. 워크플로우 빌드 ---
        # build_main_workflow 내에서 서비스/도구/노드 인스턴스화 가정
        compiled_app = build_main_workflow() # 수정된 빌드 함수 호출

        # 컴파일된 앱이 유효한지 확인 (빌드 중 오류 발생 시 None 또는 에러 반환 가능성)
        if compiled_app is None:
             raise RuntimeError("Workflow application could not be built.")

        # --- 3. 워크플로우 실행 ---
        main_logger.info("Starting workflow execution...")
        # 실행 ID 및 초기 상태 설정
        thread_id = str(uuid.uuid4()) # 고유 실행 ID
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"initial_query": initial_query}
        # 초기 config 전달 (사용자 정의 설정 등)
        if initial_config: inputs["config"] = initial_config

        final_state: Optional[ComicState] = None

        async for event in compiled_app.astream_events(inputs, config=config, version="v1"):
            kind = event["event"]
            name = event["name"]
            tags = event.get("tags", [])

            # 이벤트 로깅 (필요에 따라 상세 수준 조절)
            if kind == "on_chain_start" or kind == "on_tool_start":
                main_logger.debug(f"START: {name} (Tags: {tags})", extra={'trace_id': thread_id})
            elif kind == "on_chain_end" or kind == "on_tool_end":
                 output = event["data"].get("output")
                 # output이 너무 클 경우 요약 로깅
                 if isinstance(output, dict):
                     output_summary = f"Output keys: {list(output.keys())}"
                 elif isinstance(output, (str, bytes)) and len(output) > 200:
                     output_summary = f"Output type: {type(output)}, Len: {len(output)}, Start: {output[:100]}..."
                 else:
                    output_summary = f"Output type: {type(output)}"
                 main_logger.debug(f"END: {name} (Tags: {tags}) - {output_summary}", extra={'trace_id': thread_id})

                 # 최종 상태 저장 (END 이벤트에서 확인)
                 if name == END and isinstance(output, dict):
                      # LangGraph 구조에 따라 최종 상태 추출 방식이 달라질 수 있음
                      state_data = output
                      # 상태가 딕셔너리 내부에 중첩될 수 있음 ('__end__' 키 등 확인)
                      final_state_data = state_data.get('__end__', state_data) # '__end__' 키가 일반적

                      if isinstance(final_state_data, ComicState):
                          final_state = final_state_data
                      elif isinstance(final_state_data, dict):
                           try:
                               final_state = ComicState(**final_state_data)
                               main_logger.info("Final state parsed from END event output dict.", extra={'trace_id': thread_id})
                           except Exception as parse_err:
                                main_logger.warning(f"Failed to parse final state from dict: {parse_err}", extra={'trace_id': thread_id})
                      else:
                           main_logger.warning(f"Unexpected final state data type: {type(final_state_data)}", extra={'trace_id': thread_id})


            elif kind == "on_chain_error" or kind == "on_tool_error":
                main_logger.error(f"ERROR in {name}: {event['data']}", extra={'trace_id': thread_id})


        run_end_time = datetime.now(timezone.utc)
        total_duration = (run_end_time - run_start_time).total_seconds()
        main_logger.info(f"Workflow execution finished. Total duration: {total_duration:.2f} seconds.", extra={'trace_id': thread_id})

        # 최종 상태 로깅
        if final_state:
            final_url = final_state.final_comic.get('url') if final_state.final_comic else 'N/A'
            main_logger.info(f"Final State Summary: ComicID={final_state.comic_id}, TraceID={final_state.trace_id}, Error='{final_state.error_message or 'None'}', FinalURL='{final_url}'", extra={'trace_id': final_state.trace_id})
            # TODO: 최종 상태 DB 저장/업데이트 로직
        else:
            main_logger.warning("Could not retrieve or parse final state from workflow execution.", extra={'trace_id': thread_id})

    except Exception as e:
        main_logger.critical(f"Critical error during workflow run: {e}", exc_info=True)
        # TODO: 오류 발생 시 DB 상태 업데이트
    finally:
        # --- 4. 리소스 정리 ---
        main_logger.info("Starting resource cleanup...")
        # closable_resources 리스트에 추가된 리소스들의 close 메서드 호출
        # 여기서는 공유 세션만 정리하는 예시
        if shared_session:
            try:
                await shared_session.close()
                main_logger.info("Shared aiohttp session closed.")
            except Exception as close_err:
                main_logger.error(f"Error closing shared aiohttp session: {close_err}", exc_info=True)

        # --- Redis 클라이언트 정리 ---
        # build_main_workflow 내에서 생성된 redis_client는 여기서 직접 접근/정리하기 어려움.
        # AsyncRedisSaver 객체가 close 메서드를 제공하는지 확인 필요. (문서 확인 결과, conn 객체를 외부에서 관리해야 할 수 있음)
        # 또는, redis_client 인스턴스를 build_main_workflow 외부 (예: lifespan)에서 생성하고 주입하는 패턴 고려.
        # 여기서는 명시적인 redis_client.close() 호출은 생략 (현재 구조상 어려움).
        # main_logger.warning("Redis client for checkpointer was created inside build_main_workflow and cannot be explicitly closed here.")


        # TODO: build_main_workflow에서 생성된 다른 closable 리소스들 정리
        # 예: selenium_tool.close(), db_client.close() 등 (비동기/동기, close 메서드 유무 확인 필요)

        main_logger.info("Resource cleanup finished.")


# --- 스크립트 직접 실행 예시 ---
if __name__ == "__main__":
    # 로깅 설정 (실제 앱에서는 main 진입점에서 한 번 설정)
    # setup_logging()
    logging.basicConfig(level=logging.INFO) # 기본 로깅 설정

    test_query = "AI가 예술 생성에 미치는 영향"
    # 사용자 정의 설정 예시 (필요 시)
    custom_config = {
        # "llm_temperature_creative": 0.8, # 예시: 실행 시 특정 온도 설정
        # "translation_enabled": False,    # 예시: 번역 비활성화
        # "upload_to_s3": False,           # 예시: S3 업로드 비활성화
        # "final_comic_save_dir": "./output_comics" # 예시: 로컬 저장 경로 변경
    }

    try:
        # 비동기 워크플로우 실행
        asyncio.run(run_workflow(test_query, initial_config=custom_config))
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
    except Exception as e:
         logger.critical(f"Main execution failed: {e}", exc_info=True)