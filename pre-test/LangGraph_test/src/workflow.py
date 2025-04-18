# src/workflow.py
import asyncio
from langgraph.graph import StateGraph, END, START
import httpx # httpx 등 네트워크 오류 처리용 (예외 처리 시 사용 가능)
from typing import List, Dict, Optional, Any # List 사용을 위해 추가
from pathlib import Path
from datetime import datetime

# 코어 모듈의 함수 임포트
from .core.collectors import fetch_latest_news
from .core.analyzers import analyze_article
from .core.humorators import make_joke
from .core.prompters import make_prompt
from .core.imagers import generate_image # 내부에서 외부 API 호출
from .core.schemas import (
    WorkflowState, NewsItem, AnalysisResult, HumorResult,
    ImagePromptResult, ImageRenderResult, FinalContent
)
from .core.utils import logger, save_json
from . import settings

# --- 그래프 노드 함수 정의 ---

async def collect_news_node(state: WorkflowState) -> Dict[str, Any]:
    """뉴스 수집 노드"""
    logger.info("--- Starting News Collection Node ---")
    try:
        news_items = await fetch_latest_news()
        # 간단한 필터링 예시: 내용 없는 뉴스 제외
        news_items = [item for item in news_items if item.content and item.content.strip()]
        logger.info(f"Collected {len(news_items)} valid news items after filtering.")
        return {"news_items": news_items, "error_message": None}
    except Exception as e:
        logger.error(f"Collection node error: {e}", exc_info=True) # 상세 에러 로깅
        # 실패 시 다음 단계로 진행하지 않도록 빈 리스트 반환 또는 에러 상태 명시
        return {"news_items": [], "error_message": f"Collection failed: {e}"}

async def analyze_news_node(state: WorkflowState) -> Dict[str, Any]:
    """단일 뉴스 분석 노드 (map 연산용) - 외부 LLM API 호출"""
    item = state.get("current_news_item")
    if not item:
        # map 연산 시 빈 입력이 올 수 있으므로 처리
        return {"analysis_result": None}
    logger.info(f"--- Starting Analysis Node for {item.id} ---")
    try:
        # analyze_article 함수는 내부적으로 외부 LLM API를 호출
        analysis_result = await analyze_article(item)
        return {"analysis_result": analysis_result, "error_message": None}
    except Exception as e:
        # API 호출 실패 등 예외 발생 시 로깅하고 None 반환하여 다음 단계에서 처리
        logger.error(f"Analysis node error for {item.id}: {e}", exc_info=True)
        return {"analysis_result": None, "error_message": f"Analysis failed: {e}"}

async def humorize_node(state: WorkflowState) -> Dict[str, Any]:
    """유머 생성 노드 - 외부 LLM API 호출"""
    analysis = state.get("analysis_result")
    # 이전 단계(분석) 실패 시 실행하지 않음
    if not analysis:
        # 수정된 로깅 부분: Pydantic 객체 속성 직접 접근
        current_news_item = state.get('current_news_item')
        news_id = current_news_item.id if current_news_item else 'unknown'
        logger.warning(f"Skipping humorization for {news_id} due to missing analysis.")
        return {"humor_result": None}
    logger.info(f"--- Starting Humorization Node for {analysis.news_id} ---")
    try:
        # make_joke 함수는 내부적으로 외부 LLM API를 호출
        humor_result = await make_joke(analysis)
        return {"humor_result": humor_result, "error_message": None}
    except Exception as e:
        logger.error(f"Humorization node error for {analysis.news_id}: {e}", exc_info=True)
        return {"humor_result": None, "error_message": f"Humorization failed: {e}"}

def draft_prompt_node(state: WorkflowState) -> Dict[str, Any]:
    """이미지 프롬프트 생성 노드 (동기 함수)"""
    humor = state.get("humor_result")
    analysis = state.get("analysis_result")
    # 이전 단계(유머 생성) 실패 시 실행하지 않음
    if not humor or not analysis:
        # 수정된 로깅 부분: Pydantic 객체 속성 직접 접근
        current_news_item = state.get('current_news_item')
        news_id = current_news_item.id if current_news_item else 'unknown' # 수정됨
        logger.warning(f"Skipping prompt drafting for {news_id} due to missing humor or analysis result.")
        return {"image_prompt_result": None}
    logger.info(f"--- Starting Prompt Drafting Node for {humor.news_id} ---")
    try:
        # make_prompt는 동기 함수
        prompt_result = make_prompt(humor, analysis)
        return {"image_prompt_result": prompt_result, "error_message": None}
    except Exception as e:
        logger.error(f"Prompt drafting error for {humor.news_id}: {e}", exc_info=True)
        return {"image_prompt_result": None, "error_message": f"Drafting failed: {e}"}

async def render_image_node(state: WorkflowState) -> Dict[str, Any]:
    """이미지 생성 노드 - 외부 SD API 호출"""
    prompt = state.get("image_prompt_result")
    # 이전 단계(프롬프트 생성) 실패 시 실행하지 않음
    if not prompt:
        news_id = state.get('humor_result').news_id if state.get('humor_result') else 'unknown'
        logger.warning(f"Skipping image rendering for {news_id} due to missing image prompt.")
        # 폴백 이미지를 사용하기 위한 기본 정보라도 반환할지 결정 필요
        # 여기서는 단순히 결과 없음을 반환
        return {"image_render_result": None}
    logger.info(f"--- Starting Image Rendering Node for {prompt.news_id} ---")
    try:
        # generate_image 함수는 내부적으로 외부 SD API를 호출하고 실패 시 폴백 처리 포함
        render_result = await generate_image(prompt)
        return {"image_render_result": render_result, "error_message": None}
    except Exception as e:
        # generate_image 내부에서 처리되지 않은 예외 발생 시
        logger.error(f"Image rendering node unexpected error for {prompt.news_id}: {e}", exc_info=True)
        # 최종 폴백 처리
        fallback_path = str(Path(settings.FALLBACK_IMAGE_PATH).relative_to(settings.OUTPUT_DIR))
        render_result = ImageRenderResult(news_id=prompt.news_id, image_path=fallback_path, is_fallback=True)
        return {"image_render_result": render_result, "error_message": f"Rendering failed unexpectedly: {e}"}

def format_and_save_node(state: WorkflowState) -> Dict[str, Any]:
    """최종 결과 포맷 및 저장 노드"""
    # 각 단계의 결과 가져오기
    news_item = state.get("current_news_item")
    analysis = state.get("analysis_result") # 참고용으로 가져올 수 있음
    humor = state.get("humor_result")
    render = state.get("image_render_result")

    # 필수 데이터(뉴스 원본, 유머 결과, 이미지 결과)가 없으면 저장 불가
    if not news_item or not humor or not render:
        nid = news_item.id if news_item else 'unknown'
        logger.warning(f"Missing essential data for final formatting/saving for {nid}. Skipping.")
        # map 연산의 결과 리스트에 None이 포함되도록 반환
        return {"final_content": None}

    logger.info(f"--- Starting Format & Save Node for {news_item.id} ---")
    try:
        final_content = FinalContent(
            news_id=news_item.id,
            title=news_item.title,
            url=news_item.url,
            source=news_item.source,
            published_time=news_item.published_time,
            humor_text=humor.humor_text,
            image_path=render.image_path # 상대 경로 또는 전체 URL
        )
        # JSON 파일로 저장
        output_path = settings.FINAL_JSON_DIR / f"{news_item.id}.json"
        # save_json 유틸리티 함수 사용 (core/utils.py에 정의됨)
        save_json(final_content.model_dump(), output_path)

        # 성공 시 최종 콘텐츠 객체 반환
        return {"final_content": final_content}
    except Exception as e:
        logger.error(f"Format/Save node error for {news_item.id}: {e}", exc_info=True)
        # 실패 시 None 반환
        return {"final_content": None, "error_message": f"Save failed: {e}"}

def aggregate_results_node(state: WorkflowState) -> Dict[str, List[FinalContent]]:
    """map 연산 결과 취합 노드"""
    # LangGraph의 map은 결과를 'processed_results' 키가 아닌,
    # 마지막 노드('format_save')의 출력 키('final_content')를 가진 리스트로 반환하는 경향이 있음.
    # 상태에서 map의 결과 리스트를 가져오는 방식 확인 필요 (LangGraph 버전에 따라 다를 수 있음)
    # 여기서는 map의 결과가 state['final_content'] 리스트에 담긴다고 가정 (또는 다른 키일 수 있음)
    # 혹은 map 호출 시 'output_key'를 지정할 수도 있음.

    # 예시: map 연산이 각 결과를 state['final_content']에 넣고, 이게 리스트로 쌓인다고 가정
    processed_outputs = state.get("final_content")
    if isinstance(processed_outputs, list):
      final_valid_results = [res for res in processed_outputs if isinstance(res, FinalContent)]
      logger.info(f"--- Aggregating Results: {len(final_valid_results)} items processed successfully out of {len(processed_outputs)} attempts ---")
      # 최종 결과를 'processed_results' 키로 명확히 저장
      return {"processed_results": final_valid_results}
    else:
      # 예상과 다른 경우 로깅
      logger.warning(f"Aggregate node received unexpected data type for final results: {type(processed_outputs)}")
      return {"processed_results": []}

def prepare_first_item_node(state: WorkflowState) -> Dict[str, Any]:
    """수집된 뉴스 리스트에서 첫 번째 아이템만 추출"""
    news_items = state.get("news_items")
    if news_items and len(news_items) > 0:
        logger.info(f"Preparing first item for processing: {news_items[0].id}")
        # 상태에 'current_news_item'으로 첫 번째 뉴스 저장
        return {"current_news_item": news_items[0]}
    else:
        logger.warning("No news item to prepare.")
        # 처리할 아이템 없음을 명시
        return {"current_news_item": None}

# --- 그래프 정의 ---
# 메인 워크플로우 빌더
workflow_builder = StateGraph(WorkflowState)

# 개별 아이템 처리 서브 그래프 빌더
item_processor_builder = StateGraph(WorkflowState)

# 서브 그래프 노드 추가
item_processor_builder.add_node("analyze", analyze_news_node)
item_processor_builder.add_node("humorize", humorize_node)
item_processor_builder.add_node("draft_prompt", draft_prompt_node)
item_processor_builder.add_node("render_image", render_image_node)
item_processor_builder.add_node("format_save", format_and_save_node)

# 서브 그래프 엣지 연결
item_processor_builder.set_entry_point("analyze")
item_processor_builder.add_edge("analyze", "humorize")
item_processor_builder.add_edge("humorize", "draft_prompt")
item_processor_builder.add_edge("draft_prompt", "render_image")
item_processor_builder.add_edge("render_image", "format_save")
item_processor_builder.add_edge("format_save", END) # 서브 그래프 종료

# 서브 그래프 컴파일
item_processor_graph = item_processor_builder.compile()


# 메인 그래프 노드 추가 (map 제거, 단일 처리 노드 추가)
workflow_builder.add_node("collect_news", collect_news_node)
workflow_builder.add_node("prepare_first_item", prepare_first_item_node)
# 컴파일된 서브 그래프를 직접 노드로 추가
workflow_builder.add_node("process_single_item", item_processor_graph)
# aggregate_results_node 제거

# 메인 그래프 엣지 연결 (map 관련 제거, 단일 처리 흐름으로 변경)
workflow_builder.set_entry_point("collect_news")# 뉴스 수집 후 분기 처리: 수집된 뉴스가 없으면 바로 종료
def route_after_collection(state: WorkflowState) -> str:
    if state.get("news_items") and len(state["news_items"]) > 0:
        return "prepare_first_item" # 첫 아이템 준비 노드로 이동
    else:
        logger.warning("No news items collected. Ending workflow.")
        return END # 처리할 뉴스 없음

workflow_builder.add_conditional_edges(
    "collect_news",
    route_after_collection,
    {
        "prepare_first_item": "prepare_first_item",
        END: END
    }
)

# 첫 아이템 준비 후 분기 처리: 처리할 아이템이 없으면 종료
def route_after_prepare(state: WorkflowState) -> str:
    if state.get("current_news_item"):
        return "process_single_item" # 아이템 처리 서브 그래프 실행
    else:
        logger.warning("No item prepared for processing. Ending workflow.")
        return END

workflow_builder.add_conditional_edges(
    "prepare_first_item",
    route_after_prepare,
    {
        "process_single_item": "process_single_item",
        END: END
    }
)

# 단일 아이템 처리 후 종료
workflow_builder.add_edge("process_single_item", END)

# 최종 워크플로우 컴파일
workflow = workflow_builder.compile()

logger.info("LangGraph workflow compiled successfully (using external APIs, no explicit with_retry).")