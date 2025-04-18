# src/core/humorators.py (수정: 외부 LLM API 사용)
import asyncio
import random
from .schemas import AnalysisResult, HumorResult
from .utils import logger
from .analyzers import llm_client # analyzers에서 초기화된 클라이언트 사용
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List
import re

async def generate_single_joke_via_api(analysis: AnalysisResult) -> str:
    """외부 LLM API를 이용해 단일 유머 생성"""
    if not llm_client:
        raise RuntimeError("LLM API client is not initialized.")

    # 템플릿 기반 프롬프트 예시 (headline + tagline)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a witty comedian AI. Generate a funny headline and tagline based on the news summary."),
        ("human", """News Summary: '{summary}'

Generate a satirical headline and a one-liner tagline (like an expert's comment) based on the summary above.
Format:
Headline: [Your funny headline]
Tagline: [Your witty tagline]""")
    ])
    parser = StrOutputParser()
    chain = prompt_template | llm_client | parser

    try:
        # 비동기 API 호출
        raw_output = await chain.ainvoke({"summary": analysis.summary})

        # 간단한 파싱 (실제로는 더 견고한 파싱 필요)
        headline = re.search(r"Headline: (.*?)\n", raw_output)
        tagline = re.search(r"Tagline: (.*)", raw_output)
        if headline and tagline:
            return f"{headline.group(1).strip()} - Expert: '{tagline.group(1).strip()}'"
        else:
            return raw_output # 파싱 실패 시 원본 반환
    except Exception as e:
        logger.error(f"LLM API call failed during humor generation for {analysis.news_id}: {e}")
        raise e

async def generate_multiple_jokes_via_api(analysis: AnalysisResult, num_candidates=3) -> List[str]:
    """동일 프롬프트로 여러 유머 후보 생성 (API 호출 병렬화)"""
    logger.info(f"Generating {num_candidates} humor candidates via API for {analysis.news_id}...")
    tasks = [generate_single_joke_via_api(analysis) for _ in range(num_candidates)]
    results = await asyncio.gather(*tasks, return_exceptions=True) # 예외 발생 시에도 계속 진행

    candidates = [res for res in results if isinstance(res, str)]
    errors = [res for res in results if isinstance(res, Exception)]

    if errors:
        logger.warning(f"{len(errors)} errors during parallel humor generation for {analysis.news_id}")

    logger.info(f"Generated {len(candidates)} humor candidates via API for {analysis.news_id}")
    return candidates

async def select_best_joke(candidates: List[str], analysis: AnalysisResult) -> str:
    """후보 중 최고의 유머 선택 (여기서는 랜덤)"""
    if not candidates:
        # 폴백: 매우 간단한 템플릿
        return f"Regarding '{analysis.summary[:30]}...', well, that's certainly something!"
    # TODO: Implement CLIP score selection or other ranking mechanism
    best_joke = random.choice(candidates)
    logger.info(f"Selected joke for {analysis.news_id}: {best_joke[:50]}...")
    return best_joke

async def make_joke(analysis: AnalysisResult) -> HumorResult:
    """분석 결과를 바탕으로 외부 LLM API 이용해 유머 생성"""
    try:
        # 수정된 함수 호출: generate_multiple_jokes -> generate_multiple_jokes_via_api
        candidates = await generate_multiple_jokes_via_api(analysis, num_candidates=3)
        best_joke_text = await select_best_joke(candidates, analysis)

        return HumorResult(
            news_id=analysis.news_id,
            humor_text=best_joke_text,
            humor_style="satire_api" # 적용된 스타일 명시
        )
    except Exception as e:
        # 오류 발생 시 로깅은 이미 humorize_node에서 하므로 여기서는 그냥 예외를 다시 발생시켜도 됨
        # logger.error(f"Humorization via API failed for {analysis.news_id}: {e}")
        raise e # 오류를 상위(workflow 노드)로 전달


# ----------------------------------------