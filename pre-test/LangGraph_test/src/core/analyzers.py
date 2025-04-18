# src/core/analyzers.py (수정: 외부 LLM API 사용)
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .schemas import NewsItem, AnalysisResult
from .utils import logger
from src import settings
import re

# 외부 LLM API 클라이언트 초기화 (애플리케이션 시작 시 한 번 로드)
try:
    llm_client = ChatOpenAI(
        openai_api_base=settings.LLM_API_BASE_URL,
        openai_api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL_NAME,
        temperature=0.7,
        max_tokens=512,
        # request_timeout=60, # 타임아웃 설정 (초)
    )
    logger.info(f"LLM API client initialized for endpoint: {settings.LLM_API_BASE_URL}")
except Exception as e:
    logger.error(f"Failed to initialize LLM API client: {e}")
    llm_client = None

# CoT 결과 파싱 함수 (동일)
def parse_cot_output(text: str) -> tuple[str | None, str | None, str | None]:
    """CoT 결과 파싱 (예시)"""
    key_event = re.search(r"Key Event: (.*?)\n", text, re.IGNORECASE)
    twist_point = re.search(r"Twist Point: (.*?)\n", text, re.IGNORECASE)
    summary = re.search(r"Summary: (.*)", text, re.IGNORECASE | re.DOTALL)
    return (key_event.group(1).strip() if key_event else None,
            twist_point.group(1).strip() if twist_point else None,
            summary.group(1).strip() if summary else text)

async def analyze_article(item: NewsItem) -> AnalysisResult:
    """외부 LLM API를 이용해 뉴스 분석 (CoT 프롬프트 적용)"""
    if not llm_client:
        raise RuntimeError("LLM API client is not initialized.")

    logger.info(f"Analyzing news via API: {item.id} - {item.title[:30]}...")
    # Chain-of-Thought 프롬프트 (템플릿 사용 권장)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant specialized in analyzing news for humor generation. Follow the steps precisely."),
        ("human", """Analyze the following news article step-by-step to identify the core narrative:
News Title: {title}
News Content: {content} # Max 1000 chars

Step 1: Identify the main subject and key event.
Step 2: Find an unexpected twist, irony, or absurdity.
Step 3: Briefly summarize the core situation.

Output format:
Key Event: [Result of Step 1]
Twist Point: [Result of Step 2]
Summary: [Result of Step 3]""")
    ])
    parser = StrOutputParser()
    chain = prompt_template | llm_client | parser

    try:
        # 비동기 API 호출
        raw_output = await chain.ainvoke({
            "title": item.title,
            "content": item.content[:1000]
        })

        key_event, twist_point, summary = parse_cot_output(raw_output)

        # 간단한 키워드 추출 (별도 API 호출 또는 정교한 로직 필요)
        keywords = summary.split()[:5] if summary else []

        result = AnalysisResult(
            news_id=item.id,
            summary=summary,
            keywords=keywords,
            key_event=key_event,
            twist_point=twist_point
        )
        logger.info(f"Analysis via API complete for {item.id}")
        return result
    except Exception as e:
        logger.error(f"LLM API call failed during analysis for {item.id}: {e}")
        raise e # LangGraph 재시도 메커니즘 활용

# ----------------------------------------