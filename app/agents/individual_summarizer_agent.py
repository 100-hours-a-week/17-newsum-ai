# app/agents/individual_summarizer_agent.py
import logging
import asyncio
from app.workflows.state import ComicState
from app.services.llm_server_client import call_llm_api
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

class IndividualSummarizerAgent:
    """
    스크랩된 각 뉴스 기사 텍스트를 개별적으로 요약하는 에이전트.
    """

    async def _summarize_single_article(self, article_text: str) -> Optional[str]:
        """단일 기사 텍스트를 LLM으로 요약하는 내부 비동기 함수"""
        if not article_text or len(article_text) < 50:
             logger.warning("[IndividualSummarizer] Article text is too short to summarize.")
             return None

        # LLM에게 전달할 프롬프트 (영어) - 개별 기사 요약 요청
        prompt = f"""Provide a concise summary of the following news article text, focusing on the main points and key facts.

Article Text:
---
{article_text[:4000]} # 모델 컨텍스트 길이 고려
---

Concise Summary:
"""
        logger.debug(f"[IndividualSummarizer] Sending text (first 100 chars) to LLM for summary: {article_text[:100]}...")

        try:
            summary = await call_llm_api(prompt, max_tokens=300, temperature=0.5) # 요약 길이는 조정 가능
            logger.info("[IndividualSummarizer] Successfully received summary from LLM.")
            logger.debug(f"[IndividualSummarizer] LLM Summary Result: {summary}")
            return summary
        except Exception as e:
            logger.error(f"[IndividualSummarizer] LLM summary failed: {e}")
            return None

    async def run(self, state: ComicState) -> Dict[str, Optional[Any]]:
        """
        state.articles 리스트의 각 기사에 대해 비동기적으로 요약을 수행하고,
        결과(개별 요약 리스트)를 포함하는 업데이트 딕셔너리를 반환합니다.
        """
        logger.info("--- [Individual Summarizer Agent] 실행 시작 ---")
        updates: Dict[str, Optional[Any]] = {}
        articles_to_summarize = state.articles

        if not articles_to_summarize:
            logger.warning("[IndividualSummarizer] 요약할 기사 내용이 없습니다.")
            updates["summaries"] = [] # 빈 리스트로 업데이트
            # 이전 단계 오류가 없다면 에러 메시지 설정 가능
            if not state.error_message:
                updates["error_message"] = "No article content to summarize."
            return updates

        logger.info(f"[IndividualSummarizer] Received {len(articles_to_summarize)} articles to summarize.")

        # asyncio.gather를 사용하여 여러 기사 요약을 동시에 실행
        summary_tasks = [self._summarize_single_article(article) for article in articles_to_summarize]
        summary_results = await asyncio.gather(*summary_tasks)

        successful_summaries = [summary for summary in summary_results if summary is not None]
        logger.info(f"[IndividualSummarizer] Successfully summarized {len(successful_summaries)} out of {len(articles_to_summarize)} articles.")

        if not successful_summaries:
            logger.error("[IndividualSummarizer] Failed to summarize any of the articles.")
            updates["summaries"] = []
            if not state.error_message:
                 updates["error_message"] = "Failed to summarize any of the scraped articles."
        else:
            updates["summaries"] = successful_summaries # 개별 요약 리스트 저장
            # 성공 시 이전 오류 메시지 초기화 (선택적)
            updates["error_message"] = None

        logger.info("--- [Individual Summarizer Agent] 실행 종료 ---")
        return updates