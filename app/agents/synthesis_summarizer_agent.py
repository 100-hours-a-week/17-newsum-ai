# app/agents/synthesis_summarizer_agent.py
import logging
from app.workflows.state import ComicState
from app.services.llm_server_client import call_llm_api
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

class SynthesisSummarizerAgent:
    """
    개별 요약문 리스트를 종합하여 최종 요약문을 생성하는 에이전트.
    """
    async def run(self, state: ComicState) -> Dict[str, Optional[str]]:
        """
        state.summaries 리스트를 입력받아 하나의 최종 요약을 생성하고,
        'final_summary' 필드를 업데이트하는 딕셔너리를 반환합니다.
        """
        logger.info("--- [Synthesis Summarizer Agent] 실행 시작 ---")
        updates: Dict[str, Optional[str]] = {}
        individual_summaries = state.summaries

        if not individual_summaries:
            logger.warning("[SynthesisSummarizer] 종합할 개별 요약문이 없습니다.")
            updates["final_summary"] = None
            # 이전 단계 오류가 없다면 에러 메시지 설정 가능
            if not state.error_message:
                updates["error_message"] = "No individual summaries available to synthesize."
            return updates

        logger.info(f"[SynthesisSummarizer] Received {len(individual_summaries)} individual summaries to synthesize.")

        # 개별 요약들을 하나의 텍스트로 합치기
        combined_text = "\n\n---\n\n".join(individual_summaries)
        logger.debug(f"[SynthesisSummarizer] Combined text for final summary (first 500 chars):\n{combined_text[:500]}")

        # 최종 요약을 위한 프롬프트 (영어)
        prompt = f"""Based on the following collection of individual news summaries, please create a single, overarching summary that captures the main theme or connection between them, if any. If they cover diverse topics, provide a brief synthesis of the key information presented overall.

Individual Summaries:
---
{combined_text[:4000]} # 모델 컨텍스트 길이 고려
---

Overall Synthesis Summary:
"""

        try:
            # LLM API 호출하여 최종 요약 생성
            final_summary = await call_llm_api(prompt, max_tokens=512, temperature=0.6) # 종합 요약이므로 temperature 약간 조정 가능
            logger.info("[SynthesisSummarizer] Successfully received final summary from LLM.")
            logger.debug(f"[SynthesisSummarizer] Final Summary: {final_summary}")
            updates["final_summary"] = final_summary # 최종 요약 저장
            updates["error_message"] = None # 성공 시 오류 초기화

        except Exception as e:
            logger.error(f"[SynthesisSummarizer] LLM final summarization failed: {e}")
            updates["final_summary"] = None
            updates["error_message"] = f"Failed to generate final summary: {str(e)}"

        logger.info("--- [Synthesis Summarizer Agent] 실행 종료 ---")
        return updates