# app/agents/humorator_agent.py
import logging
import json
import re
from app.workflows.state import ComicState
from app.services.llm_server_client import call_llm_api
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

class HumoratorAgent:
    """
    Agent that identifies empathetic humor points from news summaries and additional context.
    """
    async def run(self, state: ComicState) -> Dict[str, Optional[Any]]:
        """
        Takes state.final_summary and state.additional_context as inputs to extract empathetic humor points,
        and returns a dictionary to update the 'humor_texts' field.
        """
        logger.info("--- [Humorator Agent] Starting execution ---")
        updates: Dict[str, Optional[Any]] = {}
        final_summary = state.final_summary
        additional_context = state.additional_context

        if not final_summary:
            logger.warning("[Humorator] No summary available to find humor points.")
            updates["humor_texts"] = []
            if not state.error_message:
                updates["error_message"] = "No summary available to find humor points."
            return updates

        logger.info("[Humorator] Received summary and additional context for humor generation.")
        
        # Extract additional context information
        context_info = "No additional context available"
        perspectives = []
        
        if additional_context:
            try:
                context_info = additional_context.get("summary", "No additional context")
                perspectives = additional_context.get("perspectives", [])
                logger.info(f"[Humorator] Additional context available with {len(perspectives)} perspectives")
            except Exception as e:
                logger.error(f"[Humorator] Error processing additional context: {e}")
                context_info = "Error processing additional context"

        # Prompt for identifying empathetic humor points
        prompt = f"""Based on the following news summary and additional context from YouTube videos and comments, identify 'empathetic' humor points.

News Summary:
---
{final_summary[:2500]}
---

Additional Context from YouTube:
---
{context_info[:1500]}
---

Key Perspectives from Audience:
---
{chr(10).join([f"- {p}" for p in perspectives])}
---

Empathetic Humor Guidelines:
1. Look for irony, absurdity, exaggeration, or unexpected contrasts in the situation.
2. Focus on systemic issues, contradictions in official positions, or situational ironies.
3. Use the audience perspectives to inform your humor, addressing their concerns or observations.
4. Never include content that mocks vulnerable groups or victims ('punching down').
5. If the situation involves tragedy or suffering, be especially careful to focus humor on systemic failures rather than individuals' misfortune.

List 3-5 empathetic humor points, each on a separate line starting with '-'. 
Each point should be insightful and suitable for a panel in a comic strip.
If you can't find appropriate humor points, respond with "None".

Empathetic Humor Points:"""  # 마지막 줄의 "-" 제거

        try:
            # LLM API call
            llm_response = await call_llm_api(prompt, max_tokens=400, temperature=0.6)
            logger.info("[Humorator] Successfully received LLM response.")
            logger.debug(f"[Humorator] Raw LLM response:\n{llm_response}")

            # 개선된 응답 파싱 로직
            # 1. 응답에서 "Empathetic Humor Points:" 이후의 텍스트만 추출
            response_content = llm_response
            if "Empathetic Humor Points:" in response_content:
                response_content = response_content.split("Empathetic Humor Points:", 1)[1].strip()
            
            # 2. 정규식을 사용하여 대시로 시작하는 줄 추출
            humor_points = []
            for line in response_content.split('\n'):
                line = line.strip()
                if line.startswith('-') or re.match(r'^\d+\.', line):  # '-' 또는 '숫자.' 형식으로 시작하는 줄 확인
                    # 앞의 '-' 또는 '숫자.' 제거
                    clean_line = re.sub(r'^-\s*|^\d+\.\s*', '', line).strip()
                    if clean_line and clean_line.lower() != "none":
                        humor_points.append(clean_line)
            
            # 3. 결과 검증 및 로깅
            if not humor_points:
                # 파싱이 실패하면 단순한 방법으로 다시 시도: 빈 줄이 아닌 모든 줄 사용
                if "none" not in response_content.lower():
                    potential_points = [line.strip() for line in response_content.split('\n') if line.strip()]
                    if potential_points:
                        humor_points = potential_points[:5]  # 최대 5개 항목만 사용
                        logger.warning(f"[Humorator] Fallback parsing used, found {len(humor_points)} potential points")

            # "None" 응답이거나 빈 결과 처리
            if not humor_points or (len(humor_points) == 1 and "none" in humor_points[0].lower()):
                logger.info("[Humorator] No appropriate humor points identified.")
                updates["humor_texts"] = []  # 유머 포인트가 없으면 빈 리스트
            else:
                # 최종 검증: 빈 문자열이나 너무 짧은 항목 제거
                humor_points = [point for point in humor_points if len(point) > 5]
                logger.info(f"[Humorator] Identified {len(humor_points)} humor points: {humor_points}")
                updates["humor_texts"] = humor_points

            updates["error_message"] = None  # 성공 시 오류 초기화

        except Exception as e:
            logger.error(f"[Humorator] LLM call or processing failed: {e}")
            updates["humor_texts"] = []
            updates["error_message"] = f"Failed to generate humor points: {str(e)}"

        logger.info("--- [Humorator Agent] Execution complete ---")
        return updates
