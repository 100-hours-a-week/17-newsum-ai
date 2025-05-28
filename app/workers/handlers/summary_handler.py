# ai/app/workers/handlers/summary_handler.py

import json
from typing import Dict, Any, List, Optional

from app.services.llm_service import LLMService
from app.services.postgresql_service import PostgreSQLService
from app.services.database_client import DatabaseClient
from app.services.backend_client import BackendApiClient  # 타입 힌트 및 일관성
from app.utils.logger import get_logger
from transformers import AutoTokenizer  # 타입 힌트

logger = get_logger("SummaryHandler")

SUMMARY_THRESHOLD = 500  # 설정 파일에서 가져오도록 변경 고려
DEFAULT_SUMMARY_PARAMS = {"max_tokens": 512, "temperature": 0.3}


def count_tokens_for_summary(text: str, tokenizer: Optional[AutoTokenizer]) -> int:
    """텍스트의 토큰 수를 계산합니다."""
    if tokenizer and text:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            logger.warning(f"Summary Tokenizer encoding failed: {e}. Falling back to word count.")
            return len(text.split())  # 폴백으로 단어 수 기반 계산
    elif text:  # 토크나이저가 없어도 텍스트가 있으면 단어 수 기반 계산
        logger.warning("Summary Tokenizer not available or text is None. Using word count if text exists.")
        return len(text.split())
    return 0  # 텍스트가 없으면 0 반환


def format_history_for_summary_prompt(history: list) -> str:
    """DB 대화 이력을 LLM 요약 프롬프트용 문자열로 변환합니다."""
    return "\n".join([f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in history])


async def handle_summary_update(
        payload: Dict[str, Any],
        llm: LLMService,  # LLMService 인스턴스
        pg: PostgreSQLService,
        redis: DatabaseClient,
        tokenizer: Optional[AutoTokenizer],  # 주입된 토크나이저
        backend_client: BackendApiClient  # 일관성을 위해 받지만 현재 사용 안 함
):
    """'update_summary' 작업을 처리합니다."""
    room_id = payload.get("room_id")
    request_id = payload.get("request_id", "N/A")  # 원본 채팅 요청 ID

    if not room_id:
        logger.error(f"SummaryHandler: Invalid payload for 'update_summary' (OrigReqID: {request_id}): {payload}")
        return

    logger.info(f"SummaryHandler: Processing 'update_summary' (OrigReqID: {request_id}) for room: {room_id}")

    try:
        full_history = await pg.get_full_history(room_id)
        if not full_history:
            logger.warning(
                f"SummaryHandler: No history found to summarize for room {room_id} (OrigReqID: {request_id}).")
            await pg.update_summary(room_id, "")
            return

        full_history_text = format_history_for_summary_prompt(full_history)
        token_count = count_tokens_for_summary(full_history_text, tokenizer)  # 토크나이저 사용

        new_summary = ""
        if token_count < SUMMARY_THRESHOLD:
            new_summary = full_history_text
            logger.info(
                f"SummaryHandler: Room {room_id} (OrigReqID: {request_id}) is below token threshold ({token_count}/{SUMMARY_THRESHOLD}). Storing full history as summary.")
        else:
            logger.info(
                f"SummaryHandler: Room {room_id} (OrigReqID: {request_id}) is above token threshold ({token_count}/{SUMMARY_THRESHOLD}). Requesting LLM summary.")

            summary_prompt_text = f"""Please provide a concise summary (ideally 3-5 sentences, focusing on key topics and outcomes, **within 200 tokens**) of the following conversation:

Conversation History:
---
{full_history_text}
---

Concise Summary:"""

            llm_params_from_payload = payload.get("llm_params", {}).copy()
            final_llm_params = DEFAULT_SUMMARY_PARAMS.copy()
            final_llm_params.update(llm_params_from_payload)
            if 'use_cot' in final_llm_params:  # 요약에는 CoT 불필요 가정
                del final_llm_params['use_cot']

            # LLMService.generate_text는 messages 인자를 받으므로 변환
            summary_messages = [{"role": "user", "content": summary_prompt_text}]

            summary_result = await llm.generate_text(
                messages=summary_messages,
                request_id=f"summary-{request_id}",  # 요약 작업용 고유 ID 생성
                **final_llm_params  # max_tokens 등 파라미터 전달
            )
            new_summary = summary_result.get("generated_text", "").strip()
            if not new_summary or "error" in summary_result:  # 에러 확인 추가
                new_summary = f"Failed to generate summary. Error: {summary_result.get('error', 'Unknown LLM error')}"
                logger.warning(
                    f"SummaryHandler: LLM returned empty or error for summary, room {room_id} (OrigReqID: {request_id}). Response: {summary_result}")

        await pg.update_summary(room_id, new_summary)
        logger.info(f"SummaryHandler: Room {room_id} (OrigReqID: {request_id}): Summary updated successfully.")

    except Exception as e:
        logger.error(
            f"SummaryHandler: Error processing 'update_summary' (OrigReqID: {request_id}) for room {room_id}: {e}",
            exc_info=True)