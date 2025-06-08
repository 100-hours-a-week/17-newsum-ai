# ai/app/workers/handlers/summary_handler.py

import json
from typing import Dict, Any, List, Optional
import uuid

from app.services.llm_service import LLMService
from app.services.postgresql_service import PostgreSQLService
from app.services.database_client import DatabaseClient
from app.services.backend_client import BackendApiClient
from app.utils.logger import get_logger
from transformers import AutoTokenizer

logger = get_logger(__name__)

SUMMARY_THRESHOLD = 500
DEFAULT_SUMMARY_PARAMS = {"max_tokens": 350, "temperature": 0.3}


def count_tokens_for_summary(text: str, tokenizer: Optional[AutoTokenizer]) -> int:
    if tokenizer and text:  # tokenizer가 None이 아니고 text가 존재할 때만 encode 시도
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            logger.warning(f"Summary Tokenizer encoding failed: {e}. Falling back to word count.")
            return len(text.split())
    elif text:
        logger.warning("Summary Tokenizer not available or text is None for summary. Using word count if text exists.")
        return len(text.split())
    return 0


def format_history_for_summary_prompt(history: list) -> str:
    return "\n".join([f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in history])


async def handle_summary_update(
        payload: Dict[str, Any],
        llm: LLMService,
        pg: PostgreSQLService,
        redis: DatabaseClient,
        tokenizer: Optional[AutoTokenizer],
        backend_client: BackendApiClient
):
    room_id_str = payload.get("room_id")
    work_id = payload.get("work_id", "N/A_SUMMARY_WORK_ID")

    log_extra = {"trace_id": work_id, "room_id": room_id_str}  # logger.py의 ContextFilter와 연동

    if not room_id_str:
        logger.error(f"Invalid payload for 'update_summary': Missing room_id. Payload: {payload}", extra=log_extra)
        return

    logger.info(f"Processing 'update_summary' for room: {room_id_str}", extra=log_extra)

    try:
        full_history = await pg.get_full_history(room_id_str)
        if not full_history:
            logger.warning(
                f"No history found to summarize for room {room_id_str}.", extra=log_extra)
            await pg.update_summary(room_id_str, "")
            return

        full_history_text = format_history_for_summary_prompt(full_history)
        token_count = count_tokens_for_summary(full_history_text, tokenizer)

        new_summary = ""
        if token_count < SUMMARY_THRESHOLD:
            new_summary = full_history_text
            logger.info(
                f"Room {room_id_str} is below token threshold ({token_count}/{SUMMARY_THRESHOLD}). Storing full history as summary.",
                extra=log_extra)
        else:
            logger.info(
                f"Room {room_id_str} is above token threshold ({token_count}/{SUMMARY_THRESHOLD}). Requesting LLM summary.",
                extra=log_extra)

            summary_prompt_text = f"""Please provide a concise summary (ideally 5-7 sentences, focusing on key topics, decisions, and outcomes, **strictly within 300 tokens**) of the following conversation:

Conversation History:
---
{full_history_text}
---

Concise Summary (within 300 tokens):"""

            llm_params_from_payload = payload.get("llm_params", {}).copy()
            final_llm_params = DEFAULT_SUMMARY_PARAMS.copy()
            final_llm_params.update(llm_params_from_payload)
            if 'use_cot' in final_llm_params:
                del final_llm_params['use_cot']

            summary_messages = [{"role": "user", "content": summary_prompt_text}]

            summary_result = await llm.generate_text(
                messages=summary_messages,
                request_id=f"summary-{work_id}",
                **final_llm_params
            )

            generated_text = summary_result.get("generated_text", "").strip()
            if "error" in summary_result or not generated_text:
                error_message = summary_result.get('error', 'Unknown LLM error or empty response')
                new_summary = f"(Summary generation failed. Error: {error_message})"
                logger.warning(
                    f"LLM returned empty or error for summary, room {room_id_str}. Response: {summary_result}",
                    extra=log_extra)
            else:
                new_summary = generated_text
                summary_token_count = count_tokens_for_summary(new_summary, tokenizer)
                if summary_token_count > 350:  # 목표 300 + 여유 50
                    logger.warning(
                        f"Generated summary for room {room_id_str} exceeds target+margin tokens ({summary_token_count}/300).",
                        extra=log_extra)

        await pg.update_summary(room_id_str, new_summary)
        logger.info(f"Room {room_id_str}: Summary updated successfully.", extra=log_extra)

    except Exception as e:
        logger.error(
            f"Error processing 'update_summary' for room {room_id_str}: {e}",
            exc_info=True, extra=log_extra)