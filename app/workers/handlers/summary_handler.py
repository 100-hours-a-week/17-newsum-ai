# ai/app/workers/handlers/summary_handler.py

import os
from typing import Dict, Any, List

from app.services.llm_service import LLMService
from app.services.postgresql_service import PostgreSQLService
from app.services.database_client import DatabaseClient # Redis 클라이언트 임포트 추가
from app.utils.logger import get_logger

# Qwen 토크나이저 설정 (이전과 동일, 경로 확인 필요)
try:
    from transformers import AutoTokenizer
    # QWEN_TOKENIZER_PATH = "mnt/c/Users/xodnr/Downloads/dev/project/17-team-4cut/ai/pre-test/qwen3-14b-awq"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    TOKENIZER = AutoTokenizer.from_pretrained(current_dir, trust_remote_code=True)
    logger_init = get_logger("TokenizerInit")
    logger_init.info("Qwen Tokenizer loaded successfully.")
except Exception as e:
    TOKENIZER = None
    logger_init = get_logger("TokenizerInit")
    logger_init.error(f"Failed to load Qwen Tokenizer: {e}")


logger = get_logger("SummaryHandler")

SUMMARY_THRESHOLD = 500
DEFAULT_SUMMARY_PARAMS = {"max_tokens": 512, "temperature": 0.3}


def count_tokens_for_summary(text: str) -> int:
    """텍스트의 토큰 수를 계산합니다."""
    if TOKENIZER:
        try:
            return len(TOKENIZER.encode(text, add_special_tokens=False))
        except Exception as e:
            logger.warning(f"Tokenizer encoding failed: {e}. Falling back to word count.")
            return len(text.split())
    else:
        logger.warning("Tokenizer not available. Using word count.")
        return len(text.split())


def format_history_for_summary_prompt(history: list) -> str:
    """DB 대화 이력을 LLM 요약 프롬프트용 문자열로 변환합니다."""
    # PG 서비스에서 반환된 'role' (닉네임 또는 'AI') 사용
    return "\n".join([f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in history])


async def handle_summary_update(
        payload: Dict[str, Any],
        llm: LLMService,
        pg: PostgreSQLService,
        redis: DatabaseClient  # <-- 사용하지 않더라도 인자 추가
):
    """
    'update_summary' 타입의 작업을 처리합니다.
    (redis 인자는 워커 호출 시그니처와의 호환성을 위해 추가되었으며, 이 함수 내에서는 사용되지 않습니다.)
    """
    room_id = payload.get("room_id")
    request_id = payload.get("request_id", "N/A")

    if not room_id:
        logger.error(f"SummaryHandler: Invalid payload for 'update_summary' (OrigReqID: {request_id}): {payload}")
        return

    logger.info(f"SummaryHandler: Processing 'update_summary' (OrigReqID: {request_id}) for room: {room_id}")

    try:
        # 1. PG에서 전체 대화 이력 조회 (이미 role 매핑됨)
        full_history = await pg.get_full_history(room_id)
        if not full_history:
            logger.warning(
                f"SummaryHandler: No history found to summarize for room {room_id} (OrigReqID: {request_id}).")
            await pg.update_summary(room_id, "")  # 빈 요약으로 업데이트 또는 스킵
            return

        # 2. 요약 프롬프트용 문자열 변환 및 토큰 수 계산
        full_history_text = format_history_for_summary_prompt(full_history)
        token_count = count_tokens_for_summary(full_history_text)

        new_summary = ""
        # 3. 임계값 비교
        if token_count < SUMMARY_THRESHOLD:
            new_summary = full_history_text  # 전체 대화 텍스트를 요약으로 사용
            logger.info(
                f"SummaryHandler: Room {room_id} (OrigReqID: {request_id}) is below token threshold ({token_count}/{SUMMARY_THRESHOLD}). Storing full history as summary.")
        else:
            # 4. 임계값 초과 시 LLM 요약 호출
            logger.info(
                f"SummaryHandler: Room {room_id} (OrigReqID: {request_id}) is above token threshold ({token_count}/{SUMMARY_THRESHOLD}). Requesting LLM summary.")

            summary_prompt = f"""Please provide a concise summary (ideally 3-5 sentences, focusing on key topics and outcomes) of the following conversation:

Conversation History:
---
{full_history_text}
---

Concise Summary:"""

            # 페이로드에서 LLM 파라미터 추출 (API에서 설정한 기본값 또는 작업 생성 시 설정한 값 사용)
            llm_params_from_payload = payload.get("llm_params", {}).copy()
            final_llm_params = DEFAULT_SUMMARY_PARAMS.copy()
            final_llm_params.update(llm_params_from_payload)

            # 요약은 CoT를 사용하지 않도록 use_cot 명시적 제거 또는 False 설정 (필요시)
            if 'use_cot' in final_llm_params:
                del final_llm_params['use_cot']  # 요약 프롬프트는 CoT 지시가 없음

            logger.debug(f"SummaryHandler: LLM params for summary (OrigReqID: {request_id}): {final_llm_params}")

            summary_result = await llm.generate_text(prompt=summary_prompt, **final_llm_params)
            new_summary = summary_result.get("generated_text", "").strip()
            if not new_summary:
                new_summary = "Failed to generate a valid summary."
                logger.warning(
                    f"SummaryHandler: LLM returned empty summary for room {room_id} (OrigReqID: {request_id}).")

        # 5. PG에 요약 갱신
        await pg.update_summary(room_id, new_summary)
        logger.info(f"SummaryHandler: Room {room_id} (OrigReqID: {request_id}): Summary updated successfully.")

    except Exception as e:
        logger.error(
            f"SummaryHandler: Error processing 'update_summary' (OrigReqID: {request_id}) for room {room_id}: {e}",
            exc_info=True)