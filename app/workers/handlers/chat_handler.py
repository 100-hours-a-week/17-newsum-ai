# ai/app/workers/handlers/chat_handler.py

import json
import re
from typing import Dict, Any, List
from app.services.backend_client import BackendApiClient # 방금 생성한 클라이언트 임포트 (경로 확인)
import aiohttp, asyncpg
import uuid

from app.services.llm_service import LLMService
from app.services.postgresql_service import PostgreSQLService
from app.services.database_client import DatabaseClient
from app.utils.logger import get_logger

logger = get_logger("ChatHandler")

QUEUE_NAME = "llm_task_queue"
AI_USER_NICKNAME = "SLM_Assistant" # AI 응답을 저장할 때 사용할 닉네임

SYSTEM_PROMPT_COT = """You are a highly efficient AI assistant. Your primary goal is to provide accurate and extremely concise answers. Follow these instructions precisely:

1.  **Analyze & Think:** First, understand the user's request and the conversation context. Briefly outline your step-by-step reasoning or analysis within `[THOUGHTS]` and `[/THOUGHTS]` tags. Keep your thoughts as brief as possible.
2.  **Formulate Answer:** Based on your thoughts, construct the final answer.
3.  **Provide Answer:** Present the final answer ONLY within `[ANSWER]` and `[/ANSWER]` tags.
4.  **BE CONCISE:** The final answer within `[ANSWER]` tags MUST be the shortest possible response that directly and completely answers the question. Do NOT include any introductions, apologies, or extra remarks.
5.  **STRICT FORMAT:** You MUST output in this exact format, with nothing before or after: `[THOUGHTS]Your brief thoughts.[/THOUGHTS][ANSWER]Your concise answer.[/ANSWER]`
"""
SYSTEM_PROMPT_SIMPLE = """You are a highly efficient AI assistant. Provide a direct and concise answer to the user's request based on the conversation context. Do not add any introductions, extra remarks, or apologies.
"""

DEFAULT_CHAT_PARAMS = {"max_tokens": 256, "temperature": 0.7} # use_cot는 여기서 관리
DEFAULT_SUMMARY_PARAMS_FOR_HANDLER = {"max_tokens": 512, "temperature": 0.3, "use_cot": False}


def format_context_to_messages(context_summary: str, recent_history: list) -> list:
    """요약본과 최근 대화를 LLM messages 형식으로 변환합니다."""
    messages = []
    if context_summary:
        messages.append({"role": "system", "content": f"Summary of the previous conversation: {context_summary}"})

    for item in recent_history:
        # PG 서비스에서 이미 'user', 'assistant'로 매핑된 role 사용
        role = item.get('role')
        if role in ['user', 'assistant']:
            messages.append({"role": role, "content": item.get('content')})
    return messages


def parse_llm_output_for_answer(full_response: str) -> str:
    """LLM 출력에서 [ANSWER] 태그 안의 내용을 추출합니다."""
    answer_match = re.search(r"\[ANSWER\](.*?)\[/ANSWER\]", full_response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    else:
        logger.warning(f"Could not find [ANSWER] tags. Returning full response: {full_response[:150]}...")
        no_thoughts = re.sub(r"\[THOUGHTS\].*?\[/THOUGHTS\]", "", full_response, flags=re.DOTALL).strip()
        return no_thoughts if no_thoughts else full_response.strip()


async def handle_chat_processing(
        payload: Dict[str, Any],
        llm: LLMService,
        pg: PostgreSQLService,
        redis: DatabaseClient
):
    """'process_chat' 작업을 처리합니다."""
    room_id = payload.get("room_id")
    user_id_str = payload.get("user_id_str") # API에서 전달된 문자열 ID
    user_message = payload.get("user_message")
    request_id = payload.get("request_id", "N/A")

    if not all([room_id, user_id_str, user_message]):
        logger.error(f"ChatHandler: Invalid payload (ReqID: {request_id}): {payload}")
        return

    logger.info(f"ChatHandler: Processing 'process_chat' (ReqID: {request_id}) for room: {room_id}")

    try:
        # 0. 사용자 ID와 AI ID를 PG에서 조회/생성
        user_id_int = await pg.get_or_create_user(user_id_str)
        ai_user_id_int = await pg.get_or_create_user(AI_USER_NICKNAME)

        # A. 사용자 메시지를 PG에 저장 (DB 스키마 요구사항 반영)
        # await pg.add_message(room_id, user_id_int, user_message, 'user', request_id)
        logger.info(f"ChatHandler: User message check (ReqID: {request_id}) to PG.")

        # 1. PG에서 문맥 조회 (사용자 메시지 저장 후 조회)
        summary = await pg.get_summary(room_id)
        recent_history = await pg.get_recent_history(room_id, limit=4)
        history_messages = format_context_to_messages(summary, recent_history)

        # 2. LLM 파라미터 및 CoT 설정
        llm_params_from_payload = payload.get("llm_params", {}).copy()
        use_cot = llm_params_from_payload.pop("use_cot", True) # API에서 온 use_cot 사용, 없으면 True
        final_llm_params = DEFAULT_CHAT_PARAMS.copy()
        final_llm_params.update(llm_params_from_payload)
        system_prompt_to_use = SYSTEM_PROMPT_COT if use_cot else SYSTEM_PROMPT_SIMPLE

        messages_for_llm = [
            {"role": "system", "content": system_prompt_to_use},
            *history_messages,
            {"role": "user", "content": user_message} # 최신 사용자 메시지 추가
        ]
        logger.debug(f"ChatHandler: Messages for LLM (ReqID: {request_id}): {messages_for_llm}")
        logger.debug(f"ChatHandler: LLM params (ReqID: {request_id}): {final_llm_params}, CoT: {use_cot}")

        # 3. LLM 호출
        llm_output_data = await llm.generate_text(messages=messages_for_llm, **final_llm_params)
        full_response = llm_output_data.get("generated_text", "죄송합니다. 답변을 생성하지 못했습니다.")

        # 4. 응답 파싱
        final_answer = parse_llm_output_for_answer(full_response) if use_cot else full_response.strip()

        # 5. SLM 응답을 reception_api를 통해 전송 (기존 PG 저장 주석 처리)
        # await pg.add_message(room_id, ai_user_id_int, final_answer, 'final_ai', request_id)
        # logger.info(f"ChatHandler: Saved AI response (ReqID: {request_id}) to PG.")

        # --- 수정된 코드 ---
        # Redis 작업 데이터에서 callback_url 과 request_id 를 가져와야 합니다.
        # callback_url = task_data['payload'].get('callback_url')
        # request_id = task_data['payload']['request_id']

        # aiohttp 세션을 생성하여 클라이언트 사용 (세션 관리는 앱 전체적으로 고려하는 것이 좋음)
        async with aiohttp.ClientSession() as session:
            client = BackendApiClient(session)
            success = await client.send_ai_response(uuid.UUID(request_id), final_answer)

            if success:
                logger.info(f"ChatHandler: AI response callback sent successfully for ReqID: {request_id}.")
            else:
                logger.error(f"ChatHandler: Failed to send AI response callback for ReqID: {request_id}.")
                # TODO: 콜백 전송 실패 시, 재시도 로직이나 DB에 에러 상태를 기록하는 등의 후처리 필요.

        # 6. 요약 작업 큐에 추가
        summary_llm_params = DEFAULT_SUMMARY_PARAMS_FOR_HANDLER.copy()
        summary_task_payload = {
            "room_id": room_id,
            "request_id": request_id,
            "llm_params": summary_llm_params
        }
        summary_task = {"type": "update_summary", "payload": summary_task_payload}
        await redis.lpush(QUEUE_NAME, json.dumps(summary_task))
        logger.info(f"ChatHandler: Added 'update_summary' task (OrigReqID: {request_id}).")

    except asyncpg.exceptions.ForeignKeyViolationError as fke:
         logger.error(f"ChatHandler: Foreign Key Error (ReqID: {request_id}), Room {room_id} or User {user_id_str} might not exist: {fke}", exc_info=True)
         # 여기서 적절한 실패 처리를 할 수 있습니다 (예: 에러 메시지 저장)
    except Exception as e:
        logger.error(f"ChatHandler: Error processing 'process_chat' (ReqID: {request_id}): {e}", exc_info=True)