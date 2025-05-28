# ai/app/workers/handlers/chat_handler.py

import json
import re
from typing import Dict, Any, List, Optional
import asyncpg
import uuid  # UUID 변환을 위해 임포트

from app.services.llm_service import LLMService
from app.services.postgresql_service import PostgreSQLService
from app.services.database_client import DatabaseClient
from app.services.backend_client import BackendApiClient
from transformers import AutoTokenizer  # 타입 힌트

from app.config.settings import Settings
from app.utils.logger import get_logger

settings = Settings()  # settings.py를 통해 설정값 로드
logger = get_logger("ChatHandler")

# chat_worker.py 에서 QUEUE_NAME을 설정에서 가져오므로, 여기서 중복 정의 제거 가능
# QUEUE_NAME = "chat_task_queue"
AI_USER_NICKNAME = "SLM_Assistant"  # 설정 파일로 옮기는 것 고려
MAX_CONTEXT_TOKENS = 4096  # 설정 파일로 옮기는 것 고려
RESPONSE_BUDGET = 512  # 설정 파일로 옮기는 것 고려

# --- 영문 프롬프트 (안전 지침 없음) ---
SYSTEM_PROMPT_HEADER = """You are a highly efficient AI assistant. Your primary goal is to provide accurate and extremely concise answers.
**Context Instructions:** Use the 'Summary' for background and 'Recent History' for immediate context."""

SYSTEM_PROMPT_COT = SYSTEM_PROMPT_HEADER + """Your primary goal is to provide accurate and extremely concise answers. Think step-by-step (the model will naturally use <think> tags if it needs to show its reasoning).
 Provide ONLY the final, direct answer to the user without any extra conversational fluff, introductions, or self-corrections.
"""

SYSTEM_PROMPT_SIMPLE = SYSTEM_PROMPT_HEADER + """
**Task Instructions:** Provide a direct and concise answer to the user's request based on the conversation context. Do not add any introductions, extra remarks, or apologies.
"""

DEFAULT_CHAT_PARAMS = {"max_tokens": 512, "temperature": 0.4}  # API가 설정한 값으로 덮어쓰여짐
DEFAULT_SUMMARY_PARAMS_FOR_HANDLER = {"max_tokens": 512, "temperature": 0.3}


def count_tokens(text: str, tokenizer: Optional[AutoTokenizer]) -> int:
    if tokenizer and text:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            logger.warning(f"Token count failed for text (len {len(text)}), falling back to split().")
            return len(text.split())
    elif text:
        logger.warning("Tokenizer not available for token count, using split().")
        return len(text.split())
    return 0


def build_and_truncate_context(
        system_prompt: str,
        summary: str,
        recent_history: List[Dict[str, str]],
        user_message: str,
        tokenizer: Optional[AutoTokenizer]
) -> List[Dict[str, str]]:
    messages_for_llm = []  # 최종 LLM 메시지 목록

    # 1. 시스템 프롬프트 추가
    messages_for_llm.append({"role": "system", "content": system_prompt})
    current_tokens = count_tokens(system_prompt, tokenizer)

    # 2. 요약 추가 (있다면)
    summary_prefix = "Summary of the previous conversation: "
    full_summary_text = f"{summary_prefix}{summary}" if summary else ""
    summary_tokens = count_tokens(full_summary_text, tokenizer)

    # 3. 사용자 메시지 추가 (항상 포함)
    user_message_tokens = count_tokens(user_message, tokenizer)

    # 시스템, 요약, 사용자 메시지, 응답 예산을 제외한 가용 토큰 계산
    available_for_history = MAX_CONTEXT_TOKENS - (
                current_tokens + summary_tokens + user_message_tokens + RESPONSE_BUDGET)

    added_history = []
    if available_for_history > 0 and recent_history:
        temp_history_tokens = 0
        for msg in reversed(recent_history):  # 최신 메시지부터
            msg_content = msg.get('content', '')
            msg_role = msg.get('role')
            if not msg_content or not msg_role:  # 유효하지 않은 메시지 스킵
                continue

            msg_tokens = count_tokens(msg_content, tokenizer)
            if temp_history_tokens + msg_tokens <= available_for_history:
                added_history.insert(0, {"role": msg_role, "content": msg_content})  # 시간 순서 유지를 위해 앞에 삽입
                temp_history_tokens += msg_tokens
            else:
                logger.info(
                    f"History token limit reached ({temp_history_tokens}/{available_for_history}), truncating older history.")
                break

    if summary:  # 요약은 시스템 메시지 다음에 추가
        messages_for_llm.append({"role": "system", "content": full_summary_text})

    messages_for_llm.extend(added_history)  # 절삭된 최근 대화 추가
    messages_for_llm.append({"role": "user", "content": user_message})  # 사용자 메시지 마지막에 추가

    total_estimated_tokens = count_tokens(system_prompt, tokenizer) + \
                             count_tokens(full_summary_text, tokenizer) + \
                             sum(count_tokens(m["content"], tokenizer) for m in added_history) + \
                             user_message_tokens
    logger.debug(f"Built context with ~{total_estimated_tokens} input tokens (excluding response budget).")
    return messages_for_llm


def parse_llm_output_for_answer(full_response: str) -> str:
    # 1. <think>...</think> 블록 추출 (로깅 또는 다른 용도로 활용 가능)
    think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL | re.IGNORECASE)
    if think_match:
        thoughts = think_match.group(1).strip()
        logger.debug(f"LLM Thoughts: {thoughts[:500]}...") # 예시 로깅
        # <think>...</think> 블록을 전체 응답에서 제거
        answer_part = full_response.replace(think_match.group(0), "").strip()
        return answer_part # <think> 태그 이후의 내용을 최종 답변으로 간주
    else:
        # <think> 태그가 없는 경우, 전체 응답을 그대로 반환
        logger.debug("No <think> tags found, returning full response.")
        return full_response.strip()


def should_use_cot(user_message: str, params: Dict, tokenizer: Optional[AutoTokenizer]) -> bool:
    if "use_cot" in params:  # API에서 명시적으로 지정하면 그 값을 따름
        return params.get("use_cot", True)

    # 휴리스틱: 추론 단어 포함 또는 특정 토큰 수 이상이면 CoT 사용
    reasoning_words = {"why", "how", "explain", "describe", "what if", "tell me about"}
    if any(word in user_message.lower() for word in reasoning_words) or \
            count_tokens(user_message, tokenizer) > 30:  # 예: 30 토큰 이상
        return True
    return False


async def handle_chat_processing(
        payload: Dict[str, Any],
        llm: LLMService,  # LLMService 인스턴스
        pg: PostgreSQLService,
        redis: DatabaseClient,
        tokenizer: Optional[AutoTokenizer],  # 주입된 토크나이저
        backend_client: BackendApiClient  # 주입된 백엔드 클라이언트
):
    """'process_chat' 작업을 처리합니다."""
    room_id = payload.get("room_id")
    user_id_str = payload.get("user_id_str")
    user_message = payload.get("user_message")
    request_id = payload.get("request_id", "N/A")  # Redis 페이로드에서 request_id 가져옴

    if not all([room_id, user_id_str, user_message, request_id]):
        logger.error(f"ChatHandler: Invalid payload for 'process_chat' (ReqID: {request_id}): {payload}")
        return

    logger.info(f"ChatHandler: Processing 'process_chat' (ReqID: {request_id}) for room: {room_id}")

    try:
        # 1. 사용자 ID 조회 및 사용자 메시지 저장
        user_id_int = await pg.get_or_create_user(user_id_str)
        #await pg.add_message(room_id, user_id_int, user_message, 'user', request_id)
        logger.info(f"ChatHandler: Saved User message (ReqID: {request_id}) to PG for room {room_id}.")

        # 2. 문맥 조회
        summary = await pg.get_summary(room_id)
        recent_history = await pg.get_recent_history(room_id, limit=10)  # 토큰 절삭을 위해 충분히 가져옴

        # 3. LLM 파라미터 및 CoT 설정
        llm_params_from_payload = payload.get("llm_params", {}).copy()
        use_cot_flag = should_use_cot(user_message, llm_params_from_payload, tokenizer)
        system_prompt_to_use = SYSTEM_PROMPT_COT if use_cot_flag else SYSTEM_PROMPT_SIMPLE

        final_llm_params = DEFAULT_CHAT_PARAMS.copy()  # 기본값으로 시작
        final_llm_params.update(llm_params_from_payload)  # API에서 온 값으로 덮어쓰기
        if 'use_cot' in final_llm_params:  # use_cot는 프롬프트 선택에만 사용
            del final_llm_params['use_cot']

        # 4. 동적 컨텍스트 구축
        messages_for_llm = build_and_truncate_context(
            system_prompt_to_use, summary, recent_history, user_message, tokenizer
        )
        logger.debug(
            f"ChatHandler: Messages for LLM (ReqID: {request_id}, Count: {len(messages_for_llm)}): {str(messages_for_llm)[:300]}...")
        logger.debug(f"ChatHandler: LLM params for call (ReqID: {request_id}): {final_llm_params}, CoT: {use_cot_flag}")

        # 5. LLMService 호출
        llm_output_data = await llm.generate_text(
            messages=messages_for_llm,
            request_id=request_id,  # LLMService에 request_id 전달
            **final_llm_params  # max_tokens 등 포함
        )

        if "error" in llm_output_data or not llm_output_data.get("generated_text"):
            logger.error(
                f"ChatHandler: LLM generation failed or returned empty (ReqID: {request_id}). Response: {llm_output_data}")
            final_answer = "Sorry, I encountered an issue generating a response."
            # 콜백 전송 로직으로 바로 넘어감 (PG에는 저장하지 않음, 콜백 실패 시 저장 고려)
        else:
            full_response = llm_output_data.get("generated_text", "")
            logger.debug(f"ChatHandler: Full LLM response (ReqID: {request_id}): {full_response[:200]}...")
            # 6. 응답 파싱
            final_answer = parse_llm_output_for_answer(full_response) if use_cot_flag else full_response.strip()

        # 7. Backend API로 응답 전송
        try:
            req_uuid = uuid.UUID(request_id)  # request_id를 UUID 객체로 변환
            callback_success = await backend_client.send_ai_response(req_uuid, final_answer)

            if callback_success:
                logger.info(f"ChatHandler: AI response callback sent successfully for ReqID: {request_id}.")
                # 콜백 성공 시, AI 메시지는 백엔드가 저장한다고 가정 (여기서 PG 저장 안 함)
            else:
                logger.error(
                    f"ChatHandler: Failed to send AI response callback for ReqID: {request_id}. Saving to PG as fallback.")
                # 콜백 실패 시, PG에 에러 상태 또는 AI 메시지 저장
                ai_user_id_int = await pg.get_or_create_user(AI_USER_NICKNAME)
                await pg.add_message(room_id, ai_user_id_int, f"(Callback Failed) {final_answer}", 'final_ai',
                                     request_id)

        except ValueError:  # request_id가 유효한 UUID가 아닐 경우
            logger.error(
                f"ChatHandler: Invalid request_id format for UUID conversion: {request_id}. Cannot send callback. Saving to PG.")
            ai_user_id_int = await pg.get_or_create_user(AI_USER_NICKNAME)
            await pg.add_message(room_id, ai_user_id_int, f"(Invalid ReqID for Callback) {final_answer}", 'final_ai',
                                 request_id)

        # 8. 다음 'update_summary' 작업을 Redis 큐에 추가 (콜백 성공 여부와 관계없이 수행)
        summary_llm_params = DEFAULT_SUMMARY_PARAMS_FOR_HANDLER.copy()
        summary_task_payload = {
            "room_id": room_id,
            "request_id": request_id,
            "llm_params": summary_llm_params
        }
        summary_task = {"type": "update_summary", "payload": summary_task_payload}

        await redis.lpush(getattr(Settings, 'CHAT_TASK_QUEUE_NAME', "chat_task_queue"),
                          json.dumps(summary_task))
        logger.info(f"ChatHandler: Added 'update_summary' task to queue for room {room_id} (OrigReqID: {request_id}).")

    except asyncpg.exceptions.ForeignKeyViolationError as fke:
        logger.error(
            f"ChatHandler: Foreign Key Error (ReqID: {request_id}) for room {room_id}, user {user_id_str}: {fke}",
            exc_info=True)
    except Exception as e:
        logger.error(f"ChatHandler: Error processing 'process_chat' (ReqID: {request_id}) for room {room_id}: {e}",
                     exc_info=True)