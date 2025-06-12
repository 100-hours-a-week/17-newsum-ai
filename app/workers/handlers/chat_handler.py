# ai/app/workers/handlers/chat_handler.py

import json
import re
from typing import Dict, Any, List, Optional
import asyncpg
import uuid
import datetime

from app.services.llm_service import LLMService
from app.services.postgresql_service import PostgreSQLService
from app.services.database_client import DatabaseClient
from app.services.backend_client import BackendApiClient
from transformers import AutoTokenizer

from app.config.settings import Settings
from app.utils.logger import get_logger

settings = Settings()
logger = get_logger(__name__)

AI_USER_NICKNAME = "SLM_Assistant"
MAX_CONTEXT_TOKENS = 4096
RESPONSE_BUDGET = 2048
DEFAULT_CHAT_PARAMS = {"max_tokens": 1024, "temperature": 0.4}
DEFAULT_SUMMARY_PARAMS_FOR_HANDLER = {"max_tokens": 512, "temperature": 0.3}  # summary_handler와 일치

SYSTEM_PROMPT_HEADER = """You are a highly efficient AI assistant. Your primary goal is to provide accurate and extremely concise answers.
**Context Instructions:** Use the 'Summary' for background and 'Recent History' for immediate context."""

SYSTEM_PROMPT_COT = SYSTEM_PROMPT_HEADER + """
Your primary goal is to provide accurate and extremely concise answers. Think step-by-step (the model will naturally use <think> tags if it needs to show its reasoning).
**Critically consider the 'Current Workflow State', 'Current Time', and 'User Information' provided in the '추가 중요 정보' section to tailor your response and actions.**
If the user's request implies a change to the workflow state, clearly state your reasoning and the proposed new state within your <think> tags using the format [STATE_UPDATE: {"status": "new_status", "task_details": {...}}].
After your <think> tags, provide ONLY the final, direct answer to the user. If the answer requires explaining your thought process, you may articulate it clearly as part of the answer, but do not use <think> tags in the final answer.
"""

SYSTEM_PROMPT_SIMPLE = SYSTEM_PROMPT_HEADER + """
**Task Instructions:** Provide a direct and concise answer to the user's request based on the conversation context. Do not add any introductions, extra remarks, or apologies.
Pay attention to 'Current Workflow State', 'Current Time', and 'User Information' if provided. If the answer requires explaining your thought process, you may articulate it clearly.
"""


def count_tokens(text: str, tokenizer: Optional[AutoTokenizer]) -> int:
    if tokenizer and text:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            logger.warning(f"Token count failed for text (len {len(text)}), falling back to split().")
            return len(text.split())
    elif text:
        logger.warning("Tokenizer not available for token count. Using word count if text exists.")
        return len(text.split())
    return 0


def build_and_truncate_context(
        system_prompt: str,
        summary: str,
        recent_history: List[Dict[str, str]],
        user_message: str,
        tokenizer: Optional[AutoTokenizer],
        #workflow_state_str: str,
        current_time_str: str,
        main_user_info: str,
        state_field_explanations_str: str
) -> List[Dict[str, str]]:
    messages_for_llm = []
    messages_for_llm.append({"role": "system", "content": system_prompt})
    # current_tokens = count_tokens(system_prompt, tokenizer) # [주석 처리]
    # - 현재 워크플로우 상태: {workflow_state_str}
    additional_context_content = f"""**추가 중요 정보:**
- 현재 시간: {current_time_str}
- 사용자 관련 참고사항: {main_user_info}"""
    # - 워크플로우 상태 필드 가이드: {state_field_explanations_str}
    messages_for_llm.append({"role": "system", "content": additional_context_content})
    # current_tokens += count_tokens(additional_context_content, tokenizer) # [주석 처리]

    summary_prefix = "이전 대화 요약: "
    full_summary_text = f"{summary_prefix}{summary}" if summary else ""
    summary_tokens = count_tokens(full_summary_text, tokenizer)
    user_message_tokens = count_tokens(user_message, tokenizer)

    # [주석 처리] 아래 라인은 current_tokens에 의존하므로 함께 주석 처리합니다.
    # available_for_history = MAX_CONTEXT_TOKENS - (
    #         current_tokens + summary_tokens + user_message_tokens + RESPONSE_BUDGET
    # )

    # [임시 조치] 위 available_for_history가 없으므로 임시로 값을 할당하거나,
    # 아래 로직을 수정해야 합니다. 우선 토큰 계산을 비활성화하는 데 중점을 둡니다.
    # 이 값은 실제 토큰 계산 로직이 복구될 때 함께 수정되어야 합니다.
    available_for_history = MAX_CONTEXT_TOKENS - (
                summary_tokens + user_message_tokens + RESPONSE_BUDGET + 2048)  # 임시 값 (2048은 시스템 프롬프트용)

    added_history = []
    if available_for_history > 0 and recent_history:
        temp_history_tokens = 0
        for msg in reversed(recent_history):
            msg_content = msg.get('content', '')
            msg_role = msg.get('role')
            if not msg_content or not msg_role:
                continue
            msg_tokens = count_tokens(msg_content, tokenizer)
            if temp_history_tokens + msg_tokens <= available_for_history:
                added_history.insert(0, {"role": msg_role, "content": msg_content})
                temp_history_tokens += msg_tokens
            else:
                logger.info(
                    f"History token limit reached ({temp_history_tokens}/{available_for_history}), truncating.")
                break

    if summary:
        messages_for_llm.append({"role": "system", "content": full_summary_text})

    messages_for_llm.extend(added_history)
    messages_for_llm.append({"role": "user", "content": user_message})

    total_input_tokens = sum(count_tokens(m["content"], tokenizer) for m in messages_for_llm)
    logger.debug(
        f"Built context with ~{total_input_tokens} input tokens (MAX: {MAX_CONTEXT_TOKENS}, ResponseBudget: {RESPONSE_BUDGET}).")
    return messages_for_llm


def parse_llm_output_for_answer_and_thoughts(full_response: str) -> Dict[str, str]:
    thoughts = []
    answer_parts = []
    think_pattern_re = r"<think>(.*?)</think>"

    last_end = 0
    for match in re.finditer(think_pattern_re, full_response, re.DOTALL | re.IGNORECASE):
        answer_parts.append(full_response[last_end:match.start()].strip())
        thoughts.append(match.group(1).strip())
        last_end = match.end()

    answer_parts.append(full_response[last_end:].strip())

    final_thoughts = "\n---\n".join(filter(None, thoughts))
    final_answer = "\n".join(filter(None, answer_parts)).strip()

    if not final_answer and final_thoughts and not list(
            re.finditer(think_pattern_re, full_response, re.DOTALL | re.IGNORECASE)):
        final_answer = full_response.strip()

    return {"thoughts": final_thoughts, "answer": final_answer}


def should_use_cot(user_message: str, params: Dict, tokenizer: Optional[AutoTokenizer]) -> bool:
    return params.get("use_cot", True)


def get_state_field_explanations_for_prompt(workflow_state_dict: Optional[Dict[str, Any]]) -> str:
    base_explanations = """
    - `meta.work_id`: 현재 작업의 고유 ID (워크플로우 전체 ID).
    - `meta.workflow_status`: 노드별 작업 진행 상태.
    - `query.original_query`: 사용자 원본 질문.
    - `report.report_content`: 생성된 보고서 내용(HTML).
    - `idea.final_comic_ideas`: 최종 만화 아이디어 목록.
    (세부 필드는 state_v2.py 참조)
    """
    return base_explanations


async def handle_chat_processing(
        payload: Dict[str, Any],
        llm: LLMService,
        pg: PostgreSQLService,
        redis: DatabaseClient,
        tokenizer: Optional[AutoTokenizer],
        backend_client: BackendApiClient
):
    room_id_str = payload.get("room_id")
    user_id_str = payload.get("user_id_str")
    user_message = payload.get("user_message")
    request_id = payload.get("request_id")  # 개별 메시지의 고유 ID
    work_id_str = payload.get("work_id")  # 워크플로우 상태 조회를 위한 ID (state_v2의 work_id에 해당할 수 있음)

    # request_id를 trace_id로 사용 (개별 요청 추적)
    # work_id_str도 로깅에 포함하여 워크플로우 컨텍스트 추적
    log_extra = {"request_id": request_id, "room_id": room_id_str, "work_id_for_wf_state": work_id_str}

    # 필수 필드: room_id, user_id_str, user_message, request_id, work_id
    if not all([room_id_str, user_id_str, user_message, request_id, work_id_str]):
        # 로그에서 어떤 필드가 누락되었는지 명시적으로 보여주는 것이 좋음
        missing_fields = [
            field_name for field_name, field_value in {
                "room_id": room_id_str, "user_id_str": user_id_str,
                "user_message": user_message, "request_id": request_id,
                "work_id": work_id_str
            }.items() if not field_value
        ]
        logger.error(f"Invalid payload: Missing required fields: {missing_fields}. Payload: {payload}", extra=log_extra)
        return

    try:
        room_id_for_pg = int(room_id_str)
    except ValueError:
        logger.error(f"room_id '{room_id_str}' is not a valid integer.", extra=log_extra)
        return
    try:
        # work_id_str (워크플로우 상태 조회용 ID)는 UUID여야 함 (pg.get_workflow_state 스펙)
        workflow_context_id_uuid = uuid.UUID(work_id_str)
    except ValueError:
        logger.error(f"work_id '{work_id_str}' (for workflow state) is not a valid UUID.", extra=log_extra)
        return

    logger.info(f"Processing 'process_chat' for room: {room_id_str}, wf_id: {work_id_str}", extra=log_extra)

    try:
        user_id_int = await pg.get_or_create_user(user_id_str)
        summary = await pg.get_summary(str(room_id_for_pg))
        recent_history = await pg.get_recent_history(str(room_id_for_pg), limit=10)

        # 워크플로우 상태는 work_id_str (UUID로 변환된 workflow_context_id_uuid)를 사용해 조회
        workflow_state_dict = await pg.get_workflow_state(workflow_context_id_uuid, room_id=room_id_for_pg)
        workflow_state_str = json.dumps(workflow_state_dict, ensure_ascii=False,
                                        indent=2) if workflow_state_dict else "현재 워크플로우 상태 정보 없음"

        current_time_str = datetime.datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
        main_user_info = "이 대화의 주요 사용자는 한국인입니다. 사용자의 의도를 파악하여 워크플로우 상태 변경이 필요하다면, <think> 태그 내에 [STATE_UPDATE: {...}] 형식으로 제안해주세요."
        state_field_explanations = get_state_field_explanations_for_prompt(workflow_state_dict)

        llm_params_from_payload = payload.get("llm_params", {}).copy()
        use_cot_flag = should_use_cot(user_message, llm_params_from_payload, tokenizer)
        system_prompt_to_use = SYSTEM_PROMPT_COT if use_cot_flag else SYSTEM_PROMPT_SIMPLE

        final_llm_params = DEFAULT_CHAT_PARAMS.copy()
        final_llm_params.update(llm_params_from_payload)
        if 'use_cot' in final_llm_params:
            del final_llm_params['use_cot']

        messages_for_llm = build_and_truncate_context(
            system_prompt_to_use, summary, recent_history, user_message, tokenizer,
            # workflow_state_str,
            current_time_str, main_user_info, state_field_explanations
        )

        logger.debug(f"LLM params for call: {final_llm_params}, CoT: {use_cot_flag}", extra=log_extra)

        # LLM 호출 시 request_id는 개별 메시지 ID 사용
        llm_output_data = await llm.generate_text(
            messages=messages_for_llm, request_id=request_id, **final_llm_params
        )

        final_answer = "죄송합니다. 현재 질문에 대한 답변을 드리기 어렵습니다."
        llm_thoughts = ""

        if "error" in llm_output_data or not llm_output_data.get("generated_text"):
            logger.error(f"LLM generation failed. Response: {llm_output_data}", extra=log_extra)
        else:
            full_response = llm_output_data.get("generated_text", "")
            logger.debug(f"Full LLM response (first 500 chars): {full_response[:500]}...", extra=log_extra)
            parsed_output = parse_llm_output_for_answer_and_thoughts(full_response)
            final_answer = parsed_output["answer"]
            llm_thoughts = parsed_output["thoughts"]

            if not final_answer and llm_thoughts:
                logger.warning(f"LLM provided thoughts but no clear answer. Thoughts: {llm_thoughts[:300]}...",
                               extra=log_extra)
                final_answer = "답변을 준비하는 데 시간이 조금 더 필요할 것 같습니다. 잠시 후 다시 시도해 주시거나, 다른 질문을 해주세요."
            elif not final_answer and not llm_thoughts:
                final_answer = "응답 내용이 없습니다. 다시 질문해주시겠어요?"

            if llm_thoughts:
                logger.info(f"LLM thoughts (first 1000 chars): {llm_thoughts[:1000]}...", extra=log_extra)

                # state_v2.py의 MetaSection.llm_think_traces에 로그 추가 고려
                # (예: pg_service에 add_llm_think_log(workflow_context_id_uuid, node_name, request_id, llm_thoughts) 함수 추가)

                state_update_match = re.search(r"\[STATE_UPDATE:\s*(\{.*?\})\s*\]", llm_thoughts,
                                               re.DOTALL | re.IGNORECASE)
                if state_update_match:
                    try:
                        state_update_json_str = state_update_match.group(1)
                        proposed_task_details = json.loads(state_update_json_str)  # LLM이 제안한 task_details 부분

                        status_for_pg_update = proposed_task_details.pop("new_status_for_pg",
                                                                         "LLM_SUGGESTED_STATE_CHANGE")

                        if isinstance(proposed_task_details, dict):
                            logger.info(
                                f"LLM proposed state update (task_details). Status: '{status_for_pg_update}', Details: {proposed_task_details}",
                                extra=log_extra)

                            # 상태 수정은 workflow_context_id_uuid (기존 work_id) 기준으로 수행
                            await pg.update_workflow_state(workflow_context_id_uuid, room_id_for_pg,
                                                           status_for_pg_update, proposed_task_details)
                            logger.info(f"Workflow state updated by LLM for wf_id: {work_id_str}.", extra=log_extra)
                            final_answer += f"\n\n(안내: 워크플로우 상태가 '{status_for_pg_update}'(으)로 변경 제안/반영되었습니다.)"
                        else:
                            logger.warning(
                                f"LLM proposed task_details is not a dictionary. Update skipped. Proposal: {proposed_task_details}",
                                extra=log_extra)

                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to decode state update JSON from LLM thoughts: {state_update_match.group(1)}",
                            extra=log_extra)
                    except Exception as e_state:
                        logger.error(f"Error processing LLM state update for wf_id {work_id_str}: {e_state}",
                                     exc_info=True, extra=log_extra)

        callback_id_to_send: Any = request_id  # 콜백 ID는 개별 메시지 ID인 request_id 사용
        try:
            # request_id가 UUID 형식이면 UUID 객체로 변환하여 전달
            callback_id_uuid = uuid.UUID(request_id)
            callback_id_to_send = callback_id_uuid
        except ValueError:
            logger.warning(f"request_id '{request_id}' for callback is not a valid UUID. Sending as string.",
                           extra=log_extra)

        callback_success: Optional[bool] = None
        try:
            callback_success = await backend_client.streamlit_send_ai_response(callback_id_to_send, final_answer)
            if callback_success:
                logger.info(f"AI response callback sent successfully.", extra=log_extra)
            else:
                logger.error(f"Failed to send AI response callback (returned False). Saving to PG fallback.",
                             extra=log_extra)
                ai_user_id_int = await pg.get_or_create_user(AI_USER_NICKNAME)
                # 메시지 저장 시 request_id 사용
                await pg.add_message(str(room_id_for_pg), ai_user_id_int, f"(Callback Failed) {final_answer}",
                                     'final_ai', request_id)
        except Exception as e_callback:
            logger.error(f"Error sending AI response callback: {e_callback}. Orig request_id: {request_id}",
                         exc_info=True, extra=log_extra)
            if callback_success is None or not callback_success:
                try:
                    ai_user_id_int = await pg.get_or_create_user(AI_USER_NICKNAME)
                    await pg.add_message(str(room_id_for_pg), ai_user_id_int,
                                         f"(Callback Exception/Failed) {final_answer}", 'final_ai', request_id)
                except Exception as e_pg_fallback:
                    logger.error(f"Failed to save AI message to PG as fallback after callback error: {e_pg_fallback}",
                                 extra=log_extra)

        summary_llm_params = DEFAULT_SUMMARY_PARAMS_FOR_HANDLER.copy()
        summary_task_payload = {
            "room_id": str(room_id_for_pg),
            "request_id": request_id,  # summary_handler 에도 request_id 전달 (추적용)
            "work_id": work_id_str,  # summary_handler 에 work_id (워크플로우 ID) 전달
            "llm_params": summary_llm_params
        }
        # state_v2.py에 work_id가 정의되었으므로, 가능하다면 work_id를 생성하거나 받아서 요약 페이로드에 포함하는 것이 좋음.
        # 만약 work_id가 state_v2.MetaSection.work_id와 같다면, work_id를 work_id로 사용.
        # summary_task_payload["work_id"] = work_id_str # work_id를 work_id로 간주하여 전달

        summary_task = {"type": "update_summary", "payload": summary_task_payload}
        await redis.lpush(getattr(settings, 'CHAT_TASK_QUEUE_NAME', "chat_task_queue"), json.dumps(summary_task))
        logger.info(f"Added 'update_summary' task to queue for room {room_id_str}.", extra=log_extra)

    except asyncpg.exceptions.ForeignKeyViolationError as fke:
        logger.error(f"Foreign Key Error for room {room_id_str}: {fke}", exc_info=True, extra=log_extra)
    except ValueError as ve:  # int(room_id_str) 또는 uuid.UUID(work_id_str) 등에서 발생 가능
        logger.error(f"Value Error processing request: {ve}", exc_info=True, extra=log_extra)
    except Exception as e:
        logger.error(f"Error processing 'process_chat' for room {room_id_str}: {e}", exc_info=True, extra=log_extra)