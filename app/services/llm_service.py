# # ai/app/services/llm_service.py
#
# import httpx
# import json
# import asyncio
# import re
# from typing import Optional, Dict, Any, List, Tuple
# from pathlib import Path
# from app.config.settings import Settings
# from app.utils.logger import get_logger
#
# settings = Settings()  # settings.py를 통해 설정값 로드
# logger = get_logger("LLMService")
#
# from transformers import AutoTokenizer
# # VLLM 최대 토큰 예산 설정 (settings.py에서 가져오거나 기본값 설정)
# VLLM_MAX_TOKEN_BUDGET = getattr(settings, 'VLLM_MAX_TOKEN_BUDGET', 12000)
#
# class LLMService:
#     """
#     LLM API 서버와 상호 작용하는 서비스 (토큰 예산 관리 기능 포함).
#     """
#     DEFAULT_TIMEOUT = 60.0
#
#     def __init__(
#             self,
#             tokenizer: AutoTokenizer,  # 토크나이저 주입
#             endpoint: Optional[str] = None,
#             timeout: Optional[float] = None,
#             max_token_budget: int = VLLM_MAX_TOKEN_BUDGET
#     ):
#         self.tokenizer = tokenizer
#         self.endpoint = endpoint or settings.LLM_API_ENDPOINT
#         self.timeout = timeout if timeout is not None else float(settings.LLM_API_TIMEOUT)
#
#         if not self.endpoint:
#             logger.error("LLM API 엔드포인트가 설정되지 않았습니다.")
#             raise ValueError("LLM API endpoint must be provided.")
#         if not self.tokenizer:  # 토크나이저 존재 유무 확인
#             logger.error("토크나이저가 LLMService에 제공되지 않았습니다.")
#             raise ValueError("Tokenizer must be provided to LLMService.")
#
#         try:
#             self.client = httpx.AsyncClient(timeout=self.timeout)
#         except Exception as e:  # httpx 클라이언트 초기화 실패 시 로깅 및 예외 발생
#             logger.error(f"httpx.AsyncClient 초기화 실패: {e}", exc_info=True)
#             self.client = None  # 명시적으로 None 할당
#             raise ConnectionError(f"Failed to initialize httpx client: {e}") from e
#
#         self.max_budget = max_token_budget
#         self.available_tokens = max_token_budget
#         self.lock = asyncio.Lock()  # Lock은 Condition 객체 내에 이미 포함되어 있어 별도 Lock 불필요
#         self.condition = asyncio.Condition()  # Lock 인자 제거, 내부적으로 Lock 생성
#
#         logger.info(
#             f"LLMService 초기화 완료. Endpoint: {self.endpoint}, Max Token Budget: {self.max_budget}, Timeout: {self.timeout}s")
#
#     def _estimate_tokens(self, messages: List[Dict[str, str]], max_new_tokens_requested: int) -> int:
#         """
#         요청의 예상 토큰 수를 계산합니다 (입력 + 요청된 최대 새 토큰).
#         """
#         input_tokens = 0
#         try:
#             for msg in messages:
#                 content = msg.get("content")
#                 if content and isinstance(content, str):  # content 타입 체크 추가
#                     input_tokens += len(self.tokenizer.encode(content, add_special_tokens=False))
#                 elif content is not None:  # content가 있지만 문자열이 아닌 경우 경고
#                     logger.warning(f"Token estimation: Non-string content found in message: {type(content)}")
#         except Exception as e:
#             logger.warning(f"토큰 수 계산 중 오류 (폴백 사용): {e}")
#             input_tokens = sum(len(str(m.get("content", "")).split()) for m in messages) * 2
#
#         return input_tokens + max_new_tokens_requested  # API에서 요청한 max_tokens 사용
#
#     async def _acquire_tokens(self, required_tokens: int, request_id: str = "N/A"):
#         """
#         필요한 토큰만큼 예산을 확보합니다. 부족하면 대기합니다.
#         """
#         if required_tokens <= 0:  # 0 이하 토큰 요청 방지
#             logger.warning(f"ReqID: {request_id} - 유효하지 않은 토큰 요청 값: {required_tokens}")
#             return
#
#         if required_tokens > self.max_budget:
#             logger.error(f"ReqID: {request_id} - 요청 토큰({required_tokens})이 시스템 최대 예산({self.max_budget})을 초과합니다. 요청 거부.")
#             raise ValueError(f"Requested tokens ({required_tokens}) exceed system maximum budget ({self.max_budget}).")
#
#         async with self.condition:  # Condition 객체의 Lock을 사용
#             while self.available_tokens < required_tokens:
#                 logger.info(
#                     f"ReqID: {request_id} - 토큰 부족. 대기 시작. (요청: {required_tokens}, 사용 가능: {self.available_tokens}, 최대: {self.max_budget})")
#                 await self.condition.wait()
#
#             self.available_tokens -= required_tokens
#             logger.info(f"ReqID: {request_id} - 토큰 확보 성공. (요청: {required_tokens}, 남은 예산: {self.available_tokens})")
#
#     async def _release_tokens(self, released_tokens: int, request_id: str = "N/A"):
#         """
#         사용한 토큰만큼 예산을 반환하고 대기 중인 작업을 깨웁니다.
#         """
#         if released_tokens <= 0:  # 0 이하 토큰 반환 방지
#             logger.warning(f"ReqID: {request_id} - 유효하지 않은 토큰 반환 값: {released_tokens}")
#             return
#
#         async with self.condition:  # Condition 객체의 Lock을 사용
#             self.available_tokens += released_tokens
#             if self.available_tokens > self.max_budget:  # 만약을 위해 최대 예산 초과 방지
#                 logger.warning(
#                     f"ReqID: {request_id} - 토큰 반환 후 사용 가능 토큰({self.available_tokens})이 최대 예산({self.max_budget})을 초과. 조정됨.")
#                 self.available_tokens = self.max_budget
#
#             logger.info(f"ReqID: {request_id} - 토큰 반환. (반환: {released_tokens}, 현재 예산: {self.available_tokens})")
#             self.condition.notify_all()
#
#     def _extract_generated_text(self, response_data: Dict[str, Any]) -> Optional[str]:
#         if not isinstance(response_data, dict):
#             logger.warning(f"텍스트 추출 실패: 응답 데이터가 딕셔너리 타입이 아님 (Type: {type(response_data)})")
#             return None
#         try:
#             if "choices" in response_data and isinstance(response_data["choices"], list) and response_data["choices"]:
#                 first_choice = response_data["choices"][0]
#                 if isinstance(first_choice, dict):
#                     if "message" in first_choice and isinstance(first_choice["message"], dict) and "content" in \
#                             first_choice["message"]:
#                         return first_choice["message"]["content"]
#                     if "text" in first_choice and isinstance(first_choice["text"], str):
#                         return first_choice["text"]
#             if "generated_text" in response_data and isinstance(response_data["generated_text"], str):
#                 return response_data["generated_text"]
#             logger.warning(f"LLM 응답에서 생성된 텍스트를 알려진 패턴으로 추출할 수 없습니다. 응답 데이터 일부: {str(response_data)[:200]}...")
#             return None
#         except (KeyError, IndexError, TypeError) as e:
#             logger.error(f"텍스트 추출 중 오류 발생: {e}. 응답 데이터 일부: {str(response_data)[:200]}...", exc_info=True)
#             return None
#
#     def _extract_main_llm_output(self, response_data: Dict[str, Any]) -> Optional[str]:
#         """LLM 응답 구조에서 주요 생성 텍스트를 추출합니다. (기존 _extract_generated_text 역할)"""
#         if not isinstance(response_data, dict):
#             logger.warning(f"텍스트 추출 실패: 응답 데이터가 딕셔너리 타입이 아님 (Type: {type(response_data)})")
#             return None
#         try:
#             # (기존 _extract_generated_text의 다양한 추출 로직 유지)
#             if "choices" in response_data and isinstance(response_data["choices"], list) and response_data["choices"]:
#                 first_choice = response_data["choices"][0]
#                 if isinstance(first_choice, dict):
#                     if "message" in first_choice and isinstance(first_choice["message"], dict) and "content" in \
#                             first_choice["message"]:
#                         return first_choice["message"]["content"]
#                     if "text" in first_choice and isinstance(first_choice["text"], str):  # 일부 모델은 text 필드 사용
#                         return first_choice["text"]
#             if "generated_text" in response_data and isinstance(response_data["generated_text"],
#                                                                 str):  # vLLM 등 일부 OpenAI 호환 서버
#                 return response_data["generated_text"]
#             logger.warning(f"LLM 응답에서 생성된 텍스트를 알려진 패턴으로 추출할 수 없습니다. 응답 데이터 일부: {str(response_data)[:200]}...")
#             return None
#         except (KeyError, IndexError, TypeError) as e:
#             logger.error(f"텍스트 추출 중 오류 발생: {e}. 응답 데이터 일부: {str(response_data)[:200]}...", exc_info=True)
#             return None
#
#     def _extract_and_remove_think_tags(self, text: Optional[str]) -> Tuple[str, Optional[str]]:
#         """
#         주어진 텍스트에서 모든 <think> 태그의 내용을 추출하고, 원본 텍스트에서는 해당 태그들을 제거합니다.
#         반환값: (정리된 텍스트, 추출된 생각들의 연결된 문자열 또는 None)
#         """
#         if not text:
#             return "", None  # 원본 텍스트가 비어있으면 빈 문자열과 None 반환
#
#         think_contents = []
#         # <think>...</think> 패턴을 찾습니다. (대소문자 구분 없음, 여러 줄 가능)
#         # (.*?)는 태그 내부의 내용을 비탐욕적(non-greedy)으로 찾습니다.
#         pattern = r"<think>(.*?)</think>"
#
#         def collect_think_content(match_obj):
#             # 매칭된 그룹(태그 안의 내용)을 리스트에 추가합니다.
#             think_contents.append(match_obj.group(1).strip())
#             return ""  # 원본 텍스트에서 <think>...</think> 부분을 빈 문자열로 대체하여 제거합니다.
#
#         # re.sub를 사용하여 모든 <think> 태그를 처리합니다.
#         cleaned_text = re.sub(pattern, collect_think_content, text, flags=re.DOTALL | re.IGNORECASE)
#
#         # 추출된 생각 내용이 있다면, 여러 개인 경우 분리자를 넣어 연결합니다.
#         concatenated_think_contents = "\n--- <THINK_SEPARATOR> ---\n".join(think_contents) if think_contents else None
#
#         return cleaned_text.strip(), concatenated_think_contents
#
#     async def generate_text(
#             self,
#             messages: List[Dict[str, str]],
#             request_id: str = "N/A",
#             max_tokens: int = 512,
#             temperature: float = 0.7,
#             stop_sequences: Optional[List[str]] = None,
#             model_name: Optional[str] = None,
#             **kwargs: Any
#     ) -> Dict[str, Any]:
#         # ... (초기화 및 메시지 유효성 검사 로직은 기존과 동일) ...
#         if not self.client:
#             logger.error("LLMService is not properly initialized (no httpx client).")
#             return {"error": "LLMService is not initialized.", "request_id": request_id}
#         if not messages:
#             return {"error": "'messages' must be provided.", "request_id": request_id}
#
#         estimated_tokens_for_budget = self._estimate_tokens(messages, max_tokens)
#         await self._acquire_tokens(estimated_tokens_for_budget, request_id)
#
#         raw_response_content_for_log = None  # 로깅용 원본 응답 텍스트
#         final_result: Dict[str, Any] = {"request_id": request_id}  # request_id를 결과에 포함
#
#         try:
#             request_payload: Dict[str, Any] = {
#                 "model": model_name or settings.DEFAULT_LLM_MODEL,
#                 "messages": messages,
#                 "max_tokens": max_tokens,
#                 "temperature": temperature,
#                 **kwargs
#             }
#             if stop_sequences:
#                 request_payload["stop"] = stop_sequences
#             headers = {"Content-Type": "application/json", "Accept": "application/json"}
#
#             logger.debug(
#                 f"ReqID: {request_id} - LLM API 요청 시작: URL={self.endpoint}, Model={request_payload.get('model')}")
#
#             response = await self.client.post(str(self.endpoint), json=request_payload, headers=headers)
#             raw_response_content_for_log = response.text  # 로깅 및 오류 시 사용
#             response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
#
#             raw_response_data = response.json()  # JSON 파싱
#             logger.debug(f"ReqID: {request_id} - LLM API 원본 응답 수신 (일부): {str(raw_response_data)[:200]}...")
#
#             # 1. LLM 응답 구조에서 주요 텍스트 추출
#             main_llm_output_text = self._extract_main_llm_output(raw_response_data)
#
#             if main_llm_output_text is not None:
#                 # 2. 추출된 주요 텍스트에서 <think> 태그 내용 분리 및 제거
#                 cleaned_text, think_content = self._extract_and_remove_think_tags(main_llm_output_text)
#
#                 logger.info(f"ReqID: {request_id} - LLM API 텍스트 생성 성공 (정리 후 길이: {len(cleaned_text)}).")
#                 final_result.update({
#                     "generated_text": cleaned_text.strip(),
#                     "raw_response": raw_response_data
#                 })
#                 if think_content:
#                     final_result["think_content"] = think_content  # 분리된 생각 내용 추가
#                     logger.info(f"ReqID: {request_id} - <think> 태그 내용 추출됨 (길이: {len(think_content)}).")
#             else:
#                 logger.error(f"ReqID: {request_id} - LLM 응답에서 생성된 텍스트 추출 실패.")
#                 final_result.update({"error": "Failed to extract generated text.", "raw_response": raw_response_data})
#
#             return final_result
#
#         except httpx.TimeoutException as e:
#             logger.error(f"ReqID: {request_id} - LLM API 요청 타임아웃 ({self.timeout}s): {e}", exc_info=True)
#             final_result.update({"error": f"Request timed out after {self.timeout} seconds: {e}", "raw_response": None})
#             return final_result
#         except httpx.HTTPStatusError as e:  # HTTP 오류 먼저 처리
#             logger.error(
#                 f"ReqID: {request_id} - LLM API HTTP 오류: Status={e.response.status_code}, Response={raw_response_content_for_log[:500] if raw_response_content_for_log else 'N/A'}")
#             final_result.update({"error": f"API call failed with status {e.response.status_code}",
#                                  "raw_response": raw_response_content_for_log})
#             return final_result
#         except httpx.RequestError as e:  # 네트워크 오류 등 (HTTP 오류 이후)
#             logger.error(f"ReqID: {request_id} - LLM API 요청 중 네트워크 오류: {e}", exc_info=True)
#             final_result.update({"error": f"Network error during request: {e}", "raw_response": None})
#             return final_result
#         except json.JSONDecodeError as e:  # JSON 파싱 오류
#             logger.error(
#                 f"ReqID: {request_id} - LLM API 응답 JSON 파싱 실패: {e}. Response text: {raw_response_content_for_log[:500] if raw_response_content_for_log else 'N/A'}",
#                 exc_info=True)
#             final_result.update(
#                 {"error": f"Failed to parse JSON response: {e}", "raw_response": raw_response_content_for_log})
#             return final_result
#         except Exception as e:  # 그 외 모든 예외
#             logger.error(f"ReqID: {request_id} - LLM API 호출 중 예상치 못한 오류: {e}", exc_info=True)
#             final_result.update(
#                 {"error": f"An unexpected error occurred: {e}", "raw_response": raw_response_content_for_log})
#             return final_result
#         finally:
#             await self._release_tokens(estimated_tokens_for_budget, request_id)
#
#     async def close(self):
#         # ... (기존 로직 유지) ...
#         if self.client:
#             try:
#                 await self.client.aclose()
#                 logger.info("LLMService의 httpx 클라이언트가 성공적으로 닫혔습니다.")
#             except Exception as e:
#                 logger.error(f"LLMService의 httpx 클라이언트 닫기 실패: {e}", exc_info=True)
#             finally:
#                 self.client = None
# ai/app/services/llm_service.py

import httpx
import json
import asyncio
import re
from typing import Optional, Dict, Any, List, Tuple, AsyncGenerator
# from pathlib import Path # [수정됨] 사용되지 않는 import 제거
from app.config.settings import Settings
from app.utils.logger import get_logger

# --- 기본 설정 ---
settings = Settings()
logger = get_logger("LLMService")

# [수정됨] 정확한 타입 힌트를 위해 PreTrainedTokenizer 추가
from transformers import PreTrainedTokenizer

# VLLM 최대 토큰 예산 설정 (settings.py에서 가져오거나 기본값 설정)
VLLM_MAX_TOKEN_BUDGET = getattr(settings, 'VLLM_MAX_TOKEN_BUDGET', 12000)


class LLMService:
    """
    LLM API 서버와 상호 작용하며, 스트리밍 및 지능형 재시도 로직을 포함하는 서비스.
    """
    DEFAULT_TIMEOUT = 60.0
    MAX_RETRIES = 3  # 최대 재시도 횟수

    def __init__(
            self,
            # [수정됨] AutoTokenizer -> PreTrainedTokenizer로 타입 힌트 변경
            tokenizer: PreTrainedTokenizer,
            endpoint: Optional[str] = None,
            timeout: Optional[float] = None,
            max_token_budget: int = VLLM_MAX_TOKEN_BUDGET
    ):
        self.tokenizer = tokenizer
        self.endpoint = endpoint or settings.LLM_API_ENDPOINT
        self.timeout = timeout if timeout is not None else float(settings.LLM_API_TIMEOUT)

        if not self.endpoint:
            logger.error("LLM API 엔드포인트가 설정되지 않았습니다.")
            raise ValueError("LLM API endpoint must be provided.")
        if not self.tokenizer:
            logger.error("토크나이저가 LLMService에 제공되지 않았습니다.")
            raise ValueError("Tokenizer must be provided to LLMService.")

        try:
            self.client = httpx.AsyncClient(timeout=self.timeout)
        except Exception as e:
            logger.error(f"httpx.AsyncClient 초기화 실패: {e}", exc_info=True)
            self.client = None
            raise ConnectionError(f"Failed to initialize httpx client: {e}") from e

        self.max_budget = max_token_budget
        self.available_tokens = max_token_budget
        self.condition = asyncio.Condition()

        logger.info(
            f"LLMService 초기화 완료. Endpoint: {self.endpoint}, Max Token Budget: {self.max_budget}, Timeout: {self.timeout}s")

    async def generate_text(
            self,
            messages: List[Dict[str, str]],
            request_id: str = "N/A",
            max_tokens: int = 1024,
            temperature: float = 0.7,
            stop_sequences: Optional[List[str]] = None,
            model_name: Optional[str] = None,
            **kwargs: Any
    ) -> Dict[str, Any]:
        """
        LLM 텍스트 생성을 오케스트레이션합니다.
        스트림이 'length'(길이 제한)에 의해 종료되고, '생각'만 하다가 끝난 경우에만
        지능적으로 재시도 로직을 수행합니다.
        """
        retry_count = 0
        current_messages = messages
        final_result = {}

        while retry_count <= self.MAX_RETRIES:
            if retry_count > 0:
                logger.info(f"ReqID: {request_id} - 재시도 {retry_count}/{self.MAX_RETRIES} 시작...")

            is_thinking_done = False
            is_truncated_by_length = False
            accumulated_text = ""
            final_think_content = None
            raw_response_for_log = ""
            error_message = None

            stream_generator = self.generate_text_stream(
                messages=current_messages,
                request_id=request_id,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                model_name=model_name,
                **kwargs
            )

            async for chunk in stream_generator:
                chunk_type = chunk.get("type")
                content = chunk.get("content")

                if chunk_type == "think_done":
                    is_thinking_done = True
                    final_think_content = content
                elif chunk_type == "content_delta":
                    accumulated_text += content
                elif chunk_type == "finish":
                    if content.get("reason") == "length":
                        is_truncated_by_length = True
                elif chunk_type == "raw_chunk":
                    raw_response_for_log += content
                elif chunk_type == "error":
                    error_message = content
                    break

            if error_message:
                final_result = {"error": error_message, "raw_response": raw_response_for_log, "request_id": request_id}
                break

            if is_truncated_by_length and not is_thinking_done:
                logger.warning(f"ReqID: {request_id} - 응답이 생각 도중 길이 제한으로 잘렸습니다. 재시도를 준비합니다.")
                retry_count += 1
                if retry_count > self.MAX_RETRIES:
                    logger.error(f"ReqID: {request_id} - 최대 재시도 횟수({self.MAX_RETRIES})를 초과하여 처리를 중단합니다.")
                    final_result = {"error": "Response truncated during thinking and max retries exceeded.",
                                    "raw_response": raw_response_for_log, "request_id": request_id}
                    break

                continuation_prompt = (
                    f"You were previously thinking about a problem but were cut off. "
                    f"Here is your incomplete thought process:\n\n{accumulated_text}\n\n"
                    f"Please continue your thought process from where you left off, and then provide the final, complete answer."
                )
                current_messages = messages + [{"role": "user", "content": continuation_prompt}]
                max_tokens = max_tokens * 2
                continue
            else:
                logger.info(f"ReqID: {request_id} - LLM API 텍스트 생성 성공 (시도 {retry_count + 1}회).")
                final_result = {
                    "generated_text": accumulated_text.strip(),
                    "think_content": final_think_content,
                    "raw_response": raw_response_for_log,
                    "request_id": request_id,
                    "retried": retry_count > 0
                }
                break

        return final_result

    async def generate_text_stream(
            self,
            messages: List[Dict[str, str]],
            request_id: str = "N/A",
            max_tokens: int = 1024,
            temperature: float = 0.7,
            stop_sequences: Optional[List[str]] = None,
            model_name: Optional[str] = None,
            **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not self.client:
            yield {"type": "error", "content": "LLMService is not initialized."}
            return

        estimated_tokens = self._estimate_tokens(messages, max_tokens)
        await self._acquire_tokens(estimated_tokens, request_id)

        finish_reason = None
        try:
            request_payload = {
                "model": model_name or settings.DEFAULT_LLM_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                **kwargs
            }
            if stop_sequences:
                request_payload["stop"] = stop_sequences

            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            logger.debug(f"ReqID: {request_id} - LLM API 스트림 요청 시작. (max_tokens: {max_tokens})")

            accumulated_response = ""
            is_thinking_done = False

            async with self.client.stream("POST", str(self.endpoint), json=request_payload,
                                          headers=headers) as response:
                response.raise_for_status()

                async for raw_chunk in response.aiter_text():
                    yield {"type": "raw_chunk", "content": raw_chunk}

                    if not raw_chunk.strip().startswith('data:'):
                        continue

                    content_part = raw_chunk[len('data:'):].strip()
                    if content_part == '[DONE]':
                        break

                    try:
                        json_content = json.loads(content_part)
                        choice = json_content.get("choices", [{}])[0]
                        delta = choice.get("delta", {}).get("content", "")

                        if choice.get("finish_reason"):
                            finish_reason = choice.get("finish_reason")
                            break

                        if not delta:
                            continue

                        if not is_thinking_done:
                            accumulated_response += delta
                            if "</think>" in accumulated_response:
                                is_thinking_done = True
                                cleaned_text, think_content = self._extract_and_remove_think_tags(accumulated_response)
                                yield {"type": "think_done", "content": think_content}
                                if cleaned_text:
                                    yield {"type": "content_delta", "content": cleaned_text}
                        else:
                            yield {"type": "content_delta", "content": delta}

                    except (json.JSONDecodeError, IndexError):
                        logger.warning(f"ReqID: {request_id} - 스트림 청크 파싱 실패: {content_part}")
                        continue

        except httpx.HTTPStatusError as e:
            error_body = await e.response.aread()
            error_text = error_body.decode('utf-8')
            logger.error(
                f"ReqID: {request_id} - LLM API HTTP 오류: Status={e.response.status_code}, Response={error_text[:500]}")
            yield {"type": "error", "content": f"API call failed with status {e.response.status_code}: {error_text}"}
        except Exception as e:
            logger.error(f"ReqID: {request_id} - LLM 스트림 중 예상치 못한 오류: {e}", exc_info=True)
            yield {"type": "error", "content": f"An unexpected stream error occurred: {e}"}
        finally:
            yield {"type": "finish", "content": {"reason": finish_reason}}
            await self._release_tokens(estimated_tokens, request_id)

    def _estimate_tokens(self, messages: List[Dict[str, str]], max_new_tokens_requested: int) -> int:
        input_tokens = 0
        try:
            for msg in messages:
                content = msg.get("content")
                if content and isinstance(content, str):
                    input_tokens += len(self.tokenizer.encode(content, add_special_tokens=False))
                elif content is not None:
                    logger.warning(f"Token estimation: Non-string content found: {type(content)}")
        except Exception as e:
            logger.warning(f"토큰 수 계산 중 오류 (폴백 사용): {e}")
            input_tokens = sum(len(str(m.get("content", "")).split()) for m in messages) * 2
        return input_tokens + max_new_tokens_requested

    async def _acquire_tokens(self, required_tokens: int, request_id: str = "N/A"):
        if required_tokens <= 0: return
        if required_tokens > self.max_budget:
            logger.error(f"ReqID: {request_id} - 요청 토큰({required_tokens})이 시스템 최대 예산({self.max_budget})을 초과. 요청 거부.")
            raise ValueError(f"Requested tokens ({required_tokens}) exceed system maximum budget ({self.max_budget}).")
        async with self.condition:
            while self.available_tokens < required_tokens:
                logger.info(f"ReqID: {request_id} - 토큰 부족. 대기. (요청: {required_tokens}, 사용 가능: {self.available_tokens})")
                await self.condition.wait()
            self.available_tokens -= required_tokens
            logger.info(f"ReqID: {request_id} - 토큰 확보. (요청: {required_tokens}, 남은 예산: {self.available_tokens})")

    async def _release_tokens(self, released_tokens: int, request_id: str = "N/A"):
        if released_tokens <= 0: return
        async with self.condition:
            self.available_tokens += released_tokens
            if self.available_tokens > self.max_budget:
                self.available_tokens = self.max_budget
            logger.info(f"ReqID: {request_id} - 토큰 반환. (반환: {released_tokens}, 현재 예산: {self.available_tokens})")
            self.condition.notify_all()

    # [수정됨] @staticmethod 추가 및 self 파라미터 제거
    @staticmethod
    def _extract_and_remove_think_tags(text: Optional[str]) -> Tuple[str, Optional[str]]:
        if not text:
            return "", None
        think_contents = []
        pattern = r"<think>(.*?)</think>"

        def collect_think_content(match_obj):
            think_contents.append(match_obj.group(1).strip())
            return ""

        cleaned_text = re.sub(pattern, collect_think_content, text, flags=re.DOTALL | re.IGNORECASE)
        concatenated_think_contents = "\n--- <THINK_SEPARATOR> ---\n".join(think_contents) if think_contents else None
        return cleaned_text.strip(), concatenated_think_contents

    async def close(self):
        if self.client:
            try:
                await self.client.aclose()
                logger.info("LLMService의 httpx 클라이언트가 성공적으로 닫혔습니다.")
            except Exception as e:
                logger.error(f"LLMService의 httpx 클라이언트 닫기 실패: {e}", exc_info=True)