# ai/app/services/llm_service.py

import httpx
import json
from typing import Optional, Dict, Any, List
from app.config.settings import Settings  # (경로 확인 필요)
from app.utils.logger import get_logger  # (경로 확인 필요)

settings = Settings()


class LLMService:
    """
    LLM API 서버와 상호 작용하는 서비스 (Llama 3 및 messages 입력 지원 확장).
    기존 prompt: str 방식과 새로운 messages: List 방식 모두 지원.
    """
    DEFAULT_TIMEOUT = 60.0

    def __init__(
            self,
            endpoint: Optional[str] = None,
            timeout: Optional[float] = None,
            logger_name: str = "LLMService"  # 기존 클래스 이름 유지
    ):
        self.endpoint = endpoint or settings.LLM_API_ENDPOINT
        self.logger = get_logger(logger_name)
        # settings.LLM_API_TIMEOUT은 int일 수 있으므로 float로 변환
        self.timeout = timeout if timeout is not None else float(settings.LLM_API_TIMEOUT)
        self.logger.debug(f"LLMService 타임아웃 설정: {self.timeout}초")

        if not self.endpoint:
            self.logger.error("LLM API 엔드포인트가 설정되지 않았습니다.")
            raise ValueError("LLM API endpoint must be provided.")

        try:
            self.client = httpx.AsyncClient(timeout=self.timeout)
            self.logger.info(f"LLMService 초기화 완료. Endpoint: {self.endpoint}")
        except Exception as e:
            self.logger.error(f"httpx.AsyncClient 초기화 실패: {e}", exc_info=True)
            self.client = None
            raise ConnectionError(f"Failed to initialize httpx client: {e}") from e

    def _extract_generated_text(self, response_data: Dict[str, Any]) -> Optional[str]:
        # (이전 답변의 _extract_generated_text 로직과 동일)
        if not isinstance(response_data, dict):
            self.logger.warning(f"텍스트 추출 실패: 응답 데이터가 딕셔너리 타입이 아님 (Type: {type(response_data)})")
            return None
        try:
            # Case 1: OpenAI-like structure (choices list) - vLLM OpenAI API 표준
            if "choices" in response_data and isinstance(response_data["choices"], list) and response_data["choices"]:
                first_choice = response_data["choices"][0]
                if isinstance(first_choice, dict):
                    if "message" in first_choice and isinstance(first_choice["message"], dict) and "content" in \
                            first_choice["message"]:
                        return first_choice["message"]["content"]  # Chat completion format
                    if "text" in first_choice and isinstance(first_choice["text"], str):  # Completion format
                        return first_choice["text"]

            # Case 2: Simple 'generated_text' key (일부 다른 API용)
            if "generated_text" in response_data and isinstance(response_data["generated_text"], str):
                return response_data["generated_text"]

            # Case 3: Hugging Face Inference API (TGI 등) - 보통 리스트로 반환되나, 여기서는 딕셔너리 내 필드 가정
            # 만약 response_data가 리스트일 수 있다면, 이 메서드 초입에서 타입 체크 분기 필요
            # if isinstance(response_data, list) and response_data:
            #     first_item = response_data[0]
            #     if isinstance(first_item, dict) and "generated_text" in first_item:
            #         return first_item["generated_text"]

            self.logger.warning(f"LLM 응답에서 생성된 텍스트를 알려진 패턴으로 추출할 수 없습니다. 응답 데이터 일부: {str(response_data)[:200]}...")
            return None
        except (KeyError, IndexError, TypeError) as e:
            self.logger.error(f"텍스트 추출 중 오류 발생: {e}. 응답 데이터 일부: {str(response_data)[:200]}...", exc_info=True)
            return None

    async def generate_text(
            self,
            prompt: Optional[str] = None,  # 기존 인터페이스 호환용
            messages: Optional[List[Dict[str, str]]] = None,  # Llama 3 및 채팅 형식용
            system_prompt_content: Optional[str] = None,  # prompt 와 함께 사용될 시스템 메시지
            max_tokens: int = 512,
            temperature: float = 0.7,
            stop_sequences: Optional[List[str]] = None,
            model_name: Optional[str] = None,  # 사용할 모델 지정 (vLLM은 페이로드에 model 필드 필요)
            apply_llama3_template: bool = False,
            # 명시적으로 Llama3 템플릿 적용 여부 (여기서는 False로 두고, content에 직접 포함하거나 vLLM이 처리하도록 가정)
            **kwargs: Any
    ) -> Dict[str, Any]:
        if not self.client:
            return {"error": "LLMService is not initialized."}

        final_messages: List[Dict[str, str]] = []

        if messages:  # messages 인자가 제공되면 최우선으로 사용
            final_messages = messages
            if system_prompt_content and not any(m['role'] == 'system' for m in final_messages):
                # messages에 시스템 프롬프트가 없고 system_prompt_content가 제공되면 추가
                final_messages.insert(0, {"role": "system", "content": system_prompt_content})
        elif prompt:  # messages가 없고 prompt가 제공되면 이를 user 메시지로 사용
            if system_prompt_content:
                final_messages.append({"role": "system", "content": system_prompt_content})
            final_messages.append({"role": "user", "content": prompt})
        else:
            return {"error": "Either 'prompt' or 'messages' must be provided."}

        if not final_messages:  # 최종적으로 보낼 메시지가 없으면 오류
            return {"error": "No valid prompt content to send after processing inputs."}

        # Llama 3 특수 토큰 처리 (필요한 경우)
        # 만약 vLLM의 OpenAI API가 Llama 3 채팅 템플릿을 자동으로 적용하지 않거나,
        # 또는 특정 방식으로 content에 특수 토큰을 포함해야 한다면 여기서 처리.
        # 예시: (이 부분은 실제 vLLM 설정 및 Llama 3 서빙 방식에 따라 매우 달라짐)
        # if apply_llama3_template:
        #     templated_content_parts = ["<|begin_of_text|>"]
        #     for msg in final_messages:
        #         templated_content_parts.append(f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>")
        #     templated_content_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        #     # vLLM이 단일 'prompt' 필드를 기대한다면, 여기서 request_payload를 다르게 구성해야 함
        #     # request_payload = {"prompt": "".join(templated_content_parts), "model": ..., "max_tokens": ...}
        #     # 하지만 여기서는 vLLM OpenAI API가 'messages'를 받는다고 가정.
        #     # 이 경우, 각 message의 content에 토큰을 넣거나, vLLM의 템플릿 기능을 사용해야 함.
        #     # 가장 간단한 것은 vLLM의 기본 Llama3 템플릿을 믿고, final_messages를 그대로 전달하는 것.
        #     pass # 여기서는 final_messages를 그대로 사용한다고 가정

        request_payload: Dict[str, Any] = {
            "model": model_name or settings.DEFAULT_LLM_MODEL,
            "messages": final_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        if stop_sequences:
            request_payload["stop"] = stop_sequences

        # API 키가 필요한 경우 헤더에 추가 (settings.LLM_API_KEY 등 사용)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        # if settings.LLM_API_KEY:
        #     headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"

        raw_response_content = None
        try:
            self.logger.debug(
                f"LLM API 요청: URL={self.endpoint}, Model={request_payload.get('model')}, Messages Count={len(request_payload.get('messages', []))}")
            # self.logger.debug(f"Request Payload (sample): {str(request_payload)[:500]}") # 필요시 페이로드 로깅

            response = await self.client.post(str(self.endpoint), json=request_payload, headers=headers)
            raw_response_content = response.text  # 오류 시 로깅/반환 위해 미리 저장
            response.raise_for_status()  # HTTP 오류 시 예외 발생

            raw_response_data = response.json()
            self.logger.debug(f"LLM API 원본 응답 수신 (일부): {str(raw_response_data)[:200]}...")
            generated_text = self._extract_generated_text(raw_response_data)

            if generated_text is not None:
                self.logger.info(f"LLM API 텍스트 생성 성공 (길이: {len(generated_text)}).")
                return {"generated_text": generated_text.strip(), "raw_response": raw_response_data}
            else:
                self.logger.error("LLM 응답에서 생성된 텍스트 추출 실패.")
                return {"error": "Failed to extract generated text from LLM response.",
                        "raw_response": raw_response_data}

        except httpx.TimeoutException as e:
            self.logger.error(f"LLM API 요청 타임아웃 발생 ({self.timeout}s): {e}", exc_info=True)
            return {"error": f"Request timed out after {self.timeout} seconds: {e}", "raw_response": None}
        except httpx.RequestError as e:
            self.logger.error(f"LLM API 요청 중 네트워크 오류 발생: {e}", exc_info=True)
            return {"error": f"Network error during request: {e}", "raw_response": None}
        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"LLM API HTTP 오류: Status={e.response.status_code}, Response={raw_response_content[:500] if raw_response_content else 'N/A'}",
                exc_info=False)
            return {"error": f"API call failed with status {e.response.status_code}",
                    "raw_response": raw_response_content}
        except json.JSONDecodeError as e:
            self.logger.error(
                f"LLM API 응답 JSON 파싱 실패: {e}. Response text: {raw_response_content[:500] if raw_response_content else 'N/A'}",
                exc_info=True)
            return {"error": f"Failed to parse JSON response: {e}", "raw_response": raw_response_content}
        except Exception as e:
            self.logger.error(f"LLM API 호출 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}", "raw_response": raw_response_content}

    async def close(self):
        if self.client:
            try:
                await self.client.aclose()
                self.logger.info("LLMService의 httpx 클라이언트가 성공적으로 닫혔습니다.")
            except Exception as e:
                self.logger.error(f"LLMService의 httpx 클라이언트 닫기 실패: {e}", exc_info=True)
            finally:
                self.client = None