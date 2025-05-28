# ai/app/services/llm_service.py

import httpx
import json
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from app.config.settings import Settings
from app.utils.logger import get_logger

settings = Settings()  # settings.py를 통해 설정값 로드
logger = get_logger("LLMService")

from transformers import AutoTokenizer
# VLLM 최대 토큰 예산 설정 (settings.py에서 가져오거나 기본값 설정)
VLLM_MAX_TOKEN_BUDGET = getattr(settings, 'VLLM_MAX_TOKEN_BUDGET', 12000)

class LLMService:
    """
    LLM API 서버와 상호 작용하는 서비스 (토큰 예산 관리 기능 포함).
    """
    DEFAULT_TIMEOUT = 60.0

    def __init__(
            self,
            tokenizer: AutoTokenizer,  # 토크나이저 주입
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
        if not self.tokenizer:  # 토크나이저 존재 유무 확인
            logger.error("토크나이저가 LLMService에 제공되지 않았습니다.")
            raise ValueError("Tokenizer must be provided to LLMService.")

        try:
            self.client = httpx.AsyncClient(timeout=self.timeout)
        except Exception as e:  # httpx 클라이언트 초기화 실패 시 로깅 및 예외 발생
            logger.error(f"httpx.AsyncClient 초기화 실패: {e}", exc_info=True)
            self.client = None  # 명시적으로 None 할당
            raise ConnectionError(f"Failed to initialize httpx client: {e}") from e

        self.max_budget = max_token_budget
        self.available_tokens = max_token_budget
        self.lock = asyncio.Lock()  # Lock은 Condition 객체 내에 이미 포함되어 있어 별도 Lock 불필요
        self.condition = asyncio.Condition()  # Lock 인자 제거, 내부적으로 Lock 생성

        logger.info(
            f"LLMService 초기화 완료. Endpoint: {self.endpoint}, Max Token Budget: {self.max_budget}, Timeout: {self.timeout}s")

    def _estimate_tokens(self, messages: List[Dict[str, str]], max_new_tokens_requested: int) -> int:
        """
        요청의 예상 토큰 수를 계산합니다 (입력 + 요청된 최대 새 토큰).
        """
        input_tokens = 0
        try:
            for msg in messages:
                content = msg.get("content")
                if content and isinstance(content, str):  # content 타입 체크 추가
                    input_tokens += len(self.tokenizer.encode(content, add_special_tokens=False))
                elif content is not None:  # content가 있지만 문자열이 아닌 경우 경고
                    logger.warning(f"Token estimation: Non-string content found in message: {type(content)}")
        except Exception as e:
            logger.warning(f"토큰 수 계산 중 오류 (폴백 사용): {e}")
            input_tokens = sum(len(str(m.get("content", "")).split()) for m in messages) * 2

        return input_tokens + max_new_tokens_requested  # API에서 요청한 max_tokens 사용

    async def _acquire_tokens(self, required_tokens: int, request_id: str = "N/A"):
        """
        필요한 토큰만큼 예산을 확보합니다. 부족하면 대기합니다.
        """
        if required_tokens <= 0:  # 0 이하 토큰 요청 방지
            logger.warning(f"ReqID: {request_id} - 유효하지 않은 토큰 요청 값: {required_tokens}")
            return

        if required_tokens > self.max_budget:
            logger.error(f"ReqID: {request_id} - 요청 토큰({required_tokens})이 시스템 최대 예산({self.max_budget})을 초과합니다. 요청 거부.")
            raise ValueError(f"Requested tokens ({required_tokens}) exceed system maximum budget ({self.max_budget}).")

        async with self.condition:  # Condition 객체의 Lock을 사용
            while self.available_tokens < required_tokens:
                logger.info(
                    f"ReqID: {request_id} - 토큰 부족. 대기 시작. (요청: {required_tokens}, 사용 가능: {self.available_tokens}, 최대: {self.max_budget})")
                await self.condition.wait()

            self.available_tokens -= required_tokens
            logger.info(f"ReqID: {request_id} - 토큰 확보 성공. (요청: {required_tokens}, 남은 예산: {self.available_tokens})")

    async def _release_tokens(self, released_tokens: int, request_id: str = "N/A"):
        """
        사용한 토큰만큼 예산을 반환하고 대기 중인 작업을 깨웁니다.
        """
        if released_tokens <= 0:  # 0 이하 토큰 반환 방지
            logger.warning(f"ReqID: {request_id} - 유효하지 않은 토큰 반환 값: {released_tokens}")
            return

        async with self.condition:  # Condition 객체의 Lock을 사용
            self.available_tokens += released_tokens
            if self.available_tokens > self.max_budget:  # 만약을 위해 최대 예산 초과 방지
                logger.warning(
                    f"ReqID: {request_id} - 토큰 반환 후 사용 가능 토큰({self.available_tokens})이 최대 예산({self.max_budget})을 초과. 조정됨.")
                self.available_tokens = self.max_budget

            logger.info(f"ReqID: {request_id} - 토큰 반환. (반환: {released_tokens}, 현재 예산: {self.available_tokens})")
            self.condition.notify_all()

    def _extract_generated_text(self, response_data: Dict[str, Any]) -> Optional[str]:
        if not isinstance(response_data, dict):
            logger.warning(f"텍스트 추출 실패: 응답 데이터가 딕셔너리 타입이 아님 (Type: {type(response_data)})")
            return None
        try:
            if "choices" in response_data and isinstance(response_data["choices"], list) and response_data["choices"]:
                first_choice = response_data["choices"][0]
                if isinstance(first_choice, dict):
                    if "message" in first_choice and isinstance(first_choice["message"], dict) and "content" in \
                            first_choice["message"]:
                        return first_choice["message"]["content"]
                    if "text" in first_choice and isinstance(first_choice["text"], str):
                        return first_choice["text"]
            if "generated_text" in response_data and isinstance(response_data["generated_text"], str):
                return response_data["generated_text"]
            logger.warning(f"LLM 응답에서 생성된 텍스트를 알려진 패턴으로 추출할 수 없습니다. 응답 데이터 일부: {str(response_data)[:200]}...")
            return None
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"텍스트 추출 중 오류 발생: {e}. 응답 데이터 일부: {str(response_data)[:200]}...", exc_info=True)
            return None

    async def generate_text(
            self,
            messages: List[Dict[str, str]],
            request_id: str = "N/A",
            max_tokens: int = 512,  # 이 max_tokens는 LLM이 '생성할' 최대 토큰 수
            temperature: float = 0.7,
            stop_sequences: Optional[List[str]] = None,
            model_name: Optional[str] = None,
            **kwargs: Any
    ) -> Dict[str, Any]:
        if not self.client:  # 클라이언트 초기화 확인
            logger.error("LLMService is not properly initialized (no httpx client).")
            return {"error": "LLMService is not initialized."}
        if not messages:
            return {"error": "'messages' must be provided."}

        # 예상 필요 토큰 = 입력 메시지 토큰 + LLM이 생성할 최대 토큰 (max_tokens 파라미터)
        estimated_tokens_for_budget = self._estimate_tokens(messages, max_tokens)

        await self._acquire_tokens(estimated_tokens_for_budget, request_id)

        raw_response_content = None
        try:
            request_payload: Dict[str, Any] = {
                "model": model_name or settings.DEFAULT_LLM_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,  # LLM에게 전달하는 생성할 최대 토큰 수
                "temperature": temperature,
                **kwargs
            }
            if stop_sequences:
                request_payload["stop"] = stop_sequences

            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            # if settings.LLM_API_KEY:  # API 키가 설정에 있다면 헤더에 추가
            #     headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"

            logger.debug(
                f"ReqID: {request_id} - LLM API 요청 시작: URL={self.endpoint}, Model={request_payload.get('model')}")

            response = await self.client.post(str(self.endpoint), json=request_payload, headers=headers)
            raw_response_content = response.text
            response.raise_for_status()

            raw_response_data = response.json()
            logger.debug(f"ReqID: {request_id} - LLM API 원본 응답 수신 (일부): {str(raw_response_data)[:200]}...")
            generated_text = self._extract_generated_text(raw_response_data)

            if generated_text is not None:
                logger.info(f"ReqID: {request_id} - LLM API 텍스트 생성 성공 (길이: {len(generated_text)}).")
                return {"generated_text": generated_text.strip(), "raw_response": raw_response_data}
            else:
                logger.error(f"ReqID: {request_id} - LLM 응답에서 생성된 텍스트 추출 실패.")
                return {"error": "Failed to extract generated text.", "raw_response": raw_response_data}

        except httpx.TimeoutException as e:
            logger.error(f"ReqID: {request_id} - LLM API 요청 타임아웃 발생 ({self.timeout}s): {e}", exc_info=True)
            return {"error": f"Request timed out after {self.timeout} seconds: {e}", "raw_response": None}
        except httpx.RequestError as e:
            logger.error(f"ReqID: {request_id} - LLM API 요청 중 네트워크 오류 발생: {e}", exc_info=True)
            return {"error": f"Network error during request: {e}", "raw_response": None}
        except httpx.HTTPStatusError as e:
            logger.error(
                f"ReqID: {request_id} - LLM API HTTP 오류: Status={e.response.status_code}, Response={raw_response_content[:500] if raw_response_content else 'N/A'}",
                exc_info=False)  # 스택 트레이스는 불필요할 수 있음
            return {"error": f"API call failed with status {e.response.status_code}",
                    "raw_response": raw_response_content}
        except json.JSONDecodeError as e:
            logger.error(
                f"ReqID: {request_id} - LLM API 응답 JSON 파싱 실패: {e}. Response text: {raw_response_content[:500] if raw_response_content else 'N/A'}",
                exc_info=True)
            return {"error": f"Failed to parse JSON response: {e}", "raw_response": raw_response_content}
        except Exception as e:
            logger.error(f"ReqID: {request_id} - LLM API 호출 중 예상치 못한 오류 발생: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}", "raw_response": raw_response_content}
        finally:
            # 예산 확보 시 사용했던 토큰 양을 그대로 반환
            await self._release_tokens(estimated_tokens_for_budget, request_id)

    async def close(self):
        if self.client:
            try:
                await self.client.aclose()
                logger.info("LLMService의 httpx 클라이언트가 성공적으로 닫혔습니다.")
            except Exception as e:
                logger.error(f"LLMService의 httpx 클라이언트 닫기 실패: {e}", exc_info=True)
            finally:  # 에러 발생 여부와 관계없이 client를 None으로 설정
                self.client = None