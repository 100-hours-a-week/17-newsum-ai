# ai/app/services/llm_service.py

import httpx
import json
from typing import Optional, Dict, Any, List
from app.config.settings import Settings # 변경: 중앙 설정 객체 임포트
from app.utils.logger import get_logger

settings = Settings()

class LLMService:
    """
    LLM API 서버와 상호 작용 하는 서비스 (최적화 버전)
    환경 변수/인자 설정, 표준 인증, 유연한 텍스트 추출, 오류 반환 방식을 지원 합니다.
    """
    DEFAULT_TIMEOUT = 60.0 # 기본 타임아웃 값

    def __init__(
        self,
        endpoint: Optional[str] = None,
        #api_key: Optional[str] = None, # API 키 (Bearer 토큰 등에 사용)
        timeout: Optional[float] = None,
        logger_name: str = "LLMService" # 클래스 이름과 통일
    ):
        """
        LLM 서비스 초기화.

        Args:
            endpoint (Optional[str]): LLM API 엔드포인트 URL. 기본값: 환경변수 LLM_API_ENDPOINT.
            api_key (Optional[str]): LLM API 키 (Bearer 토큰으로 사용됨). 기본값: 환경변수 LLM_API_KEY.
            timeout (Optional[float]): 요청 타임아웃(초). 기본값: 환경변수 LLM_API_TIMEOUT 또는 60.0.
            logger_name (str): 로거 인스턴스 이름.
        """
        # 환경 변수 또는 인자 값 사용
        self.endpoint = endpoint or settings.LLM_API_ENDPOINT
        #self.api_key = api_key or settings.LLM_API_KEY

        # 로거 설정
        self.logger = get_logger(logger_name)

        # settings.LLM_API_TIMEOUT은 int 이므로 float로 변환
        self.timeout = timeout if timeout is not None else float(settings.LLM_API_TIMEOUT)
        self.logger.debug(f"LLMService 타임아웃 설정: {self.timeout}초")  # 디버그 로그 추가

        # 필수 설정 값 확인
        if not self.endpoint:
            self.logger.error("LLM API 엔드포인트가 설정되지 않았습니다 (settings 또는 인자 필요).")
            raise ValueError("LLM API endpoint must be provided via argument or settings.LLM_API_ENDPOINT")

        # 비동기 HTTP 클라이언트 생성
        try:
            # timeout 값은 float 형태여야 함
            self.client = httpx.AsyncClient(timeout=self.timeout)
            self.logger.info(f"LLMService 초기화 완료. Endpoint: {self.endpoint}")
        except Exception as e:
            self.logger.error(f"httpx.AsyncClient 초기화 실패: {e}", exc_info=True)
            self.client = None # client 초기화 실패 시 None으로 설정
            raise ConnectionError(f"Failed to initialize httpx client: {e}") from e

    # --- 수정 1: _extract_generated_text 메서드를 클래스 내부로 들여쓰기 ---
    def _extract_generated_text(self, response_data: Dict[str, Any]) -> Optional[str]:
        """
        다양한 LLM API 응답 형식에서 생성된 텍스트를 추출하는 내부 헬퍼 메서드.
        (실제 사용하는 LLM API 응답 형식에 맞게 이 부분을 가장 먼저 커스터마이징해야 합니다.)
        """
        if not isinstance(response_data, dict):
            self.logger.warning(f"텍스트 추출 실패: 응답 데이터가 딕셔너리 타입이 아님 (Type: {type(response_data)})")
            return None

        try:
            # Case 1: OpenAI-like structure (choices list)
            if "choices" in response_data and isinstance(response_data["choices"], list) and response_data["choices"]:
                first_choice = response_data["choices"][0]
                if isinstance(first_choice, dict):
                    if "text" in first_choice and isinstance(first_choice["text"], str):
                        return first_choice["text"]
                    if "message" in first_choice and isinstance(first_choice["message"], dict) and "content" in first_choice["message"]:
                        return first_choice["message"]["content"] # Chat completion format

            # Case 2: Simple 'generated_text' key
            if "generated_text" in response_data and isinstance(response_data["generated_text"], str):
                return response_data["generated_text"]

            # Case 3: Simple 'text' key at top level
            if "text" in response_data and isinstance(response_data["text"], str):
                 return response_data["text"]

            # Case 4: Hugging Face Inference API (common list structure)
            # 이 케이스는 response_data가 리스트일 때를 가정하므로, 첫 번째 방어 로직(isinstance(response_data, dict))과 충돌할 수 있습니다.
            # 만약 리스트 형태의 응답도 처리해야 한다면, 초반 타입 체크 로직 수정 필요.
            # 여기서는 일단 딕셔너리 기반 응답만 가정하고 진행합니다.
            # if isinstance(response_data, list) and response_data:
            #     first_item = response_data[0]
            #     if isinstance(first_item, dict) and "generated_text" in first_item:
            #          return first_item["generated_text"]

            # Add more cases based on the specific LLM APIs you use
            # ...

            self.logger.warning(f"LLM 응답에서 생성된 텍스트를 알려진 패턴으로 추출할 수 없습니다. 응답 데이터 일부: {str(response_data)[:200]}...")
            return None # 추출 실패 시 None 반환

        except (KeyError, IndexError, TypeError) as e:
            self.logger.error(f"텍스트 추출 중 오류 발생: {e}. 응답 데이터 일부: {str(response_data)[:200]}...", exc_info=True)
            return None

    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any # 추가 API 파라미터
    ) -> Dict[str, Any]:
        """
        주어진 프롬프트와 파라미터로 LLM API를 호출하여 텍스트를 생성합니다.

        Args:
            prompt (str): LLM에 전달할 프롬프트.
            max_tokens (int): 생성할 최대 토큰 수.
            temperature (float): 생성 시 샘플링 온도.
            stop_sequences (Optional[List[str]]): 생성을 중단할 시퀀스 목록.
            **kwargs: LLM API에 전달할 추가 파라미터 (예: top_p, presence_penalty 등).

        Returns:
            Dict[str, Any]: 성공 시 {"generated_text": str, "raw_response": dict}
                           실패 시 {"error": str, "raw_response": Optional[Any]}
        """
        if not self.client:
            self.logger.error("LLMService가 초기화되지 않아 generate_text 작업을 수행할 수 없습니다.")
            # 클래스 초기화 실패 시 client가 None일 수 있으므로 raw_response 키 없이 반환
            return {"error": "LLMService is not initialized."}

        if not prompt:
            self.logger.error("빈 프롬프트가 LLM 생성 서비스에 제공되었습니다.")
            return {"error": "Prompt cannot be empty"}

        # 요청 페이로드 구성
        request_payload: Dict[str, Any] = {
            "model": settings.DEFAULT_LLM_MODEL,  # 모델 이름 추가 필요 (vLLM OpenAI API는 model 파라미터 필수)
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        if stop_sequences:
            request_payload["stop"] = stop_sequences

        # 요청 헤더 구성 (Bearer 토큰 기본 사용)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        # if self.api_key:
        #     headers["Authorization"] = f"Bearer {self.api_key}" # 표준 Bearer 토큰 사용

        raw_response_content = None # 오류 발생 시 원본 응답 저장용
        try:
            self.logger.debug(f"LLM API 요청: URL={self.endpoint}, Headers={list(headers.keys())}, Payload={request_payload}")

            # --- 수정: httpx.post 호출 시 endpoint를 문자열로 변환 ---
            response = await self.client.post(
                str(self.endpoint), # <<< AnyHttpUrl 객체를 str()로 변환
                json=request_payload,
                headers=headers
            )
            # ----------------------------------------------------
            raw_response_content = response.text # 오류 발생 시 로깅/반환 위해 미리 저장
            response.raise_for_status() # HTTP 오류 시 예외 발생

            # 응답 처리
            raw_response_data = response.json() # JSON 파싱 시도
            self.logger.debug(f"LLM API 원본 응답 수신 (일부): {str(raw_response_data)[:200]}...")

            # 텍스트 추출 시도
            generated_text = self._extract_generated_text(raw_response_data)

            if generated_text is not None:
                self.logger.info(f"LLM API 텍스트 생성 성공 (길이: {len(generated_text)}).")
                return {
                    "generated_text": generated_text.strip(),
                    "raw_response": raw_response_data
                }
            else:
                # 텍스트 추출 실패
                self.logger.error("LLM 응답에서 생성된 텍스트 추출 실패.") # 명시적 에러 로그 추가
                return {
                    "error": "Failed to extract generated text from LLM response.",
                    "raw_response": raw_response_data
                }

        except httpx.TimeoutException as e:
            self.logger.error(f"LLM API 요청 타임아웃 발생: {e}", exc_info=True)
            # 타임아웃 시 raw_response_content는 없을 수 있음
            return {"error": f"Request timed out after {self.timeout} seconds: {e}", "raw_response": None}
        except httpx.RequestError as e:
            # 네트워크 연결 관련 오류 등
            self.logger.error(f"LLM API 요청 중 네트워크 오류 발생: {e}", exc_info=True)
            return {"error": f"Network error during request: {e}", "raw_response": None}
        except httpx.HTTPStatusError as e:
            # 4xx, 5xx 등 HTTP 오류 상태 코드 수신 시
            self.logger.error(
                f"LLM API HTTP 오류: Status={e.response.status_code}, Response={raw_response_content[:200] if raw_response_content else 'N/A'}...",
                exc_info=False # 스택 트레이스 없이 상태와 응답만 로깅
            )
            return {
                "error": f"API call failed with status {e.response.status_code}",
                "raw_response": raw_response_content # 오류 시에도 원본 텍스트 응답 반환 시도
            }
        except json.JSONDecodeError as e:
             # --- 수정 2: 아래 두 라인의 앞쪽 추가 공백 제거 ---
             self.logger.error(f"LLM API 응답 JSON 파싱 실패: {e}. Response text: {raw_response_content[:200] if raw_response_content else 'N/A'}...", exc_info=True)
             return {
                 "error": f"Failed to parse JSON response: {e}",
                 "raw_response": raw_response_content # 파싱 실패 시 원본 텍스트 응답 반환
             }
        except Exception as e:
            # 예상치 못한 모든 종류의 오류 처리
            # <<< 수정: 오류 메시지에 httpx 오류 타입 명시적으로 포함 >>>
            if isinstance(e, TypeError) and "Invalid type for url" in str(e):
                # httpx URL 타입 오류 특화 로깅
                self.logger.error(f"LLM API 호출 URL 타입 오류 발생: {e}. Endpoint type: {type(self.endpoint)}", exc_info=True)
                return {"error": f"Internal configuration error: Invalid URL type provided to HTTP client.",
                        "raw_response": None}
            else:
                # 기타 예상치 못한 오류
                self.logger.error(f"LLM API 호출 중 예상치 못한 오류 발생: {e}", exc_info=True)
                return {"error": f"An unexpected error occurred: {e}", "raw_response": raw_response_content}

    async def close(self):
        """
        HTTP 클라이언트 세션을 닫습니다.
        """
        if self.client:
            try:
                await self.client.aclose()
                self.logger.info("LLMService의 httpx 클라이언트가 성공적으로 닫혔습니다.")
                self.client = None # 닫힌 후 None으로 설정하여 상태 명확화
            except Exception as e:
                self.logger.error(f"LLMService의 httpx 클라이언트 닫기 실패: {e}", exc_info=True)
                # 이미 닫혔거나 오류 발생 시에도 client를 None으로 설정할 수 있음
                self.client = None
        else:
            self.logger.info("LLMService의 httpx 클라이언트가 이미 닫혀 있거나 초기화되지 않았습니다.")


# --- 선택적 싱글턴 인스턴스 ---
# 필요에 따라 주석 해제하여 사용
# llm_service = LLMService()

# --- 사용 예시 ---
# import asyncio # asyncio 임포트 추가

# async def main():
#     # 로깅 레벨 설정 (디버그 메시지 확인용)
#     # get_logger 대신 표준 logging 사용 예시
#     logging.basicConfig(level=logging.DEBUG)
#     logger = logging.getLogger("LLMService") # LLMService 내부 로거와 이름 통일

#     try:
#         # 환경 변수 기반 초기화 (필요 시 .env 파일에 LLM_API_ENDPOINT, LLM_API_KEY 설정)
#         # 예시 실행을 위해 임시 엔드포인트 설정 (실제 사용 시 환경 변수 또는 인자로 전달)
#         client = LLMService(endpoint="http://127.0.0.1:8080/generate") # <--- 실제 엔드포인트로 변경 필요
#     except (ValueError, ConnectionError) as e:
#         logger.critical(f"클라이언트 초기화 실패: {e}")
#         return

#     # 텍스트 생성 요청
#     result = await client.generate_text(
#         prompt="LangGraph와 LLM 서비스를 연동하는 방법에 대해 설명해주세요.",
#         max_tokens=150,
#         temperature=0.5,
#         stop_sequences=["\n\n"], # 중단 시퀀스 예시
#         # 추가 파라미터 예시 (API가 지원하는 경우)
#         # top_p=0.9,
#     )

#     if "error" in result:
#         print(f"Error: {result['error']}")
#         # raw_response가 None일 수도 있으므로 get 사용
#         print(f"Raw Response: {result.get('raw_response', 'N/A')}")
#     else:
#         print(f"Success! Generated Text:\n{result['generated_text']}")
#         # print(f"Raw Response: {result['raw_response']}") # 필요시 원본 응답 확인

#     # 종료 시 클라이언트 닫기
#     await client.close()

# if __name__ == "__main__":
#      # 사용 예시 실행 시 asyncio 필요
#      # import asyncio
#      # 로깅 설정 확인 (main 함수 시작 시 basicConfig로 설정됨)
#      asyncio.run(main())