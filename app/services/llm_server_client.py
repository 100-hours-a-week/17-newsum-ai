# app/services/llm_server_client.py

import httpx
import logging
from typing import Optional, Dict, Any
from app.config.settings import settings

logger = logging.getLogger(__name__)

async def call_llm_api(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stop_sequences: Optional[list[str]] = None,
    **kwargs: Any
) -> str:
    if not settings.LLM_API_ENDPOINT:
        logger.error("LLM API 엔드포인트가 설정되지 않았습니다.")
        raise ValueError("LLM_API_ENDPOINT is not configured.")

    request_payload: Dict[str, Any] = {
        "model": settings.LLM_API_MODEL,  # 예: "meta-llama/Llama-3-8b-instruct"
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }
    # request_payload: Dict[str, Any] = {
    #     "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, **kwargs
    # }
    if stop_sequences: request_payload["stop"] = stop_sequences
    headers = {"Content-Type": "application/json"}
    if settings.LLM_API_KEY: headers["X-API-Key"] = settings.LLM_API_KEY # 예시 헤더

    logger.debug(f"LLM API 요청: URL={settings.LLM_API_ENDPOINT}, Payload={request_payload}")

    async with httpx.AsyncClient(timeout=settings.LLM_API_TIMEOUT) as client:
        try:
            response = await client.post(settings.LLM_API_ENDPOINT, json=request_payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            print(f"LLM API 응답")
            # print(f"⚠️ LLM 전체 응답: {response_data}") # 디버깅용

            # --- 중요: 실제 LLM API 응답 형식에 맞게 수정 필요 ---
            generated_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # generated_text = response_data.get("choices", [{}])[0].get("text", "")
            if not generated_text and "generated_text" in response_data:
                 generated_text = response_data.get("generated_text", "")
            # --- ---
            
            logger.debug(f"LLM API 응답 수신: {generated_text[:100]}...")
            return generated_text.strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API 서버 오류: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"LLM API 요청 오류: {e}")
            raise
        except Exception as e:
            logger.exception(f"LLM API 호출 중 예기치 않은 오류 발생: {e}")
            raise