# app/services/papago_translation_service.py
import aiohttp
import asyncio
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger("PapagoTranslationService")

class PapagoTranslationService:
    """Papago NMT API와 상호작용하는 서비스 (내부 재시도 포함)"""
    def __init__(self):
        # 설정에서 Naver Client ID/Secret 로드 (Papago용)
        self.client_id = settings.NAVER_CLIENT_ID
        self.client_secret = settings.NAVER_CLIENT_SECRET
        self.api_url = "https://openapi.naver.com/v1/papago/n2mt"
        self.is_enabled = bool(self.client_id and self.client_secret)
        if not self.is_enabled:
            logger.warning("Papago Client ID or Secret is missing in settings. Translation service disabled.")

    # 설정의 재시도 횟수 사용 (LLM과 공유하거나 별도 PAPAGO_API_RETRIES 설정)
    @retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True
    )
    async def translate(
        self, text: str, source_lang: str, target_lang: str, session: aiohttp.ClientSession, trace_id: Optional[str] = None
        ) -> Optional[str]:
        """Papago API를 호출하여 텍스트 번역"""
        if not self.is_enabled or not text:
            return None # 비활성화 또는 빈 텍스트 시 None 반환

        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Papago Service: Translating '{text[:30]}...' from {source_lang} to {target_lang}")

        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
        }
        data = {'source': source_lang, 'target': target_lang, 'text': text}

        try:
            # 외부에서 전달받은 세션 사용
            async with session.post(self.api_url, headers=headers, data=data, timeout=aiohttp.ClientTimeout(total=10)) as response:
                # Papago는 오류 시에도 200 OK를 반환하고 응답 본문에 에러 코드를 포함할 수 있음
                if response.status == 200:
                    result = await response.json()
                    translated = result.get("message", {}).get("result", {}).get("translatedText")
                    if translated:
                        logger.debug(f"{log_prefix} Papago translation successful.")
                        return translated
                    else:
                        # API 호출은 성공했으나 번역 결과가 없는 경우 (오류 코드 확인 가능)
                        error_code = result.get("errorCode")
                        error_message = result.get("errorMessage")
                        logger.warning(f"{log_prefix} Papago API returned success status but no translation. Code: {error_code}, Msg: {error_message}. Response: {result}")
                        return None # 번역 실패로 간주
                elif response.status == 429:
                     logger.warning(f"{log_prefix} Papago API rate limit hit. Returning None.")
                     # 재시도 대신 None 반환 (또는 다른 처리)
                     return None
                else:
                    # 4xx, 5xx 등 HTTP 오류
                    error_text = await response.text()
                    logger.error(f"{log_prefix} Papago API HTTP error (Status {response.status}): {error_text}")
                    response.raise_for_status() # tenacity 재시도 유발
                    return None # raise_for_status 이후 도달하지 않음
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
             logger.error(f"{log_prefix} Papago API connection/timeout error: {e}")
             raise # tenacity 재시도 위해 예외 다시 발생
        except Exception as e:
             # JSON 파싱 오류 등 예상치 못한 오류
             logger.exception(f"{log_prefix} Papago translation unexpected error: {e}")
             return None # 최종 실패 시 None 반환