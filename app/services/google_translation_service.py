# app/services/google_translation_service.py
import asyncio
import httpx  # httpx로 변경 (비동기 HTTP 요청)
from typing import Optional, Dict, Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.settings import Settings
from app.utils.logger import get_logger, summarize_for_logging

logger = get_logger("GoogleRestTranslationService")  # 클래스명 변경에 따른 로거명 변경
settings = Settings()


class GoogleRestTranslationService:
    """Google Cloud Translation API (REST, API Key)와 상호작용하는 서비스"""

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.api_key = settings.GOOGLE_API_KEY
        # Google Cloud Translation API v2 REST endpoint
        self.api_url = "https://translation.googleapis.com/language/translate/v2"
        self.is_enabled = bool(self.api_key)

        self._client = http_client
        self._created_client = False

        if not self.is_enabled:
            logger.warning("GOOGLE_API_KEY is missing in settings. Google Translation service (REST API Key) disabled.")
        else:
            if self._client is None:
                # 타임아웃 설정은 settings에서 가져오거나 기본값 사용
                timeout_seconds = settings.TOOL_HTTP_TIMEOUT if hasattr(settings, 'TOOL_HTTP_TIMEOUT') else 30
                self._client = httpx.AsyncClient(timeout=float(timeout_seconds))
                self._created_client = True
                logger.info(f"GoogleRestTranslationService initialized. Using API Key. Timeout: {timeout_seconds}s")

    @retry(
        stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS or 3),
        wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN or 1,
                              max=settings.TOOL_RETRY_WAIT_MAX or 5),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError)),
        reraise=True
    )
    async def translate(
            self,
            text: str,
            target_lang: str,
            source_lang: Optional[str] = None,
            trace_id: Optional[str] = None
    ) -> Optional[str]:
        if not self.is_enabled or not self._client or not text:
            if not self.is_enabled or not self._client:
                logger.warning("Google Translation service (REST API Key) is disabled or client not initialized.")
            if not text:
                logger.debug("Empty text received for translation, returning None.")
            return None

        log_prefix = f"[{trace_id}]" if trace_id else ""
        source_info = f"from '{source_lang}' " if source_lang else "(auto-detect) "
        logger.debug(
            f"{log_prefix} Google REST Service: Translating '{summarize_for_logging(text, 30)}...' {source_info}to '{target_lang}'")

        params: Dict[str, Any] = {
            'q': text,
            'target': target_lang,
            'key': self.api_key,
            'format': 'html'  # HTML 태그 보존을 위해 (보고서가 HTML이므로)
        }
        if source_lang:
            params['source'] = source_lang

        try:
            response = await self._client.post(self.api_url, params=params)  # POST지만 파라미터는 URL에 포함될 수 있음 (API 명세 확인)
            # 또는 data=params로 form-urlencoded 데이터로 전송
            # Google Translation API v2는 GET 또는 POST를 지원하며, q 파라미터로 텍스트를 전달합니다.
            # POST로 보낼 경우, 페이로드는 {'q': text, 'target': target_lang, ...} 형태의 JSON이 될 수 있습니다.
            # 여기서는 URL 파라미터로 key를 전달하고, 본문은 form 데이터로 구성해 보겠습니다.
            # 또는 모든 것을 URL 파라미터로 GET 요청할 수도 있습니다.
            # Google API 문서는 POST + form data 또는 GET + query params를 주로 안내합니다.
            # POST + JSON body도 가능할 수 있으나, API Key 인증은 보통 URL에 key를 포함합니다.
            # 좀 더 명확한 방법은 URL에 key를 포함하고, 다른 파라미터들은 POST body로 보내는 것입니다.

            # API 문서를 다시 참조하여 정확한 요청 방식을 확인해야 합니다.
            # 일반적으로 API Key는 URL 파라미터로 전달하고, 데이터는 POST body로 전달하는 경우가 많습니다.
            # 여기서는 모든 것을 URL 파라미터로 구성하여 GET 요청으로 변경해보겠습니다. (더 간단)
            # response = await self._client.get(self.api_url, params=params)

            # POST with form data (일반적)
            form_data = {
                'q': text,
                'target': target_lang,
                'format': 'html'  # 또는 'text'
            }
            if source_lang:
                form_data['source'] = source_lang

            # API 키는 URL 파라미터로 추가
            request_url = f"{self.api_url}?key={self.api_key}"

            response = await self._client.post(request_url, data=form_data)
            response.raise_for_status()  # HTTP 오류 시 예외 발생 (4xx, 5xx)

            result = response.json()

            # 응답 구조 확인 (Google Cloud Translation API v2 기준)
            # { "data": { "translations": [ { "translatedText": "...", "detectedSourceLanguage": "ko" } ] } }
            if result and 'data' in result and 'translations' in result['data'] and result['data']['translations']:
                translation_info = result['data']['translations'][0]
                translated_text = translation_info.get('translatedText')
                detected_source = translation_info.get('detectedSourceLanguage', 'N/A')

                if translated_text:
                    logger.debug(f"{log_prefix} Google REST translation successful. Detected source: {detected_source}")
                    return translated_text

            logger.warning(
                f"{log_prefix} Google REST API returned no valid translation. Response: {summarize_for_logging(result)}")
            return None

        except httpx.HTTPStatusError as e:
            response_text = e.response.text
            logger.error(
                f"{log_prefix} Google REST API HTTP error (Status {e.response.status_code}): {summarize_for_logging(response_text)}",
                exc_info=False)
            # 오류 응답에 상세 정보가 있을 수 있음
            # 예: { "error": { "code": 400, "message": "API key not valid. Please pass a valid API key.", "errors": [...] } }
            try:
                error_details = e.response.json()
                logger.error(f"{log_prefix} Google REST API error details: {error_details}")
            except ValueError:  # JSON 디코딩 실패
                pass
            raise  # tenacity가 재시도하도록 예외를 다시 발생
        except httpx.RequestError as e:  # TimeoutException 포함
            logger.error(f"{log_prefix} Google REST API request error: {type(e).__name__} - {e}", exc_info=True)
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} Google REST translation unexpected error: {e}")
            raise  # 또는 None 반환 후 호출 측에서 처리

    async def close(self):
        """HTTP 클라이언트 세션을 닫습니다 (내부적으로 생성한 경우)."""
        if self._client and self._created_client:
            try:
                await self._client.aclose()
                logger.info("GoogleRestTranslationService: httpx client closed successfully.")
                self.client = None  # type: ignore
            except Exception as e:
                logger.error(f"GoogleRestTranslationService: Failed to close httpx client: {e}", exc_info=True)
        elif self._client:  # 외부에서 주입받았지만 닫을 필요는 없음
            logger.info("GoogleRestTranslationService: Using externally managed httpx client. Not closing here.")