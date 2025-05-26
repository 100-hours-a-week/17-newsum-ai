# ai/app/services/backend_client.py

import aiohttp
import json
import uuid
from app.utils.logger import get_logger # 프로젝트의 로거 유틸리티 경로에 맞게 수정하세요.
from app.config.settings import settings

logger = get_logger(__name__)
# BACKEND_RECEIVE_API_URL
callback_url = settings.BACKEND_RECEIVE_API_URL

class BackendApiClient:
    """
    Streamlit 백엔드 (Reception API)와 통신하는 클라이언트입니다.
    """

    def __init__(self, session: aiohttp.ClientSession):
        """
        aiohttp.ClientSession을 주입받아 초기화합니다.
        세션 관리를 통해 효율적인 HTTP 통신을 수행합니다.
        """
        self._session = session

    async def send_ai_response(self, request_id: uuid.UUID, content: str) -> bool:
        """
        최종 AI 응답을 지정된 callback_url (reception_api)로 전송합니다.

        Args:
            callback_url: 응답을 보낼 reception_api의 주소.
            request_id: 요청을 식별하는 고유 ID.
            content: AI가 생성한 최종 응답 내용.

        Returns:
            전송 성공 시 True, 실패 시 False.
        """
        if not callback_url:
            logger.error(f"Cannot send response: No callback_url provided for ReqID: {request_id}")
            return False

        # reception_api.py의 /receive_response 가 기대하는 페이로드
        payload = {
            "request_id": str(request_id),  # UUID는 JSON 전송 시 문자열로 변환
            "content": content
        }
        logger.info(f"Sending AI response (ReqID: {request_id}) to callback: {callback_url}")

        try:
            # 주입받은 세션을 사용하여 POST 요청 전송
            async with self._session.post(callback_url, json=payload) as response:
                # 200번대 응답이 아니면 에러로 처리
                response.raise_for_status()
                logger.info(f"Successfully sent response via callback (ReqID: {request_id}). Status: {response.status}")
                return True
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP Error sending response (ReqID: {request_id}) to {callback_url}: {e.status} - {e.message}", exc_info=True)
            return False
        except aiohttp.ClientError as e: # 네트워크/연결 관련 에러
            logger.error(f"Client Connection/Request Error sending response (ReqID: {request_id}) to {callback_url}: {e}", exc_info=True)
            return False
        except Exception as e: # 기타 예상치 못한 에러
            logger.error(f"Unexpected error sending response (ReqID: {request_id}) to {callback_url}: {e}", exc_info=True)
            return False