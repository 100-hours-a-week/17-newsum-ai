# ai/app/services/spam_service.py
import re
from typing import Optional

from app.config.settings import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = Settings()

class SpamService:
    """
    텍스트가 스팸인지 여부를 판별하는 서비스입니다.
    (현재는 Placeholder 규칙 기반 로직 사용. 향후 ML 모델 로딩 및 예측으로 대체 필요)
    """
    def __init__(self):
        """SpamDetectionService 초기화."""
        # TODO: 실제 ML 모델 및 벡터라이저 로딩 로직 추가
        # try:
        #     self.model = joblib.load(settings.SPAM_MODEL_PATH)
        #     self.vectorizer = joblib.load(settings.SPAM_VECTORIZER_PATH)
        #     logger.info("스팸 탐지 ML 모델 로드 완료.")
        # except Exception as e:
        #     logger.error(f"스팸 탐지 ML 모델 로드 실패: {e}. 규칙 기반으로 대체합니다.", exc_info=True)
        #     self.model = None
        #     self.vectorizer = None
        self.model = None # Placeholder
        self.vectorizer = None # Placeholder

        # 규칙 기반 필터링용 설정 로드
        self.spam_keywords = settings.SPAM_KEYWORDS
        self.max_url_count = settings.SPAM_MAX_URL_COUNT
        self.max_uppercase_ratio = settings.SPAM_MAX_UPPERCASE_RATIO
        logger.info("규칙 기반 스팸 탐지기 초기화됨 (Placeholder).")


    def is_spam(self, text: str, trace_id: Optional[str] = None, comic_id: Optional[str] = None) -> bool:
        """
        주어진 텍스트가 스팸인지 예측합니다.

        Args:
            text (str): 검사할 텍스트.
            trace_id (Optional[str]): 로깅용 추적 ID.
            comic_id (Optional[str]): 로깅용 코믹 ID.

        Returns:
            bool: 스팸이면 True, 아니면 False.
        """
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}

        # TODO: 실제 ML 모델 예측 로직 구현
        # if self.model and self.vectorizer:
        #     try:
        #         features = self.vectorizer.transform([text])
        #         prediction = self.model.predict(features)
        #         is_spam_pred = bool(prediction[0] == 1) # 스팸 레이블이 1이라고 가정
        #         logger.debug(f"ML 스팸 예측: {is_spam_pred}", extra=extra_log_data)
        #         return is_spam_pred
        #     except Exception as e:
        #         logger.error(f"ML 스팸 예측 중 오류 발생: {e}. 규칙 기반으로 대체합니다.", extra=extra_log_data)
        #         # 오류 발생 시 안전하게 스팸 아님으로 처리하거나 규칙 기반으로 넘어감

        # --- 규칙 기반 Placeholder 로직 ---
        if not text or len(text.strip()) < 15: return False # 너무 짧으면 스팸 아님 간주
        text_lower = text.lower()

        # 1. 키워드 검사
        if any(keyword in text_lower for keyword in self.spam_keywords):
            logger.debug("스팸 키워드 발견", extra=extra_log_data)
            return True
        # 2. URL 개수 검사
        url_count = len(re.findall(r'https?://\S+', text)) # 정규식으로 URL 개수 확인
        if url_count > self.max_url_count:
            logger.debug(f"허용된 URL 개수 초과 ({url_count} > {self.max_url_count})", extra=extra_log_data)
            return True
        # 3. 대문자 비율 검사 (일정 길이 이상 텍스트에만 적용)
        text_len = len(text.strip())
        if text_len > 30:
            uppercase_ratio = sum(1 for c in text if c.isupper()) / text_len
            if uppercase_ratio > self.max_uppercase_ratio:
                logger.debug(f"대문자 비율 임계값 초과 ({uppercase_ratio:.2f} > {self.max_uppercase_ratio})", extra=extra_log_data)
                return True

        # 모든 규칙 통과 시 스팸 아님
        return False