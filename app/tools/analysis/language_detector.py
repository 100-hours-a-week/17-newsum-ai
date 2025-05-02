# app/tools/analysis/language_detector.py
from typing import Optional

from app.config.settings import settings
from app.utils.logger import get_logger

# langdetect 동적 임포트
LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect as langdetect_detect, LangDetectException, DetectorFactory
    # 재현성을 위한 시드 설정 (선택 사항)
    # DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LangDetectException = None # type: ignore
    # 더미 함수 정의
    def langdetect_detect(text: str) -> str: return 'und'

logger = get_logger(__name__)

if not LANGDETECT_AVAILABLE:
    logger.warning("langdetect 라이브러리가 설치되지 않았습니다. LanguageDetectionTool이 'und'를 반환합니다.")

class LanguageDetectionTool:
    """langdetect 라이브러리를 사용하여 텍스트의 언어를 감지하는 도구입니다."""

    def __init__(self):
        """LanguageDetectionTool 초기화."""
        self.min_length = settings.MIN_LANGDETECT_TEXT_LENGTH # 설정에서 최소 길이 로드

    def detect(self, text: str, trace_id: Optional[str] = None, comic_id: Optional[str] = None) -> str:
        """
        주어진 텍스트의 언어 코드를 반환합니다.

        Args:
            text (str): 언어를 감지할 텍스트.
            trace_id (Optional[str]): 로깅용 추적 ID.
            comic_id (Optional[str]): 로깅용 코믹 ID.

        Returns:
            str: 감지된 언어 코드 (예: 'en', 'ko') 또는 실패 시 'und'.
        """
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        # 라이브러리 없거나 텍스트가 너무 짧으면 'und' 반환
        if not LANGDETECT_AVAILABLE: return 'und'
        if not text or len(text.strip()) < self.min_length:
            logger.debug(f"언어 감지를 위한 텍스트가 너무 짧음 (길이: {len(text.strip()) if text else 0})", extra=extra_log_data)
            return 'und'

        try:
            # 언어 감지 실행 (텍스트 일부 사용 가능)
            lang_code = langdetect_detect(text[:2000])
            logger.debug(f"감지된 언어: {lang_code}", extra=extra_log_data)
            return lang_code
        except LangDetectException:
             # 언어 특징을 찾지 못한 경우
             logger.warning(f"언어 감지 실패 (특징 없음?) - text: '{text[:50]}...'. 'und' 반환.", extra=extra_log_data)
             return 'und'
        except Exception as e:
            # 기타 예상치 못한 오류
            logger.error(f"언어 감지 중 예상치 못한 오류 발생: {e}", exc_info=True, extra=extra_log_data)
            return 'und'