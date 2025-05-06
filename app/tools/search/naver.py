# app/tools/search/naver.py
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Dict, Optional
from app.config.settings import settings # 설정 임포트
from app.utils.logger import get_logger # 로거 임포트

logger = get_logger("NaverSearchTool") # 로거 초기화 (클래스 이름 사용 권장)

class NaverSearchTool:
    """Naver Search API(뉴스)와 상호작용하는 도구입니다."""

    def __init__(self):
        """NaverSearchTool 초기화"""
        # 설정에서 Naver API 인증 정보 로드
        self.client_id = settings.NAVER_CLIENT_ID
        self.client_secret = settings.NAVER_CLIENT_SECRET
        # Naver 뉴스 검색 API 엔드포인트
        self.search_url = "https://openapi.naver.com/v1/search/news.json"
        # 인증 정보 누락 시 오류 로깅
        if not self.client_id or not self.client_secret:
            logger.error("settings에 Naver Client ID 또는 Secret이 설정되지 않았습니다.")

    @retry(
        stop=stop_after_attempt(settings.TOOL_RETRY_ATTEMPTS), # 설정된 재시도 횟수 사용
        wait=wait_exponential(multiplier=1, min=settings.TOOL_RETRY_WAIT_MIN, max=settings.TOOL_RETRY_WAIT_MAX), # 지수적 대기
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)), # 네트워크/타임아웃 오류 시 재시도
        reraise=True # 모든 재시도 실패 시 예외 다시 발생
    )
    async def search_news(self, keyword: str, max_results: int, session: aiohttp.ClientSession, trace_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Naver Search API를 사용하여 뉴스를 검색합니다.
        Node 03 (News Collector)에서 주로 사용될 것으로 예상됩니다.

        Args:
            keyword (str): 검색어.
            max_results (int): 반환할 최대 결과 수 (API 최대 100개).
            session (aiohttp.ClientSession): HTTP 요청을 위한 aiohttp 세션.
            trace_id (Optional[str]): 로깅을 위한 추적 ID.

        Returns:
            List[Dict[str, str]]: 뉴스 정보(url, title, description, source, search_keyword 등)를 담은 사전 목록.
        """
        # 인증 정보 없으면 빈 리스트 반환
        if not self.client_id or not self.client_secret: return []
        extra_log_data = {'trace_id': trace_id, 'keyword': keyword} # 로깅용 추가 데이터
        logger.debug(f"Naver 뉴스 검색 중: '{keyword}'", extra=extra_log_data)

        # API 파라미터 설정 (display: 결과 수, start: 시작 위치, sort: 정렬 - date: 날짜순)
        params = {"query": keyword, "display": min(max_results, 100), "start": 1, "sort": "date"}
        # API 요청 헤더 설정 (Client ID, Secret 포함)
        headers = {"X-Naver-Client-Id": self.client_id, "X-Naver-Client-Secret": self.client_secret}
        http_timeout = settings.TOOL_HTTP_TIMEOUT # 설정에서 타임아웃 값 가져오기

        try:
            # GET 요청 실행
            async with session.get(self.search_url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=http_timeout)) as response:
                response_text = await response.text() # 오류 로깅 위해 미리 텍스트 읽기
                # 오류 상태 코드 확인 (raise_for_status는 4xx, 5xx에 대해 예외 발생)
                response.raise_for_status()
                try:
                     data = await response.json(content_type=None) # content_type 무시하고 JSON 파싱 시도
                except ValueError: # JSON 파싱 실패 시
                     logger.error(f"Naver API 응답 JSON 파싱 실패. 상태: {response.status}. 응답: {response_text[:200]}...", extra=extra_log_data)
                     return [] # 빈 리스트 반환

                items = data.get("items", []) # 결과 항목 리스트 가져오기
                results = []
                # 각 항목 처리
                for item in items:
                    # 원본 링크(originallink) 우선 사용, 없으면 네이버 링크(link) 사용
                    url = item.get("originallink") or item.get("link", "")
                    if url: # URL이 유효한 경우
                        # HTML 태그 제거 (기본적인 처리)
                        title = item.get('title', '').replace('<b>', '').replace('</b>', '')
                        description = item.get('description', '').replace('<b>', '').replace('</b>', '')
                        # 결과 추가
                        results.append({
                            "url": url,
                            "title": title, # 처리된 제목
                            "description": description, # 처리된 설명
                            "pub_date": item.get('pubDate'), # 게시 날짜
                            "source": "NAVER News", # 출처 명시
                            "search_keyword": keyword # 검색 키워드 저장
                        })
                logger.debug(f"Naver 뉴스 검색: '{keyword}'에 대해 {len(results)}개의 URL을 찾았습니다.", extra=extra_log_data)
                return results # 결과 리스트 반환

        except (aiohttp.ClientError, asyncio.TimeoutError) as e: # 네트워크 또는 타임아웃 오류
             logger.error(f"Naver 검색 중 오류 발생: '{keyword}': {e}", extra=extra_log_data)
             raise # 재시도 위해 예외 발생
        except Exception as e: # 예상치 못한 오류
             logger.exception(f"Naver 검색 중 예상치 못한 오류 발생: '{keyword}': {e}", extra=extra_log_data)
             return [] # 빈 리스트 반환

    # async def close(self): # aiohttp 세션은 외부에서 관리되므로 close 불필요
    #     pass