# ai/app/nodes_v3/n_03_search_execution_node.py (최종 수정 버전)
"""n_03_SearchExecutionNode (Final Version)

NewSum LangGraph 워크플로우의 **세 번째 노드**(순번 n_03).
'지능형 연구원 에이전트'로서, n_02에서 설계된 보고서 구조(`structure`)를 입력받아,
각 섹션에 필요한 정보를 수집 및 가공한다.

주요 워크플로우:
1.  **한/영 검색어 생성**: 'role', 'description' 등의 문맥을 활용해 한국어와 영어 검색어를 모두 생성한다.
2.  **한/영 동시 검색**: 생성된 검색어로 각 언어별 웹 검색을 수행하여 관련성 높은 URL을 찾는다.
3.  **전체 텍스트 추출**: 검색된 URL의 전체 텍스트를 스크래핑하고 길이를 계산한다.
4.  **결과 매핑 및 저장**: 추출된 전체 텍스트와 메타데이터를 원본 목차 구조와 매핑하여 `state`에 저장한다.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup
from pydantic import ValidationError

from app.config.settings import Settings
from app.services.database_client import DatabaseClient
from app.services.llm_service import LLMService
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.utils.logger import get_logger
from app.workflows.state_v3 import (
    OverallWorkflowState,
)

# ──────────────────────────────────────────────────────────────────────────────
# 설정 및 상수
# ──────────────────────────────────────────────────────────────────────────────
settings = Settings()
MAX_CONTENT_LENGTH = 15000  # 처리할 최대 콘텐츠 길이 증가
SEARCH_RESULTS_PER_SECTION = 5
logger = get_logger("n_03_SourceCollectNode_Final")


# ──────────────────────────────────────────────────────────────────────────────
# 노드 구현
# ──────────────────────────────────────────────────────────────────────────────
class N03SearchExecutionNode:
    """LangGraph async 노드 – 소스 수집: 한/영 검색 및 전체 텍스트 추출 (n_03)"""

    def __init__(
            self,
            redis_client: DatabaseClient,
            llm_service: LLMService,
            search_tool: Optional[GoogleSearchTool] = None,
    ):
        self.redis = redis_client
        self.llm = llm_service
        self.search_tool = search_tool or GoogleSearchTool()
        self.logger = logger

    async def __call__(self, current_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        work_id = current_state_dict.get("work_id")
        log_extra = {"work_id": work_id or "UNKNOWN_WORK_ID_N03"}
        self.logger.info("N03SourceCollectNode (Final) 시작.", extra=log_extra)

        try:
            workflow_state = OverallWorkflowState(**current_state_dict)
        except ValidationError as e:
            self.logger.error(f"N03 State 유효성 검사 실패: {e}", extra=log_extra)
            current_state_dict["error_message"] = f"N03 State 유효성 검사 실패: {e}"
            return current_state_dict

        await self._run_source_collection_pipeline(workflow_state)
        return await self._finalize_and_save_state(workflow_state, log_extra)

    async def _run_source_collection_pipeline(self, workflow_state: OverallWorkflowState):
        """보고서의 각 섹션에 대해 한/영 검색 및 전체 텍스트 추출 파이프라인을 실행합니다."""
        work_id = workflow_state.work_id
        log_extra = {"work_id": work_id}
        self.logger.info("소스 수집 파이프라인 시작.", extra=log_extra)

        structure = workflow_state.report_planning.structure
        source_collect_map = {}

        for sec_title, sec_content in structure.items():
            self.logger.info(f"섹션 '{sec_title}' 처리 시작.", extra=log_extra)

            queries = await self._generate_optimized_search_queries(sec_content, work_id)

            ko_urls, en_urls = [], []
            if queries.get("ko"):
                ko_urls = await self._execute_contextual_search(queries['ko'], work_id, lang='ko')
            if queries.get("en"):
                en_urls = await self._execute_contextual_search(queries['en'], work_id, lang='en')

            if not ko_urls and not en_urls:
                self.logger.warning(f"섹션 '{sec_title}'에 대한 한/영 검색 결과가 모두 없습니다.", extra=log_extra)
                continue

            tasks = [self._process_url(url, 'ko', work_id) for url in ko_urls] + \
                    [self._process_url(url, 'en', work_id) for url in en_urls]

            processed_articles = await asyncio.gather(*tasks)
            source_collect_map[sec_title] = [result for result in processed_articles if result]

        workflow_state.source_collect.results = source_collect_map
        workflow_state.source_collect.is_ready = True
        self.logger.info("모든 섹션의 소스 수집 및 전체 텍스트 추출 완료.", extra=log_extra)

    async def _generate_optimized_search_queries(self, sec_content: Dict[str, Any], work_id: str) -> Dict[str, str]:
        """섹션의 전체 문맥을 기반으로 한국어와 영어 검색어를 생성합니다."""
        log_extra = {"work_id": work_id}
        role = sec_content.get("role", "")
        description = sec_content.get("description", "")

        if not description:
            return {"ko": "", "en": ""}

        prompt = f"""You are an expert search query strategist. Based on the provided context of a report section, generate two optimal search queries: one in Korean and one in English.

            **Context of the Report Section:**
            * **Role of this section:** "{role}"
            * **Core content (description):** "{description}"

            **Task:**
            Generate two detailed and comprehensive search queries (Korean and English). Each query should be **at least 80 characters long** and phrased to find in-depth analyses, case studies, or expert opinions on the topic.
            Return a single JSON object with two keys: "ko" for the Korean query and "en" for the English query.

            **Example Output:**
            {{
              "ko": "언론 산업의 디지털 전환 과정에서 AI 기술 도입이 수익 모델에 미치는 구체적인 영향과 성공적인 비즈니스 사례 분석",
              "en": "In-depth analysis of the impact of AI technology adoption on revenue models in the news media industry's digital transformation and case studies"
            }}

            Your output **must be only the JSON object**.
        """
        try:
            self.logger.info("LLM에 한/영 최적 검색어 생성을 요청합니다.", extra=log_extra)
            resp = await self.llm.generate_text(
                messages=[{"role": "user", "content": prompt}],
                request_id=f"optimize-queries-dual-lang-{work_id}",
                max_tokens=1024,
                temperature=0.1
            )
            queries = json.loads(resp.get("generated_text", "{}").strip())

            ko_query = queries.get("ko", "").replace('"', '')
            en_query = queries.get("en", "").replace('"', '')

            # Fallback: LLM이 빈 문자열을 반환하면 description을 사용
            if not ko_query:
                ko_query = description

            self.logger.info(f"한/영 검색어 생성 완료: KO='{ko_query}', EN='{en_query}'", extra=log_extra)
            return {"ko": ko_query, "en": en_query}
        except Exception as e:
            self.logger.error(f"한/영 검색어 생성 중 오류: {e}. fallback으로 description 사용.", extra=log_extra)
            return {"ko": description, "en": ""}

    async def _execute_contextual_search(self, query: str, work_id: str, lang: str) -> List[Dict[str, str]]:
        """주어진 쿼리와 언어 설정으로 웹 검색을 수행하고 URL 리스트를 반환합니다."""
        self.logger.info(f"'{lang.upper()}' 문맥 검색 실행: '{query}'", extra={"work_id": work_id})
        try:
            # GoogleSearchTool에 언어 제한 파라미터(lr) 전달
            lang_param = f'lang_{lang}'
            search_results = await self.search_tool.search_web_via_cse(
                query,
                max_results=SEARCH_RESULTS_PER_SECTION,
                trace_id=work_id,
                lr=lang_param
            )
            results_with_meta = [
                {"url": r.get('url'), "title": r.get('title'), "snippet": r.get('snippet')}
                for r in search_results if r.get('url')
            ]
            # urls = [result['url'] for result in search_results if 'url' in result]
            self.logger.info(f"{len(results_with_meta)}개의 URL 검색 완료.", extra={"work_id": work_id})
            return results_with_meta
        except Exception as e:
            self.logger.error(f"검색 실행 중 오류: {e}", extra={"work_id": work_id})
            return []

    async def _process_url(self, source_info: Dict[str, str], lang: str, work_id: str) -> Optional[Dict[str, Any]]:
        """
        단일 URL의 전체 텍스트를 추출하고 메타데이터(title, snippet)를 포함한 구조화된 데이터를 반환합니다.

        Args:
            source_info (Dict[str, str]): 'url', 'title', 'snippet'을 포함한 딕셔너리
            lang (str): 검색된 언어 ('ko' 또는 'en')
            work_id (str): 작업의 고유 ID

        Returns:
            Optional[Dict[str, Any]]: 처리된 아티클의 상세 정보 또는 실패 시 None
        """
        url = source_info.get("url")
        if not url:
            self.logger.warning("source_info에 URL이 없어 처리를 건너뜁니다.", extra={"work_id": work_id})
            return None

        log_extra = {"work_id": work_id, "url": url}
        self.logger.info("URL 처리 시작.", extra=log_extra)

        full_text = await self._fetch_and_parse_url(url, work_id)
        if not full_text:
            self.logger.warning("콘텐츠를 가져올 수 없어 처리를 중단합니다.", extra=log_extra)
            return None

        text_length = len(full_text)

        self.logger.info(f"URL 처리 및 전체 텍스트 추출 성공 (길이: {text_length}).", extra=log_extra)

        # 최종 결과물에 title과 snippet을 포함하여 반환
        return {
            "source_url": url,
            "language": lang,
            "title": source_info.get("title", ""),  # title 필드 추가
            "snippet": source_info.get("snippet", ""),  # snippet 필드 추가
            "full_text": full_text,
            "text_length": text_length
        }

    async def _fetch_and_parse_url(self, url: str, work_id: str) -> Optional[str]:
        """URL의 HTML을 가져와 본문 텍스트를 파싱합니다."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        main_content = soup.find('main') or soup.find('article') or soup.find('body')
                        if main_content:
                            text = ' '.join(
                                p.get_text(strip=True) for p in main_content.find_all(['p', 'h1', 'h2', 'h3']))
                            return text[:MAX_CONTENT_LENGTH]  # 최대 길이 제한
                        return None
                    else:
                        self.logger.warning(f"URL GET 실패, 상태 코드: {response.status}",
                                            extra={"work_id": work_id, "url": url})
                        return None
        except Exception as e:
            self.logger.error(f"URL 스크래핑 중 오류: {e}", extra={"work_id": work_id, "url": url})
            return None

    async def _finalize_and_save_state(self, workflow_state: OverallWorkflowState, log_extra: Dict) -> Dict[str, Any]:
        """최종 상태를 저장하고 반환합니다."""
        updated_state_dict = workflow_state.model_dump(mode='json')
        await self._save_workflow_state_to_redis(workflow_state.work_id, updated_state_dict)
        self.logger.info("N03 노드 처리 완료 및 상태 저장.", extra=log_extra)
        return updated_state_dict

    async def _save_workflow_state_to_redis(self, work_id: str, state_dict: Dict[str, Any]):
        key = f"workflow:{work_id}:full_state"
        try:
            json_compatible_state = json.loads(json.dumps(state_dict, default=str))
            await self.redis.set(key, json_compatible_state, expire=60 * 60 * 6)
        except Exception as e:
            self.logger.error(f"Redis 상태 저장 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})