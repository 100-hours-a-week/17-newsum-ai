# ai/app/nodes_v2/n06a_community_reaction_scraping_node.py
from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional

from app.workflows.state_v2 import WorkflowState, IdeaSection
from app.utils.logger import get_logger, summarize_for_logging
from app.tools.search.Google_Search_tool import GoogleSearchTool  # 실제 검색 도구

logger = get_logger(__name__)
NODE_ORDER = 7  # N06 다음, N06B 이전 순서 (예시)
MAX_REACTIONS_PER_PLATFORM_SCRAPE = 10  # 플랫폼별 수집할 최대 반응(스니펫) 수


class N06ACommunityReactionScrapingNode:
    """
    (N06A - 신규) N06에서 분석된 이슈 키워드를 사용하여 주요 온라인 커뮤니티에서
    실제 사용자 반응(게시글 스니펫 등)을 수집합니다.
    """

    def __init__(self, search_tool: GoogleSearchTool):
        self.search_tool = search_tool
        # 타겟 플랫폼 및 검색 시 사용할 사이트 한정어
        self.target_platforms = {
            "dcinside": "dcinside.com",  # 실제 디시인사이드 검색은 어려울 수 있음
            "reddit": "reddit.com",
            "x_twitter": "twitter.com"  # X(트위터) 검색은 API 제한 매우 심함
            # 추가 플랫폼 정의 가능
        }

    async def _scrape_platform_reactions(self, platform_site: str, keywords: List[str], original_query: str,
                                         work_id: str, extra_log: dict) -> List[str]:
        """특정 플랫폼(사이트)에서 키워드 기반으로 반응(스니펫)을 검색/수집합니다."""
        scraped_texts: List[str] = []
        if not keywords and not original_query: return []  # 검색어가 없으면 반환

        # 여러 키워드를 조합하거나, 가장 핵심적인 키워드 + 원본 쿼리 일부 사용
        search_query = f"{' '.join(keywords[:2])} {original_query.split()[0] if original_query else ''} site:{platform_site}"
        logger.debug(f"플랫폼 '{platform_site}' 검색 실행. 쿼리: '{search_query}'", extra=extra_log)

        try:
            # search_specific_sites_via_cse 또는 search_web_via_cse (site: 연산자 사용)
            # 여기서는 search_web_via_cse를 사용 (CSE 설정에 따라 결과 달라짐)
            results = await self.search_tool.search_web_via_cse(
                keyword=search_query,
                max_results=MAX_REACTIONS_PER_PLATFORM_SCRAPE,
                trace_id=work_id,
                dateRestrict="m1"  # 예: 최근 1개월 데이터 우선
            )
            for res_item in results:
                # 스니펫이나 제목을 반응으로 간주. 실제로는 더 정교한 파싱 필요.
                reaction_text = res_item.get('snippet') or res_item.get('title')
                if reaction_text:
                    scraped_texts.append(reaction_text.strip())
            logger.info(f"플랫폼 '{platform_site}'에서 {len(scraped_texts)}개의 반응 스니펫 수집.", extra=extra_log)
        except Exception as e:
            logger.error(f"플랫폼 '{platform_site}' 반응 수집 중 오류: {e}", extra=extra_log)
        return scraped_texts

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        meta_sec = state.meta
        idea_sec = state.idea
        query_sec = state.query  # 원본 쿼리 참고용

        work_id = meta_sec.work_id
        node_name = self.__class__.__name__
        extra_log_base = {"work_id": work_id, "node_name": node_name, "node_order": NODE_ORDER}

        meta_sec.workflow_status[NODE_ORDER] = "PROCESSING"
        logger.info(f"{node_name} 진입: 커뮤니티 실제 반응 수집 시작.", extra=extra_log_base)

        if not idea_sec.structured_issue_analysis or \
                not idea_sec.structured_issue_analysis.get("extracted_keywords"):
            logger.warning("이슈 분석 결과 또는 추출된 키워드가 없어 커뮤니티 반응 수집을 건너<0xEB><0x9C><0x84>니다.", extra=extra_log_base)
            meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"  # 또는 SKIPPED
            idea_sec.scraped_community_reactions = {}  # 빈 값으로 초기화
            return {"meta": meta_sec.model_dump(), "idea": idea_sec.model_dump(exclude_unset=True)}

        keywords_from_analysis = idea_sec.structured_issue_analysis.get("extracted_keywords", [])
        original_query_for_context = query_sec.original_query or ""

        scraping_tasks = []
        for platform_key, platform_domain in self.target_platforms.items():
            scraping_tasks.append(
                self._scrape_platform_reactions(
                    platform_domain, keywords_from_analysis, original_query_for_context, work_id, extra_log_base
                    # type: ignore
                )
            )

        logger.info(f"{len(scraping_tasks)}개 플랫폼에 대한 반응 수집 병렬 시작.", extra=extra_log_base)
        platform_scraped_results = await asyncio.gather(*scraping_tasks, return_exceptions=True)

        final_scraped_reactions: Dict[str, List[str]] = {}
        platform_keys = list(self.target_platforms.keys())  # 원래 순서대로 플랫폼 이름 매칭

        for idx, result_or_exc in enumerate(platform_scraped_results):
            platform_name = platform_keys[idx]
            if isinstance(result_or_exc, Exception):
                logger.error(f"플랫폼 '{platform_name}' 반응 수집 작업 중 예외: {result_or_exc}", extra=extra_log_base)
                final_scraped_reactions[platform_name] = []  # 실패 시 빈 리스트
            elif isinstance(result_or_exc, list):
                final_scraped_reactions[platform_name] = result_or_exc
            else:  # 예상치 못한 타입
                logger.warning(f"플랫폼 '{platform_name}' 반응 수집 결과가 리스트가 아님: {type(result_or_exc)}", extra=extra_log_base)
                final_scraped_reactions[platform_name] = []

        idea_sec.scraped_community_reactions = final_scraped_reactions
        meta_sec.workflow_status[NODE_ORDER] = "COMPLETED"
        logger.info(f"{node_name} 완료: 커뮤니티 실제 반응 수집 완료.", extra=extra_log_base)
        return {"meta": meta_sec.model_dump(), "idea": idea_sec.model_dump(exclude_unset=True)}