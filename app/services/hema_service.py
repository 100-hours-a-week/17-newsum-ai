# ai/app/services/hema_service.py

import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio
import re  # 간단한 텍스트 처리를 위한 정규식 모듈

from app.clients.front_backend_api_client import FrontBackendAPIClient
from app.config.settings import settings
from app.utils.logger import get_logger
from app.api.v2.schemas.hema_models import (
    InformationSnippetSchema,
    IdeaNodeSchema,
    SummaryNodeSchema,
    HEMAInternalInteractionLogSchema,
    HEMABulkOperationRequest,
    HEMABulkOperation,
    HEMAOperationType,
    HEMAEntityType,
    HEMAContext,
    HEMAContextItem,
    HEMAContextSummary,
    InteractionEventType
)

logger = get_logger(__name__)


class HEMAService:
    """HEMA 데이터 관리 및 RDB 기반 컨텍스트 구성 서비스 (대화방 요약 중심).

    주요 책임:
    - 앞단 백엔드 API 클라이언트를 통해 RDB에서 특정 대화방(세션)의 HEMA 데이터 조회 (특히 대화 요약).
    - 조회된 데이터를 기반으로 SLM에 전달할 HEMA 컨텍스트 구성.
        - 우선순위 및 토큰 예산에 따른 컨텍스트 아이템 필터링 및 정렬.
        - 컨텍스트 요약 정보 생성.
    - HEMA 데이터 변경사항을 앞단 백엔드 API를 통해 RDB에 일괄 저장.
    - 주요 상호작용 로그 기록.
    - HEMA 컨텍스트를 포함한 최종 SLM 프롬프트 생성.
    """

    def __init__(self, front_backend_client: FrontBackendAPIClient):
        """
        HEMAService 초기화.

        Args:
            front_backend_client: 앞단 백엔드 서버(RDB 인터페이스)와 통신하기 위한 API 클라이언트.
        """
        self.front_backend_client = front_backend_client

    async def retrieve_hema_context(
            self,
            user_id: str,
            session_id: str,
            user_query: str,  # 현재 사용자 쿼리는 직접적인 검색 키워드로는 덜 사용되나, 로깅 및 일부 휴리스틱에 활용 가능
            metadata: Optional[Dict[str, Any]] = None
    ) -> HEMAContext:
        """
        특정 대화방(session_id)의 HEMA 컨텍스트를 RDB에서 조회하여 구성합니다.
        주로 해당 대화방의 이전 대화 요약들을 가져오고, 우선순위 및 토큰 예산을 고려합니다.

        Args:
            user_id: 사용자 ID.
            session_id: 현재 대화방(세션) ID. 이 ID를 기준으로 관련 데이터 조회.
            user_query: 사용자의 현재 입력 쿼리.
            metadata: 요청과 관련된 추가 메타데이터 (예: 작업 유형).

        Returns:
            구성된 HEMAContext 객체. 오류 발생 시 빈 컨텍스트 객체 반환.
        """
        try:
            context_id = str(uuid.uuid4())
            context_items: List[HEMAContextItem] = []

            logger.info(
                f"Retrieving HEMA context for session_id: {session_id}, user_id: {user_id}, query: '{user_query[:50]}...'")

            # 1. HEMA 데이터 병렬 조회 (앞단 백엔드 API - RDB 연동)
            # 이 세션(대화방)과 관련된 요약, 아이디어, 정보 조각, 로그를 가져옴.
            # API 클라이언트의 각 get_* 메소드가 session_id를 주요 필터 조건으로 사용한다고 가정.

            # 이 대화방의 최근 요약들을 가져옴 (가장 중요한 컨텍스트 소스)
            summaries_task = self.front_backend_client.get_summary_nodes(
                user_id=user_id,  # user_id도 함께 전달하여 권한 확인 등에 사용 가능
                session_id=session_id,  # 핵심 필터 조건
                # summary_type 등을 추가 필터 조건으로 사용할 수 있음 (예: "turn_summary")
                # 앞단 API에서 최신순으로 정렬하여 반환한다고 가정
                limit=settings.HEMA_MAX_SUMMARIES_PER_CONTEXT if hasattr(settings,
                                                                         'HEMA_MAX_SUMMARIES_PER_CONTEXT') else 3
            )

            # 이 대화방에서 생성/언급된 아이디어들을 가져옴
            ideas_task = self.front_backend_client.get_idea_nodes(
                user_id=user_id,
                session_id=session_id,  # 핵심 필터 조건
                # status="confirmed" 등 추가 필터 가능
                limit=settings.HEMA_MAX_IDEAS_PER_CONTEXT
            )

            # 이 대화방에서 참조된 정보 조각들을 가져옴
            snippets_task = self.front_backend_client.get_information_snippets(
                user_id=user_id,
                session_id=session_id,  # 핵심 필터 조건
                # keywords 파라미터는 사용자 쿼리에서 추출한 키워드를 전달할 수 있으나,
                # 여기서는 대화방 전체 맥락에 집중하므로, session_id 필터링이 더 중요.
                # 필요시 user_query에서 추출한 키워드를 보조적으로 사용 가능.
                keywords=self._extract_keywords_for_filtering(
                    user_query) if settings.USE_QUERY_KEYWORDS_FOR_SNIPPETS else None,
                limit=settings.HEMA_MAX_SNIPPETS_PER_CONTEXT
            )

            # 이 대화방의 최근 상호작용 로그 (SLM에 직접 전달할 맥락보다는, 요약 생성의 기반이 됨)
            logs_task = self.front_backend_client.get_interaction_logs(
                user_id=user_id,
                session_id=session_id,  # 핵심 필터 조건
                limit=settings.HEMA_MAX_LOGS_PER_CONTEXT if hasattr(settings, 'HEMA_MAX_LOGS_PER_CONTEXT') else 5
            )

            results = await asyncio.gather(
                summaries_task, ideas_task, snippets_task, logs_task,
                return_exceptions=True  # 개별 작업 실패 시에도 계속 진행
            )

            summaries_result, ideas_result, snippets_result, logs_result = results

            # 2. 안전한 결과 처리 및 컨텍스트 아이템 변환
            # _convert_to_context_items 메소드는 이제 사용자 쿼리 키워드 대신, 아이템 자체의 특성으로 relevance를 판단
            if not isinstance(summaries_result, Exception):
                context_items.extend(self._convert_to_context_items(summaries_result, HEMAEntityType.SUMMARY_NODE))
            else:
                logger.warning(f"Failed to fetch summaries for session {session_id}: {summaries_result}")

            if not isinstance(ideas_result, Exception):
                context_items.extend(self._convert_to_context_items(ideas_result, HEMAEntityType.IDEA_NODE))
            else:
                logger.warning(f"Failed to fetch ideas for session {session_id}: {ideas_result}")

            if not isinstance(snippets_result, Exception):
                context_items.extend(
                    self._convert_to_context_items(snippets_result, HEMAEntityType.INFORMATION_SNIPPET))
            else:
                logger.warning(f"Failed to fetch snippets for session {session_id}: {snippets_result}")

            if not isinstance(logs_result, Exception):
                # 대화 로그는 보통 우선순위를 낮게 설정하거나, 요약의 형태로 이미 반영되었을 수 있음
                context_items.extend(
                    self._convert_to_context_items(logs_result, HEMAEntityType.HEMA_INTERNAL_INTERACTION_LOG))
            else:
                logger.warning(f"Failed to fetch interaction logs for session {session_id}: {logs_result}")

            logger.debug(f"Initial context items count for session {session_id}: {len(context_items)}")

            # 3. 우선순위 및 토큰 예산 기반 필터링 및 정렬
            final_context_items = self._filter_by_priority_and_budget(
                context_items,
                settings.HEMA_CONTEXT_TOKEN_BUDGET
            )
            logger.debug(f"Filtered context items count for session {session_id}: {len(final_context_items)}")

            # 4. 컨텍스트 요약 정보 생성
            context_summary = self._generate_context_summary(final_context_items)
            logger.debug(f"Generated context summary for session {session_id}: {context_summary}")

            return HEMAContext(
                context_id=context_id,
                user_id=user_id,
                session_id=session_id,
                query=user_query,  # 원본 사용자 쿼리도 함께 저장
                items=final_context_items,
                summary=context_summary,
                timestamp_created=datetime.now()
            )

        except Exception as e:
            logger.exception(f"Failed to retrieve HEMA context for session_id {session_id}: {e}")
            # 오류 발생 시 빈 컨텍스트 반환
            return HEMAContext(
                context_id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=session_id,
                query=user_query,
                items=[],
                summary=HEMAContextSummary(
                    total_items=0,
                    total_tokens=0,
                    items_by_type={},
                    average_relevance=0.0,
                    context_quality_score=0.0
                ),
                timestamp_created=datetime.now()
            )

    def _extract_keywords_for_filtering(self, text: str) -> List[str]:
        """
        (선택적) 정보 조각 등 특정 HEMA 데이터 조회 시 보조 필터링을 위한 키워드 추출.
        대화방 요약 중심 컨텍스트에서는 이 기능의 중요도가 낮을 수 있음.
        """
        words = re.findall(r'\b\w{3,}\b', text.lower())
        stopwords = ['그것', '그런', '이런', '저런', '그리고', '하지만', '그러나', '그래서', '나는', '너는', '우리는', '입니다', '합니다', '있는', '것은']
        keywords = [w for w in words if w not in stopwords]
        unique_keywords = list(dict.fromkeys(keywords))
        max_keywords = settings.MAX_KEYWORDS_FOR_SNIPPET_FILTERING if hasattr(settings,
                                                                              'MAX_KEYWORDS_FOR_SNIPPET_FILTERING') else 3
        logger.debug(
            f"Keywords extracted for optional filtering from '{text[:30]}...': {unique_keywords[:max_keywords]}")
        return unique_keywords[:max_keywords]

    def _assign_relevance_and_priority(
            self,
            item: Any,  # 원본 HEMA 데이터 스키마 객체
            entity_type: HEMAEntityType
    ) -> Tuple[float, int]:
        """
        주어진 HEMA 아이템과 타입에 따라 (단순) 관련도 점수와 우선순위를 할당합니다.
        대화방 요약 중심 컨텍스트에서는 아이템의 최신성, 상태, 타입 등이 주요 판단 기준이 됩니다.
        """
        relevance_score = 0.5  # 기본 관련도
        priority = 2  # 기본 우선순위 (1이 가장 높음)

        # 아이템 타입별 기본 관련도 및 우선순위 설정
        if entity_type == HEMAEntityType.SUMMARY_NODE:
            relevance_score = 0.8  # 대화방 요약은 기본적으로 관련성이 높다고 간주
            priority = 1  # 가장 높은 우선순위
            # TODO: 요약의 최신성(timestamp)에 따라 relevance_score나 priority 가중치 부여 가능
            # 예: if (datetime.now() - item.timestamp_generated).days < 1: relevance_score += 0.1
        elif entity_type == HEMAEntityType.IDEA_NODE and isinstance(item, IdeaNodeSchema):
            relevance_score = 0.7  # 아이디어도 비교적 중요
            priority = 1 if getattr(item, 'status', "") == "confirmed" else 2  # 확정된 아이디어 우선
        elif entity_type == HEMAEntityType.INFORMATION_SNIPPET and isinstance(item, InformationSnippetSchema):
            relevance_score = 0.6
            priority = 1 if getattr(item, 'status', "") == "verified" else 2  # 검증된 정보 우선
        elif entity_type == HEMAEntityType.HEMA_INTERNAL_INTERACTION_LOG:
            relevance_score = 0.4  # 개별 로그는 요약보다 덜 중요할 수 있음
            priority = 3  # 낮은 우선순위

        # TODO: 아이템의 내용이나 메타데이터를 기반으로 점수 미세 조정 로직 추가 가능
        # 예: if '중요' in item.title.lower(): priority = 0 (가장 높음)

        return round(relevance_score, 2), priority

    def _convert_to_context_items(
            self,
            source_items: List[Any],
            entity_type: HEMAEntityType
    ) -> List[HEMAContextItem]:
        """
        원본 HEMA 데이터 스키마 객체 리스트를 HEMAContextItem 리스트로 변환합니다.
        relevance_score와 priority는 _assign_relevance_and_priority를 통해 할당됩니다.
        """
        context_items = []
        if not source_items: return context_items

        for item in source_items:
            item_id = ""
            content = ""

            relevance_score, priority = self._assign_relevance_and_priority(item, entity_type)

            if entity_type == HEMAEntityType.INFORMATION_SNIPPET and isinstance(item, InformationSnippetSchema):
                item_id = item.snippet_id
                content = f"[정보] {item.title}\n요약: {item.summary_text}"
            elif entity_type == HEMAEntityType.IDEA_NODE and isinstance(item, IdeaNodeSchema):
                item_id = item.idea_id
                node_type_value = item.node_type.value if isinstance(item.node_type,
                                                                     HEMAEntityType) else item.node_type  # Enum 또는 str 처리
                content = f"[아이디어:{node_type_value}] {item.title}\n설명: {item.description}"
            elif entity_type == HEMAEntityType.SUMMARY_NODE and isinstance(item, SummaryNodeSchema):
                item_id = item.summary_id
                summary_type_value = item.summary_type.value if isinstance(item.summary_type,
                                                                           HEMAEntityType) else item.summary_type
                content = f"[대화 요약:{summary_type_value}] {item.title}\n내용: {item.summary_text}"  # "대화 요약"으로 명시
            elif entity_type == HEMAEntityType.HEMA_INTERNAL_INTERACTION_LOG and isinstance(item,
                                                                                            HEMAInternalInteractionLogSchema):
                item_id = item.log_id
                event_type_value = item.event_type.value if isinstance(item.event_type,
                                                                       InteractionEventType) else item.event_type
                content = f"[대화 기록:{event_type_value}] {item.timestamp.strftime('%Y-%m-%d %H:%M')}\n{item.content_summary}"

            if item_id and content:
                token_count = len(content) // 2  # 매우 대략적인 토큰 수
                context_items.append(HEMAContextItem(
                    item_id=item_id,
                    item_type=entity_type,
                    content=content,
                    relevance_score=relevance_score,
                    token_count=token_count,
                    priority=priority
                ))
        return context_items

    def _filter_by_priority_and_budget(
            self,
            items: List[HEMAContextItem],
            token_budget: int
    ) -> List[HEMAContextItem]:
        """
        컨텍스트 아이템들을 우선순위와 (단순)관련도 점수로 정렬한 후,
        주어진 토큰 예산 내에서 아이템을 선택합니다.
        """
        # 우선순위(낮을수록 높음) -> 관련도 점수(높을수록 높음) 순으로 정렬
        items.sort(key=lambda x: (x.priority, -x.relevance_score))

        selected_items: List[HEMAContextItem] = []
        current_total_tokens = 0

        for item in items:
            if current_total_tokens + item.token_count <= token_budget:
                selected_items.append(item)
                current_total_tokens += item.token_count
            else:
                logger.debug(
                    f"Token budget ({token_budget}) reached. Selected {len(selected_items)} items with {current_total_tokens} tokens.")
                break

        if not selected_items and items:
            logger.warning(
                f"No items selected due to token budget ({token_budget}), though {len(items)} items were initially present.")

        return selected_items

    def _generate_context_summary(self, items: List[HEMAContextItem]) -> HEMAContextSummary:
        """
        선택된 컨텍스트 아이템들에 대한 요약 정보(HEMAContextSummary)를 생성합니다.
        """
        total_items = len(items)
        total_tokens = sum(item.token_count for item in items)

        items_by_type: Dict[str, int] = {}
        for item in items:
            type_name = item.item_type.value if isinstance(item.item_type, HEMAEntityType) else str(item.item_type)
            items_by_type[type_name] = items_by_type.get(type_name, 0) + 1

        average_relevance = sum(item.relevance_score for item in items) / total_items if total_items > 0 else 0.0
        quality_score = self._calculate_context_quality_session_focused(items, total_tokens, average_relevance)

        return HEMAContextSummary(
            total_items=total_items,
            total_tokens=total_tokens,
            items_by_type=items_by_type,
            average_relevance=round(average_relevance, 2),
            context_quality_score=quality_score
        )

    def _calculate_context_quality_session_focused(self, items: List[HEMAContextItem], total_tokens: int,
                                                   average_relevance: float) -> float:
        """
        대화방 요약 중심 컨텍스트의 품질 점수를 계산합니다.
        """
        if not items:
            return 0.0

        # 다양성: 얼마나 다양한 종류의 정보(요약, 아이디어, 로그 등)가 포함되었나
        expected_max_types = len(HEMAEntityType)  # 또는 주요 3-4개 타입
        unique_types_count = len(set(item.item_type for item in items))
        diversity_score = min(unique_types_count / float(expected_max_types), 1.0) if expected_max_types > 0 else 0.0

        # 평균 관련도: 아이템들의 평균적인 (단순) 관련도
        avg_relevance_component = average_relevance

        # 토큰 활용도: 주어진 예산을 얼마나 잘 사용했나
        token_utilization = min(total_tokens / float(settings.HEMA_CONTEXT_TOKEN_BUDGET),
                                1.0) if settings.HEMA_CONTEXT_TOKEN_BUDGET > 0 else 0.0

        # 최신성 지표 (예시): 가장 최근 요약/로그가 얼마나 새로운지 (별도 계산 필요)
        # float: 가장 최근 아이템의 생성 시간으로부터 현재까지 경과된 시간(정규화)
        # recency_score = self._calculate_recency_score(items)

        # 가중치: 관련도 30%, 다양성 30%, 토큰 활용도 30%, (최신성 10% - 추가 시)
        quality_score = (avg_relevance_component * 0.3) + (diversity_score * 0.3) + (
                    token_utilization * 0.3)  # + (recency_score * 0.1)
        return round(quality_score, 2)

    async def save_hema_updates(
            self,
            user_id: str,
            session_id: str,
            updates: List[Dict[str, Any]]
    ) -> bool:
        """HEMA 데이터 변경사항 일괄 저장 (이전 버전과 거의 동일, 주석 및 로깅 강화)"""
        try:
            if not updates:
                logger.info("No HEMA updates to save.")
                return True

            operations: List[HEMABulkOperation] = []
            for i, update_item in enumerate(updates):
                try:
                    action_str = update_item.get('action', HEMAOperationType.CREATE.value).lower()  # 기본값 CREATE
                    action = HEMAOperationType(action_str)

                    entity_type_str = update_item['entity_type']
                    entity_type = HEMAEntityType(entity_type_str)
                except (ValueError, KeyError) as e:
                    logger.error(
                        f"Invalid action or entity_type in HEMA update item: {update_item}. Error: {e}. Skipping this item.")
                    continue

                operation = HEMABulkOperation(
                    operation_id=f"{session_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{i}",
                    action=action,
                    entity_type=entity_type,
                    entity_id=update_item.get('entity_id'),
                    data=update_item.get('data'),
                    metadata=update_item.get('metadata', {})
                )
                operations.append(operation)

            if not operations:
                logger.warning("No valid HEMA operations to save after filtering invalid items.")
                return False

            request_payload = HEMABulkOperationRequest(
                user_id=user_id,
                session_id=session_id,
                operations=operations,
                request_id=str(uuid.uuid4())
            )
            logger.info(
                f"Attempting to save {len(operations)} HEMA updates for user_id {user_id}, session_id {session_id}.")

            response = await self.front_backend_client.bulk_operations(request_payload)

            if response and hasattr(response, 'processed_count') and hasattr(response, 'failed_count'):
                if response.processed_count > 0:
                    success_items = response.processed_count - response.failed_count
                    success_rate = success_items / float(response.processed_count)

                    threshold = settings.HEMA_BULK_SUCCESS_THRESHOLD if hasattr(settings,
                                                                                'HEMA_BULK_SUCCESS_THRESHOLD') else 0.8
                    if success_rate >= threshold:
                        logger.info(
                            f"HEMA updates saved. Success rate: {success_rate * 100:.2f}% ({success_items}/{response.processed_count}).")
                        return True
                    else:
                        logger.warning(
                            f"HEMA updates partially failed. Success rate: {success_rate * 100:.2f}% ({response.failed_count} failures / {response.processed_count} processed). Details: {response.results if hasattr(response, 'results') else 'N/A'}")
                        return False
                elif len(operations) > 0:
                    logger.warning(
                        f"No HEMA updates were processed by the front-backend API, though {len(operations)} operations were sent. Response: {response.dict(exclude_none=True) if response else 'None'}")
                    return False
                else:
                    return True
            else:
                logger.error(
                    f"Failed to save HEMA updates: Invalid or no response from front_backend_client.bulk_operations. Response: {response.dict(exclude_none=True) if response else 'None'}")
                return False

        except Exception as e:
            logger.exception(
                f"Unexpected error in save_hema_updates for user_id {user_id}, session_id {session_id}: {e}")
            return False

    async def log_interaction(
            self,
            user_id: str,
            session_id: str,
            event_type: InteractionEventType,
            content_summary: str,
            linked_hema_ids: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """상호작용 로그 기록 (이전 버전과 거의 동일, 주석 및 로깅 강화)"""
        try:
            log_entry_data = {
                'log_id': str(uuid.uuid4()),
                'session_id': session_id,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type.value,
                'content_summary': content_summary,
                'linked_hema_ids': linked_hema_ids or [],
                'metadata': metadata or {}
            }

            update_payload = {
                'action': HEMAOperationType.CREATE.value,
                'entity_type': HEMAEntityType.HEMA_INTERNAL_INTERACTION_LOG.value,
                'data': log_entry_data
            }
            logger.info(
                f"Logging interaction for session {session_id}: event_type={event_type.value}, summary='{content_summary[:50]}...'")
            return await self.save_hema_updates(user_id, session_id, [update_payload])

        except Exception as e:
            logger.exception(f"Failed to log interaction for session {session_id}, event_type {event_type.value}: {e}")
            return False

    def create_prompt_with_context(
            self,
            user_message: str,
            hema_context: HEMAContext,
            task_type: str = "general"
    ) -> str:
        """HEMA 컨텍스트를 포함한 프롬프트 생성 (이전 버전과 거의 동일, 주석 및 로깅 강화)"""
        try:
            system_prompt = self._get_system_prompt(task_type)
            context_section = self._format_context_section(hema_context)
            user_section = f"\n\n### 사용자 현재 요청\n{user_message}"
            assistant_prompt_cue = "\n\n### AI 어시스턴트 답변:\n"
            full_prompt = f"{system_prompt}\n{context_section}{user_section}{assistant_prompt_cue}"

            logger.info(
                f"Created prompt for task_type '{task_type}' for session {hema_context.session_id if hema_context else 'N/A'}. Approx. prompt length (chars): {len(full_prompt)}")
            # TODO: 실제 토크나이저로 프롬프트 토큰 수 계산 및 로깅
            return full_prompt

        except Exception as e:
            logger.exception(f"Failed to create prompt with context: {e}")
            return f"시스템 프롬프트 생성 중 오류가 발생했습니다. 관리자에게 문의해주세요.\n\n사용자 요청: {user_message}"

    def _get_system_prompt(self, task_type: str) -> str:
        """작업 타입별 시스템 프롬프트 반환 (이전 버전과 거의 동일, 로깅 추가)"""
        prompts = {
            "general": "당신은 사용자의 창의적인 아이디어를 구체화하고, 웹툰 제작 과정을 돕는 전문 AI 어시스턴트입니다. 제공된 이전 대화 요약 및 관련 정보를 바탕으로 사용자에게 유용하고 실행 가능한 제안을 해주세요. 답변은 명확하고 친절한 어조로 작성해주세요.",
            "아이디어_생성": "당신은 혁신적인 웹툰 아이디어 발상 전문가입니다. 사용자의 요청과 제공된 이전 대화 요약 및 정보를 참고하여, 독창적이고 시장성 있는 스토리 아이디어를 최소 3가지 이상 제안해주세요. 각 아이디어는 주요 컨셉, 타겟 독자, 예상 장르를 포함해야 합니다.",
            "캐릭터_개발": "당신은 깊이 있는 캐릭터를 창조하는 작가입니다. 사용자의 캐릭터 컨셉과 관련 이전 대화 요약 및 정보를 바탕으로, 해당 캐릭터의 외형, 성격, 배경 이야기, 주요 갈등, 성장 가능성 등을 구체적으로 묘사하고 발전시켜주세요.",
            "스토리_구성": "당신은 흡입력 있는 스토리텔링 전문가입니다. 사용자의 스토리 아이디어나 줄거리를 기반으로, 이전 대화 요약 및 관련 정보를 참고하여 플롯의 주요 단계(발단, 전개, 위기, 절정, 결말), 핵심 사건, 복선, 반전 등을 포함한 탄탄한 스토리 구조를 제안해주세요.",
            "정보_요약": "당신은 주어진 정보를 명확하고 간결하게 요약하는 AI입니다. 제공된 컨텍스트를 바탕으로 핵심 내용을 정확히 요약해주세요.",
            "대화_요약": "당신은 이전 대화 내용을 간결하게 요약하는 AI입니다. 다음 대화를 위해 현재까지의 논의 내용을 핵심만 추려 요약해주세요."  # 대화 요약 생성용 프롬프트 추가
        }
        selected_prompt = prompts.get(task_type.lower(), prompts["general"])
        logger.debug(f"Selected system prompt for task_type '{task_type}': '{selected_prompt[:50]}...'")
        return selected_prompt

    def _format_context_section(self, hema_context: HEMAContext) -> str:
        """HEMAContext 객체를 SLM 프롬프트에 포함하기 적절한 문자열 형태로 포맷팅합니다."""
        if not hema_context or not hema_context.items:
            logger.debug("No HEMA context items to format for prompt.")
            return "\n### 이전 대화 요약 및 관련 정보\n현재 참고할 만한 이전 대화 요약이나 관련 정보가 없습니다."

        sections: List[str] = []
        sections.append("\n### 이전 대화 요약 및 관련 정보 (HEMA 컨텍스트)")
        if hema_context.summary:
            sections.append(f"(총 {hema_context.summary.total_items}개 항목, "
                            f"단순 일치도 평균: {hema_context.summary.average_relevance:.2f}, "  # 대화방 요약 중심에서는 이 점수의 의미가 다름
                            f"품질 점수: {hema_context.summary.context_quality_score:.2f})")

        # 아이템 타입별로 그룹화하여 표시 (가독성 향상)
        items_by_type: Dict[HEMAEntityType, List[HEMAContextItem]] = {}
        for item in hema_context.items:
            items_by_type.setdefault(item.item_type, []).append(item)

        for entity_type, type_items in items_by_type.items():
            if entity_type == HEMAEntityType.SUMMARY_NODE:
                sections.append("\n--- 이전 대화 요약 ---")
            elif entity_type == HEMAEntityType.IDEA_NODE:
                sections.append("\n--- 관련 아이디어 ---")
            elif entity_type == HEMAEntityType.INFORMATION_SNIPPET:
                sections.append("\n--- 관련 정보 조각 ---")
            elif entity_type == HEMAEntityType.HEMA_INTERNAL_INTERACTION_LOG:
                sections.append("\n--- 최근 대화 기록 ---")

            for i, item in enumerate(type_items):
                # 우선순위와 점수를 표시하여 SLM이 중요도를 인지하도록 유도 가능
                sections.append(
                    f"\n항목 {i + 1} (우선순위: {item.priority}, 관련성: {item.relevance_score:.2f}):\n{item.content}")

        formatted_section = "\n".join(sections)
        logger.debug(f"Formatted HEMA context section for prompt (first 100 chars): {formatted_section[:100]}...")
        return formatted_section

    async def close(self):
        """HEMAService가 사용하는 리소스(예: FrontBackendAPIClient)를 정리합니다."""
        try:
            if self.front_backend_client:
                await self.front_backend_client.close()
            logger.info("HEMAService and its clients closed successfully.")
        except Exception as e:
            logger.exception(f"Error closing HEMAService: {e}")

