# ai/app/nodes_v3/n_01_topic_clarification_node.py
from __future__ import annotations

import json
import re  # 의도 후보 파싱을 위해 추가
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

from pydantic import ValidationError

from app.config.settings import Settings
from app.services.database_client import DatabaseClient
from app.services.llm_service import LLMService
from app.services.postgresql_service import PostgreSQLService
from app.utils.logger import get_logger
from app.tools.search.Google_Search_tool import GoogleSearchTool
from app.workflows.state_v3 import OverallWorkflowState, TopicClarificationPydanticState

settings = Settings()

DRAFT_TARGET_LEN: int = 500
DRAFT_ALLOWED_DEVIATION: int = 50
MIN_KEYWORD_COUNT: int = 3
REDIS_WORKFLOW_KEY_TMPL: str = "workflow:{work_id}:full_state"
LLM_TIMEOUT_SEC: int = 60  # LLM 호출 시간 증가 고려
SEARCH_TOP_K: int = 5

logger = get_logger("n_01_TopicClarificationNode")


class N01TopicClarificationNode:
    def __init__(
            self,
            redis_client: DatabaseClient,
            llm_service: LLMService,
            pg_service: Optional[PostgreSQLService] = None,
            google_search_tool: Optional[GoogleSearchTool] = None,

    ):
        self.redis = redis_client
        self.llm = llm_service
        self.pg = pg_service
        self.search_tool = google_search_tool or GoogleSearchTool()
        self.logger = logger
        self.current_work_id: Optional[str] = None  # work_id를 저장하기 위한 멤버 변수


    async def __call__(
            self,
            current_state_dict: Dict[str, Any],
            user_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw_work_id = current_state_dict.get("work_id")
        log_extra = {"work_id": raw_work_id if raw_work_id else "UNKNOWN_WORK_ID"}
        self.logger.info(
            f"N01TopicClarificationNode.__call__ 시작. user_response: '{user_response if user_response else '[없음]'}'.",
            extra=log_extra)

        if not raw_work_id:
            self.logger.error(f"{self.__class__.__name__}: work_id가 current_state_dict에 없습니다.", extra=log_extra)
            current_state_dict["error_message"] = f"{self.__class__.__name__}: work_id 누락."
            return current_state_dict

        self._attach_log_context(raw_work_id)  # 여기서 self.current_work_id 설정됨
        work_id = self.current_work_id  # 이제부터 work_id 변수 사용

        try:
            workflow_state = OverallWorkflowState(**current_state_dict)
            self.logger.info("OverallWorkflowState 파싱 성공.", extra={"work_id": work_id})
        except ValidationError as e:
            self.logger.error(f"Pydantic state 유효성 검사 오류: {e}",
                              extra={"state_dict_keys": list(current_state_dict.keys()), "error_details": e.errors(),
                                     "work_id": work_id})
            current_state_dict["error_message"] = f"State 유효성 검사 실패: {str(e)}"
            return current_state_dict

        node_state = workflow_state.topic_clarification

        if not node_state.trusted_domains:
            self.logger.info("신뢰도 도메인 로드 시도.", extra={"work_id": work_id})
            node_state.trusted_domains = await self._get_trusted_domains(work_id)
            self.logger.info(f"신뢰도 도메인 {len(node_state.trusted_domains)}개 로드 완료.", extra={"work_id": work_id})

        if node_state.intent_clarification_question_outstanding and user_response:
            self.logger.info(f"사용자 의도 선택 응답 처리 시작: '{user_response[:50]}...'", extra={"work_id": work_id})
            await self._process_intent_choice(node_state, user_response, work_id)
            if node_state.chosen_intent_description and not node_state.draft:  # 의도 선택 후, 첫 초안 생성
                self.logger.info("의도 선택 완료, 첫 주제 초안 생성 시도.", extra={"work_id": work_id})
                await self._initial_or_subsequent_iteration(node_state, workflow_state.user_query, work_id)

        elif not node_state.chosen_intent_description and not node_state.draft and workflow_state.user_query and not node_state.intent_clarification_question_outstanding:
            self.logger.info("새 사용자 쿼리, 초기 의도 명확화 단계 시작.", extra={"work_id": work_id})
            await self._clarify_initial_intent(node_state, workflow_state.user_query, work_id)

        elif node_state.question and user_response and not node_state.intent_clarification_question_outstanding:
            self.logger.info(f"사용자 초안 피드백 처리 시작: '{user_response[:50]}...'", extra={"work_id": work_id})
            await self._process_user_response(node_state, workflow_state.user_query, user_response,
                                              work_id)  # user_query 전달

        elif node_state.chosen_intent_description and not node_state.is_final and not node_state.question:
            self.logger.info("주제 초안 생성/갱신 루프 시작 (의도 확정 후 또는 내부 반복).", extra={"work_id": work_id})
            await self._initial_or_subsequent_iteration(node_state, workflow_state.user_query, work_id)

        else:
            self.logger.info("특정 액션 조건 불충족, 현 상태 유지 또는 점검 필요.", extra={"work_id": work_id})

        workflow_state.topic_clarification = node_state
        updated_state_dict = workflow_state.model_dump(exclude_none=True)
        await self._save_workflow_state_to_redis(work_id, updated_state_dict)

        self.logger.info("N01TopicClarificationNode.__call__ 종료.", extra={"work_id": work_id})
        return updated_state_dict

    # _clarify_initial_intent 메서드 부분만 수정하여 표시합니다.
    # 전체 코드 구조는 이전 답변을 참고하시고, 이 메서드만 아래 내용으로 교체하십시오.

    async def _clarify_initial_intent(self, node_state: TopicClarificationPydanticState, user_query: str, work_id: str):
            self.logger.info(f"'{user_query}'에 대한 초기 의도 분석 시작 (만평 주제의 조사 관점 명확화).", extra={"work_id": work_id})

            # 시스템 프롬프트 (영문): '만평'을 위한 '조사 관점/탐사 방향' 제안 역할 명시
            prompt_sys = (
                "You are an AI assistant. Your task is to analyze a user's query, which may relate to current events or specific incidents, "
                "and propose exactly 2 distinct **perspectives or angles for investigation and critical commentary**, suitable as a starting point for developing "  # 'investigation' 강조
                "an **editorial cartoon, satirical comic, or social commentary cartoon (만평)**. "
                "Each proposed perspective should clearly articulate an **insightful angle for inquiry, identify a potential target of satire/critique, or suggest a core message** that could be explored and fact-checked for the manpyeong. "  # 'inquiry', 'fact-checked' 추가
                "Present each perspective as a numbered list. Each perspective MUST include relevant keywords in Korean, enclosed in parentheses, like (키워드: 예시1, 예시2). "
                "Follow the example answer format provided in the user prompt precisely. "
                "Your entire response MUST consist ONLY of the numbered list of 2 perspectives, and this output MUST be in Korean."
            )

            # 사용자 프롬프트: '만평'을 위한 '세부 주제/관점' 또는 '탐사 방향'을 묻도록 수정, 예시도 이에 맞게 변경
            prompt_user = (
                f"사용자 질문: '{user_query}'\n\n"
                "위 사용자 질문을 주제로 하여, 특정 사건이나 최근 이슈에 대한 만평을 제작한다고 가정했을 때, "
                "어떤 **'세부 주제'나 '핵심 관점'**을 가지고 심층적인 조사를 시작하면 좋을지, 가능한 탐사 방향 3가지를 설명해주십시오. "  # '세부 주제/핵심 관점', '탐사 방향' 명시
                "각 탐사 방향은 해당 이슈의 어떤 측면을 깊이 파고들거나 어떤 메시지를 중점적으로 전달할 수 있는지 간략히 언급하고, "
                "관련 핵심 키워드를 괄호 안에 포함하여 한 문장으로 작성해주십시오. (답변은 반드시 한국어로 작성해주세요)\n\n"
                "예시:\n"  # 사용자가 제공한 통찰력 있는 만평 아이디어의 '탐사 방향' 버전으로 수정
                "사용자 질문: 'AI 모델 학습 데이터 무단 활용 논란'\n"
                "답변:\n"
                "1. AI 학습 데이터 확보 과정에서 나타나는 일부 기업들의 윤리적 문제와 '데이터 약탈'로 비유될 수 있는 행태의 사실 관계 및 그 파장을 심층적으로 탐사하는 방향. (키워드: 데이터 약탈, AI 윤리, 기업 책임)\n"  # '탐사하는 방향'으로 명시
                "2. AI 기술 경쟁 심화가 데이터 수집 및 활용 방식에 미치는 영향과 이것이 업계의 공정성 및 사용자 신뢰에 주는 장기적인 함의를 비판적으로 분석하고 탐사하는 방향. (키워드: AI 기술 경쟁, 데이터 주권, 사용자 신뢰)\n"
            # '탐사하는 방향'으로 명시
            )

            messages = [
                {"role": "system", "content": prompt_sys},
                {"role": "user", "content": prompt_user},
            ]
            self.logger.debug(f"LLM 의도 분석 요청 메시지 (만평 조사 관점): {json.dumps(messages, ensure_ascii=False, indent=2)[:500]}",
                              extra={"work_id": work_id})

            try:
                resp = await self.llm.generate_text(
                    messages=messages,
                    request_id=f"intent-clarify-manpyeong-angle-{work_id}",  # request_id에 angle 명시
                    max_tokens=768,
                    temperature=0.5,  # 이전 버전(0.6)보다 약간 낮춰서 너무 발산하지 않고 '조사 관점'에 집중하도록 조정
                    timeout=LLM_TIMEOUT_SEC,
                )
                # 이하 LLM 응답 파싱 및 상태 업데이트 로직은 이전 답변과 동일하게 유지됩니다.
                # (raw_intents_text, parsed_intents, node_state.potential_intents, node_state.question 등 설정)
                raw_intents_text = resp.get("generated_text", "").strip()
                self.logger.info(f"LLM 의도 분석 응답 수신 (만평 조사 관점): '{raw_intents_text}'", extra={"work_id": work_id})

                parsed_intents = []
                intent_matches = re.findall(r"^\s*(\d+)\.\s*(.+?)(?:\s*\((?:키워드|keyword|Keyword):\s*(.+?)\))?\s*$",
                                            raw_intents_text, re.MULTILINE | re.DOTALL)

                if not intent_matches and raw_intents_text:
                    lines = raw_intents_text.split('\n')
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if re.match(r"^\s*\d+\.", line):
                            parsed_intents.append({"id": str(i + 1), "description": line})
                else:
                    for match in intent_matches:
                        intent_id = match[0]
                        description_text = match[1].strip()
                        keywords_text = match[2].strip() if match[2] else ""
                        full_description = f"{description_text}"
                        if keywords_text:
                            full_description += f" (키워드: {keywords_text})"
                        parsed_intents.append({"id": intent_id, "description": full_description})

                if not parsed_intents and raw_intents_text:
                    self.logger.warning("조사 관점 분석 결과 파싱 실패, LLM 응답을 단일 선택지로 사용 시도.", extra={"work_id": work_id})
                    parsed_intents.append({"id": "1", "description": raw_intents_text})

                if parsed_intents:
                    node_state.potential_intents = parsed_intents
                    options_for_user_question = "\n".join(
                        [f"{intent['id']}. {intent['description']}" for intent in parsed_intents])

                    # 사용자에게 제시하는 질문의 뉘앙스도 '탐사 방향'이나 '조사 관점' 선택으로 수정
                    node_state.question = (
                        f"'{user_query}'에 대해 다음과 같은 조사 관점(탐사 방향)들을 생각해 보았습니다. "
                        "어떤 관점으로 더 깊이 있는 조사를 진행해볼까요? 번호를 선택해주시거나, 다른 의견이 있다면 직접 작성해주세요.\n\n"
                        f"{options_for_user_question}\n\n"
                        f"{len(parsed_intents) + 1}. 위에 제시된 관점과 다릅니다 (직접 설명)."
                    )
                    node_state.intent_clarification_question_outstanding = True
                    self.logger.info(f"사용자에게 의도 선택 질문 생성 완료 (만평 조사 관점). 질문: {node_state.question[:100]}...",
                                     extra={"work_id": work_id})
                else:
                    self.logger.warning("LLM으로부터 유효한 만평 조사 관점 후보를 생성하거나 파싱하지 못했습니다.", extra={"work_id": work_id})
                    node_state.question = "죄송합니다. 현재 질문에 대한 만평의 조사 관점을 설정하는 데 어려움이 있습니다. 어떤 부분을 중점적으로 탐구하고 싶으신지 조금 더 자세히 말씀해주시겠어요?"
                    node_state.chosen_intent_description = user_query
                    node_state.intent_clarification_question_outstanding = False

            except Exception as e:
                self.logger.error(f"LLM 만평 조사 관점 분석 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})
                node_state.question = "만평 조사 관점 구상 중 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
                node_state.chosen_intent_description = user_query
                node_state.intent_clarification_question_outstanding = False

    async def _process_intent_choice(self, node_state: TopicClarificationPydanticState, user_response: str, work_id: str):
        self.logger.info(f"사용자 의도 선택 응답 처리: '{user_response}'", extra={"work_id": work_id})
        user_choice_str = user_response.strip().lower()

        chosen_option = None
        other_option_selected = False

        if node_state.potential_intents:
            try:
                match = re.match(r"(\d+)", user_choice_str)
                if match:
                    choice_id_str = match.group(1)
                    if choice_id_str == str(len(node_state.potential_intents) + 1):
                        other_option_selected = True
                    else:
                        chosen_option = next(
                            (intent for intent in node_state.potential_intents if intent["id"] == choice_id_str), None)
            except ValueError:
                self.logger.warning(f"사용자 의도 선택 '{user_response}'를 숫자로 변환 실패.", extra={"work_id": work_id})

        if chosen_option:
            node_state.chosen_intent_description = chosen_option["description"]
            self.logger.info(f"사용자가 의도 선택: '{node_state.chosen_intent_description}'", extra={"work_id": work_id})
        elif other_option_selected:
            user_direct_intent = re.sub(r"^\d+\.\s*", "", user_response).strip()
            if len(user_direct_intent) > 10:  # "3. " 이후 설명이 충분히 길다면
                node_state.chosen_intent_description = user_direct_intent
                self.logger.info(f"사용자가 '기타' 옵션 선택 후 직접 의도 설명: '{node_state.chosen_intent_description}'",
                                 extra={"work_id": work_id})
            else:  # "3."만 입력했거나 설명이 너무 짧은 경우
                self.logger.warning(f"사용자가 '기타'를 선택했으나 설명이 불충분: '{user_response}'.", extra={"work_id": work_id})
                node_state.question = "선택하신 '기타' 옵션에 대한 구체적인 설명을 부탁드립니다. 어떤 주제나 방향을 생각하고 계신가요?"
                node_state.intent_clarification_question_outstanding = True  # 다시 질문 상태 유지
                return  # 여기서 함수 종료
        elif len(user_response) > 10:  # 숫자로 시작하지 않고, 충분히 긴 텍스트면 직접 의도로 간주
            node_state.chosen_intent_description = user_response
            self.logger.info(f"사용자가 직접 의도 입력: '{node_state.chosen_intent_description}'", extra={"work_id": work_id})
        else:
            self.logger.warning(f"사용자 의도 선택이 명확하지 않음: '{user_response}'. 첫번째 제안된 의도로 진행.", extra={"work_id": work_id})
            if node_state.potential_intents:  # fallback
                node_state.chosen_intent_description = node_state.potential_intents[0]['description']
            else:  # 의도 후보도 없는 경우 (이런 경우는 거의 없어야 함)
                node_state.chosen_intent_description = user_response  # 최후의 수단
                self.logger.error("의도 후보가 없는 상태에서 명확하지 않은 사용자 응답 발생.", extra={"work_id": work_id})

        node_state.potential_intents = None
        node_state.question = None
        node_state.intent_clarification_question_outstanding = False

    async def _initial_or_subsequent_iteration(self, node_state: TopicClarificationPydanticState, user_query: str,
                                               work_id: str):
        self.logger.info("_initial_or_subsequent_iteration 시작.", extra={"work_id": work_id})

        query_to_process = node_state.chosen_intent_description if node_state.chosen_intent_description else user_query.strip()

        if not query_to_process:
            self.logger.error("처리할 사용자 쿼리 또는 선택된 의도가 없어 진행 불가.", extra={"work_id": work_id})
            node_state.question = "죄송합니다만, 어떤 주제를 다루고 싶은지 다시 입력해 주실 수 있을까요?"
            return

        self.logger.debug(f"처리할 질의 (선택된 의도 또는 원본 쿼리): '{query_to_process}'.", extra={"work_id": work_id})

        self.logger.info("정보원 검색 시작.", extra={"work_id": work_id})
        sources = await self._fetch_initial_sources(query_to_process, user_query, node_state.trusted_domains, work_id,
                                                    node_state.chosen_intent_description)
        node_state.sources = sources
        self.logger.info(f"{len(sources)}개의 정보원 검색 완료.", extra={"work_id": work_id})
        context_blob = self._aggregate_search_context(sources, work_id)  # work_id 전달

        self.logger.info("LLM으로 주제 초안 생성 시작.", extra={"work_id": work_id})
        draft_text = await self._generate_topic_draft(original_user_query=user_query,
                                                      processed_query_or_intent=query_to_process,
                                                      context_blob=context_blob,
                                                      chosen_intent=node_state.chosen_intent_description,
                                                      work_id=work_id)
        node_state.draft = draft_text
        self.logger.info(f"LLM 주제 초안 생성 완료. 초안 길이: {len(draft_text)}자.", extra={"work_id": work_id})

        self.logger.info("주제 초안 품질 검사 시작.", extra={"work_id": work_id})
        if self._is_topic_sufficient(draft_text, work_id):  # work_id 전달
            node_state.is_final = True
            node_state.question = None
            self.logger.info(f"주제 확정됨 (길이: {len(draft_text)}자). is_final: {node_state.is_final}.",
                             extra={"work_id": work_id})
        else:
            node_state.is_final = False
            node_state.question = self._formulate_followup_question(draft_text, node_state.chosen_intent_description,
                                                                    user_query, work_id)
            self.logger.info(f"추가 설명 필요 – 후속 질문 생성. 질문: '{node_state.question[:50]}...'", extra={"work_id": work_id})
        self.logger.info("_initial_or_subsequent_iteration 종료.", extra={"work_id": work_id})


    async def _fetch_initial_sources(self, query_for_search: str, original_user_query: str, trusted_domains: List[str],
                                     work_id: str, chosen_intent: Optional[str]) -> List[Dict[str, str]]:
        self.logger.info(
            f"_fetch_initial_sources 시작. 검색 질의: '{query_for_search}'. 원본 쿼리: '{original_user_query}'. 의도: '{chosen_intent if chosen_intent else '[없음]'}'.",
            extra={"work_id": work_id})

        search_keywords_str = query_for_search

        if chosen_intent:
            prompt_sys = ("Your task is to extract 2 to 3 main keywords from the given Korean sentence. "
                          "Respond ONLY with a comma-separated list of these keywords. "
                          "The keywords themselves MUST be in Korean. Do not add any other text or explanation.")
            prompt_user = f"Sentence: '{chosen_intent}'\n\nExtract keywords in Korean:"
            messages = [{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt_user}]
            try:
                resp = await self.llm.generate_text(messages=messages, request_id=f"kw-extract-{work_id}", max_tokens=300,
                                                    temperature=0.1, timeout=LLM_TIMEOUT_SEC / 2)
                extracted_kws_text = resp.get("generated_text", "").strip()
                if extracted_kws_text:
                    self.logger.info(f"의도에서 추출된 키워드: '{extracted_kws_text}'", extra={"work_id": work_id})
                    search_keywords_str = f"{query_for_search} {extracted_kws_text.replace(',', ' ')}"  # 공백으로 구분된 키워드 추가
            except Exception as e:
                self.logger.warning(f"의도에서 키워드 추출 중 오류: {e}. 기존 검색어 사용.", extra={"work_id": work_id})

        self.logger.info(f"최종 검색 키워드 문자열: '{search_keywords_str}'", extra={"work_id": work_id})

        try:
            site_results = []
            if trusted_domains:
                self.logger.debug(f"신뢰도 사이트 검색 ({len(trusted_domains)}개) 시도: {trusted_domains}", extra={"work_id": work_id})
                site_results = await self.search_tool.search_specific_sites_via_cse(
                    keyword=search_keywords_str,
                    sites=trusted_domains,
                    max_results=SEARCH_TOP_K,
                    trace_id=work_id,
                )
                self.logger.info(f"신뢰도 사이트 검색 결과 {len(site_results)}개 수신.", extra={"work_id": work_id})

            news_results = []
            remaining_results_needed = SEARCH_TOP_K - len(site_results)
            if remaining_results_needed > 0:
                self.logger.debug(f"일반 뉴스 검색 ({remaining_results_needed}개) 시도.", extra={"work_id": work_id})
                news_results = await self.search_tool.search_news_via_cse(
                    keyword=search_keywords_str,
                    max_results=remaining_results_needed,
                    trace_id=work_id,
                    dateRestrict="m6"
                )
                self.logger.info(f"일반 뉴스 검색 결과 {len(news_results)}개 수신.", extra={"work_id": work_id})

            combined = site_results + news_results
            final_sources = combined[:SEARCH_TOP_K]
            self.logger.info(f"최종 정보원 {len(final_sources)}개 선택.", extra={"work_id": work_id})
            self.logger.debug(f"선택된 정보원 (일부): {json.dumps(final_sources[:2], ensure_ascii=False, indent=2)}",
                              extra={"work_id": work_id})
            return final_sources
        except Exception as e:
            self.logger.error(f"Google 검색 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})
            now_iso = datetime.now(timezone.utc).isoformat()
            return [{"title": f"{search_keywords_str} – 검색 중 오류 발생",
                     "snippet": f"'{search_keywords_str}'에 대한 정보를 가져오는 데 실패했습니다. ({now_iso})"}]


    async def _generate_topic_draft(self, original_user_query: str, processed_query_or_intent: str, context_blob: str,
                                    chosen_intent: Optional[str], work_id: str) -> str:
        self.logger.info(
            f"_generate_topic_draft 시작. 원본쿼리: '{original_user_query[:50]}...', 처리된쿼리/의도: '{processed_query_or_intent[:50]}...', 선택된의도: '{chosen_intent[:50] if chosen_intent else '[없음]'}', 컨텍스트 길이: {len(context_blob)}.",
            extra={"work_id": work_id})

        intent_info_for_prompt = ""
        if chosen_intent:
            intent_info_for_prompt = f"\n[User's Primary Interest / Chosen Creative Direction]\n{chosen_intent}\n"

        prompt_sys = (
            "You are an expert Korean issue researcher and analyst. Your task is to synthesize the user's request and provided web materials "
            "into a clear and insightful Korean topic statement. The statement should be approximately 450-550 Korean characters. "
            "It must clearly define the main subject, core issue/event, and briefly hint at its significance or potential implications. "
            "Maintain a neutral and objective tone, suitable for further in-depth research. Focus on conciseness and clarity, avoiding jargon or explaining it briefly. "
            "If a User's Primary Interest or Chosen Creative Direction is specified, ensure the topic statement is primarily centered around that interest. "
            "Your final response MUST be ONLY the single, refined Korean topic statement, in Korean."
        )
        prompt_user = (
            f"[User's Original Question (Korean)]\n{original_user_query}\n"
            f"{intent_info_for_prompt}"
            f"[Relevant Web Scraped Information (Korean)]\n{context_blob}\n\n"
            "Based on the information above, please write a Korean topic statement. If the User's Primary Interest is provided, focus on that aspect. The statement should be in Korean."
        )
        messages = [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt_user},
        ]
        self.logger.debug(f"LLM 주제 초안 생성 요청 메시지 (일부): {json.dumps(messages, ensure_ascii=False, indent=2)[:200]}...",
                          extra={"work_id": work_id})

        resp = await self.llm.generate_text(
            messages=messages,
            request_id=f"topic-draft-{work_id}",
            max_tokens=1024,
            temperature=0.3,
            timeout=LLM_TIMEOUT_SEC,
        )
        draft = resp.get("generated_text", "").strip()
        if draft:
            self.logger.info(f"LLM 초안 생성 성공 (길이: {len(draft)}자).", extra={"work_id": work_id})
        else:
            self.logger.error("LLM 초안 생성 실패 (빈 응답).", extra={"work_id": work_id})
        return draft


    def _formulate_followup_question(self, draft: str, chosen_intent: Optional[str], user_query: str, work_id: str) -> str:
        self.logger.debug(f"후속 질문 생성 시작. 초안: '{draft[:50]}...', 의도: '{chosen_intent[:50] if chosen_intent else '[없음]'}'.",
                          extra={"work_id": work_id})
        # (참고: 이 부분은 LLM 호출을 통해 더욱 지능적인 질문 생성 가능. 현재는 템플릿 기반)
        if chosen_intent and draft:
            return (
                f"현재 구체화된 주제는 다음과 같습니다:\n\n\"{draft}\"\n\n"
                f"이 내용은 의도하셨던 방향( '{chosen_intent[:100]}...' )과 잘 맞는 것 같나요? "
                "혹시 추가하고 싶거나, 다르게 표현하고 싶은 부분이 있다면 편하게 말씀해주세요. "
                "없으시다면 이대로 진행하겠습니다."
            )
        return (  # 일반적인 후속 질문
            "현재 시스템은 사용자께서 요청하신 이슈를 다음과 같이 이해하고 있습니다.\n 구체적으로 다뤄야 할 부분이나 추가적인 사항이 필요하다면 의견주세요.\n"
            f"\n{draft}\n\n"
        )

    # n_01_topic_clarification_node.py 내 _classify_user_response_intent 메서드

    async def _classify_user_response_intent(self, current_draft: str, user_answer: str, work_id: str) -> str:
        # 시스템 프롬프트 (영문) - "<THINK>" 태그 등 불필요한 출력 금지 지시 강화
        prompt_sys = (
            "Your **sole and critical task** is to classify a user's Korean feedback on a given Korean draft text into one of three categories. "
            "The categories are: 'CO', 'RE', or 'UN'.\n"
            "1. 'CO': Indicates the user is satisfied with the current draft and implies no further changes are needed (e.g., '네 좋아요', '이대로 진행해주세요').\n"
            "2. 'RE': Indicates the user wants specific modifications, additions, or deletions to the current draft (e.g., 'A를 B로 수정해주세요', '좀 더 자세히 설명해주세요').\n"
            "3. 'UN': Indicates the user's intent isn't clearly one of the above or is a general non-committal comment (e.g., '흠...', '글쎄요').\n\n"
            "**IMPORTANT INSTRUCTIONS FOR YOUR RESPONSE:**\n"
            "- Your response MUST BE **ONLY ONE** of these exact English strings: 'CO', 'RE', or 'UN'.\n"
            "- **DO NOT** include any other text, reasoning, explanations, translations, or conversational filler.\n"
            "- **ABSOLUTELY NO XML-like tags (e.g., <think>, </think>, <REASONING>, <TRANSLATION_ATTEMPT>) are allowed in your output.**\n"
            "Output the single, precise classification label and nothing else."
        )

        # 사용자 프롬프트 (핵심 정보만 전달)
        prompt_user = (
            #f"[Current Draft (Korean)]\n{current_draft}\n\n"
            f"[User's Feedback (Korean)]\n{user_answer}\n\n"
            "Based *only* on the [User's Feedback], classify the user's intent. "
            "Choose and respond with only one of the following labels: 'CO', 'RE', 'UN'."
        )
        messages = [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt_user},
        ]
        self.logger.info(f"LLM 사용자 답변 의도 분류 요청 (강화된 프롬프트). User Answer: '{user_answer[:50]}...'",
                         extra={"work_id": work_id})

        valid_labels = ["CO", "RE", "UN"]

        try:
            resp = await self.llm.generate_text(
                messages=messages,
                request_id=f"intent-classify-strict-{work_id}",  # request_id 변경 가능
                max_tokens=50,  # 라벨 길이를 고려한 최소한의 토큰 수 (예: "CO"은 약 3-4 토큰)
                temperature=0.0,  # 결정적이고 일관된 출력 유도
                timeout=LLM_TIMEOUT_SEC / 4  # 매우 짧은 시간 내 응답 기대
            )
            raw_classification = resp.get("generated_text", "").strip().upper()
            # LLMService에서 <think> 태그 등을 이미 제거한다고 가정.
            # 만약 제거하지 않는다면, 여기서 추가 제거 로직이 필요할 수 있으나,
            # 프롬프트에서 강력히 금지했으므로 우선 LLM이 지시를 따를 것으로 기대.
            self.logger.debug(f"LLM raw classification response (strict prompt): '{raw_classification}'",
                              extra={"work_id": work_id})

            classification = "UN"  # 기본값

            # 1. LLM이 정확히 라벨만 반환했는지 먼저 확인
            if raw_classification in valid_labels:
                classification = raw_classification
            else:
                # 2. 혹시 라벨 주변에 약간의 불필요한 공백이나 예측 못한 문자가 붙었을 경우를 대비하여,
                #    응답 문자열 내에 유효한 라벨이 "포함"되어 있는지 확인.
                #    (이 로직은 LLM이 프롬프트 지시를 완벽히 따르지 못할 경우를 위한 대비책)
                found_label = False
                for label in valid_labels:
                    if label in raw_classification:  # 대소문자 구분 없이 비교하려면 .upper() 등 활용
                        classification = label
                        found_label = True
                        self.logger.warning(
                            f"LLM 의도 분류 시 라벨 외의 텍스트가 포함되었으나, 유효 라벨 '{label}' 추출 성공: '{raw_classification}'.",
                            extra={"work_id": work_id})
                        break
                if not found_label:
                    self.logger.error(f"LLM 의도 분류 결과가 유효한 라벨을 포함하지 않음: '{raw_classification}'. 최종 'UN'로 처리.",
                                      extra={"work_id": work_id})
                    classification = "UN"  # 안전하게 UN로 처리

            self.logger.info(f"LLM 사용자 답변 의도 분류 최종 결과 (강화된 프롬프트): {classification}", extra={"work_id": work_id})
            return classification
        except Exception as e:
            self.logger.error(f"LLM 사용자 답변 의도 분류 중 오류 (강화된 프롬프트): {e}", exc_info=True, extra={"work_id": work_id})
            return "UN"  # 오류 발생 시 안전하게 UN로 처리

    # n_01_topic_clarification_node.py의 _process_user_response 메서드 수정

    async def _process_user_response(self, node_state: TopicClarificationPydanticState, original_user_query: str,
                                     user_answer: str, work_id: str):
        self.logger.info(f"_process_user_response 시작. 사용자 답변: '{user_answer[:50]}...'", extra={"work_id": work_id})
        node_state.answers.append(user_answer)

        intent_classification_result = "UN"
        if node_state.draft:
            intent_classification_result = await self._classify_user_response_intent(node_state.draft, user_answer,
                                                                                     work_id)
            self.logger.info(
                f"사용자 답변이 '{intent_classification_result}'으로 분류됨. ", extra={"work_id": work_id})
        else:
            intent_classification_result = "RE"
            self.logger.warning("피드백 처리 시 현재 초안(node_state.draft)이 없습니다. 사용자 답변을 수정 요청으로 간주합니다.",
                                extra={"work_id": work_id})

        # --- "오케이 사인" 처리 로직 수정 ---
        if intent_classification_result == "CO":
            # 사용자가 "오케이" 했으면, 현재 초안의 유효성(존재 여부)만 확인하고 바로 확정.
            # 길이 등 _is_topic_sufficient 검사는 생략하거나, 그 결과와 관계없이 확정.
            if node_state.draft:  # 현재 초안이 존재하기만 하면 확정
                self.logger.info(
                    f"사용자 답변이 '최종 확정(CO)'으로 분류됨. 사용자의 최종 승인으로 현재 초안을 확정합니다. User Answer: '{user_answer}', Draft: '{node_state.draft[:50]}...'",
                    extra={"work_id": work_id})
                node_state.is_final = True
                node_state.question = None
            else:
                # 사용자가 "오케이" 했으나, 확정할 초안 자체가 없는 극히 예외적인 경우.
                # 이 경우, 사용자에게 다시 명확한 주제를 요청하거나, user_answer를 새 초안으로 간주 후 품질 검사.
                # 여기서는 사용자에게 다시 명확한 주제를 요청하는 질문을 생성.
                self.logger.warning(
                    f"사용자 답변은 '최종 확정(CO)'이나, 현재 확정할 초안이 없습니다. User Answer: '{user_answer}'. 추가 정보 요청.",
                    extra={"work_id": work_id})
                node_state.is_final = False  # 확정할 초안이 없으므로 final 아님
                node_state.question = "현재 확정할 만한 주제 내용이 없습니다. 다시 한번 주제를 말씀해주시겠어요?"
                # 또는, user_answer 자체를 draft로 설정하고 is_sufficient 검사를 여기서 수행할 수도 있음:
                # node_state.draft = user_answer
                # if self._is_topic_sufficient(node_state.draft, work_id):
                #     node_state.is_final = True
                #     node_state.question = None
                # else:
                #     node_state.is_final = False
                #     node_state.question = self._formulate_followup_question(node_state.draft, node_state.chosen_intent_description, original_user_query, work_id)

        # "오케이 사인"으로 확정되지 않았다면 (RE 또는 UN, 또는 CO이었으나 초안이 없었던 경우)
        if not node_state.is_final:
            self.logger.info(f"사용자 답변 의도 분류 결과: '{intent_classification_result}'. LLM 통한 초안 수정 진행.",
                             extra={"work_id": work_id})

            current_draft_for_llm = node_state.draft if node_state.draft else user_answer  # 초안이 없으면 사용자 답변을 기반으로 새 초안 생성 시도
            if not node_state.draft and intent_classification_result == "RE":
                self.logger.info(f"현재 초안이 없고 사용자가 수정을 요청했으므로, 사용자 답변을 기반으로 새 초안 생성 시도: '{user_answer[:50]}...'",
                                 extra={"work_id": work_id})
            elif not node_state.draft and intent_classification_result == "UN":
                self.logger.info(f"현재 초안이 없고 사용자 의도가 불명확합니다. 사용자 답변을 참고하여 새 초안 생성 시도: '{user_answer[:50]}...'",
                                 extra={"work_id": work_id})

            intent_info_for_prompt = ""
            if node_state.chosen_intent_description:
                intent_info_for_prompt = f"\n[User's Primary Interest / Chosen Creative Direction]\n{node_state.chosen_intent_description}\n"

            prompt_sys_feedback = (
                "You are a helpful Korean language assistant. Your role is to meticulously integrate user feedback into an existing Korean topic statement to improve its clarity, accuracy, and completeness, "
                "or to generate a new topic statement if the existing draft is missing or insufficient, based on user's feedback and established intent. "  # 초안 부재 시 생성 역할 추가
                "The revised statement should aim for approximately 450-550 Korean characters, but prioritize reflecting the user's latest feedback accurately. "  # 길이 목표는 유지하되, 사용자 피드백 우선
                "Pay close attention to the user's specific requests for additions, deletions, or modifications. "
                "If the user's feedback introduces new aspects, ensure they are seamlessly incorporated while maintaining the core topic and previously established user intent (if provided). "
                "The goal is to produce a single, polished Korean topic statement. "
                "Respond ONLY with the revised Korean topic statement, in Korean."
            )
            prompt_user_feedback = (
                f"[Existing Draft (Korean)]\n{current_draft_for_llm}\n\n"  # current_draft_for_llm 사용
                f"{intent_info_for_prompt}"
                f"[User's Latest Feedback (Korean)]\n{user_answer}\n\n"
                "Based on all the above, please write an improved and complete Korean topic statement. If the existing draft was missing or the user feedback is substantial, generate a new statement based on the feedback and established intent. Prioritize reflecting the user's latest feedback. Aim for 450-550 characters if possible. Ensure the output is in Korean."
            )
            messages_feedback = [
                {"role": "system", "content": prompt_sys_feedback},
                {"role": "user", "content": prompt_user_feedback},
            ]

            resp = await self.llm.generate_text(
                messages=messages_feedback,
                request_id=f"topic-feedback-merge-{work_id}",
                max_tokens=1024,
                temperature=0.4,  # 사용자 피드백을 창의적으로 반영할 수 있도록 약간 높임 (0.3 -> 0.4)
                timeout=LLM_TIMEOUT_SEC,
            )
            new_draft = resp.get("generated_text", "").strip()

            if new_draft:
                self.logger.info(f"LLM 피드백 반영된 새 초안 생성됨 (길이: {len(new_draft)}자).", extra={"work_id": work_id})
                node_state.draft = new_draft
                # 사용자 "오케이 사인"이 아니었고, LLM이 새 초안을 만들었으므로, 이 새 초안에 대한 품질 검사 및 질문 생성 필요
                if self._is_topic_sufficient(new_draft, work_id):
                    # 품질 기준은 만족했으나, 사용자 최종 확인은 아직 안 받았으므로 is_final=False 유지.
                    # 대신, 이 초안으로 질문을 생성하여 사용자에게 보여줌.
                    node_state.is_final = False  # 아직 사용자 최종 확인 전
                    node_state.question = self._formulate_followup_question(new_draft,
                                                                            node_state.chosen_intent_description,
                                                                            original_user_query, work_id)
                    self.logger.info(f"새 초안이 품질 기준은 만족. 사용자 확인을 위해 질문 생성. 질문: '{node_state.question[:50]}...'",
                                     extra={"work_id": work_id})
                else:  # 새 초안이 품질 기준 미달
                    node_state.is_final = False
                    node_state.question = self._formulate_followup_question(new_draft,
                                                                            node_state.chosen_intent_description,
                                                                            original_user_query, work_id)
                    self.logger.info(f"새 초안이 품질 기준 미달. 추가 설명 필요 – 후속 질문 생성. 질문: '{node_state.question[:50]}...'",
                                     extra={"work_id": work_id})
            else:  # LLM 초안 생성 실패
                self.logger.error("LLM 피드백 반영 실패 – 새 초안 생성되지 않음. 기존 초안 및 질문 유지 또는 재생성.", extra={"work_id": work_id})
                draft_for_followup = node_state.draft if node_state.draft else "현재까지 구체화된 내용이 없습니다."
                node_state.question = self._formulate_followup_question(draft_for_followup,
                                                                        node_state.chosen_intent_description,
                                                                        original_user_query, work_id)

        self.logger.info("_process_user_response 종료.", extra={"work_id": work_id})

    async def _get_trusted_domains(self, work_id: str) -> List[str]:
        self.logger.debug("_get_trusted_domains 시작.", extra={"work_id": work_id})
        if not self.pg:
            self.logger.warning("PostgreSQLService가 제공되지 않았습니다. 신뢰도 도메인 없이 진행합니다.", extra={"work_id": work_id})
            return []
        try:
            self.logger.info("PostgreSQL에서 신뢰도 도메인 조회 시도.", extra={"work_id": work_id})
            rows = await self.pg.fetch_all("SELECT domain FROM ai_test_seed_domains")
            domains = [row["domain"] for row in rows if row and row.get("domain")]
            self.logger.info(f"신뢰도 도메인 {len(domains)}개 로드 성공.", extra={"work_id": work_id})
            self.logger.debug(f"로드된 도메인: {domains}", extra={"work_id": work_id})
            return domains
        except Exception as e:
            self.logger.error(f"PostgreSQL 신뢰도 도메인 조회 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})
            return []


    def _aggregate_search_context(self, srcs: List[Dict[str, str]], work_id_for_log: str, max_chars: int = 500) -> str:
        self.logger.debug(f"_aggregate_search_context 시작. 정보원 수: {len(srcs)}.", extra={"work_id": work_id_for_log})
        pieces: List[str] = []
        total_len = 0
        for src in srcs:
            seg = f"• {src.get('title', '')[:120]} – {src.get('snippet', '')[:200]}"
            if total_len + len(seg) > max_chars:
                self.logger.debug(f"최대 문자 수 ({max_chars}) 도달하여 컨텍스트 통합 중단.", extra={"work_id": work_id_for_log})
                break
            pieces.append(seg)
            total_len += len(seg)
        aggregated_context = "\n".join(pieces)
        self.logger.debug(f"통합된 컨텍스트 길이: {len(aggregated_context)}.", extra={"work_id": work_id_for_log})
        return aggregated_context


    def _is_topic_sufficient(self, draft: str, work_id_for_log: str) -> bool:
        length_ok = abs(len(draft) + 50 - DRAFT_TARGET_LEN) <= DRAFT_ALLOWED_DEVIATION
        kw_count = len({w for w in draft.split() if len(w) > 1})
        self.logger.debug(
            f"주제 품질 검사: 길이 만족={length_ok} (목표 {DRAFT_TARGET_LEN}±{DRAFT_ALLOWED_DEVIATION}, 실제 {len(draft)}), 키워드 수={kw_count} (최소 {MIN_KEYWORD_COUNT}). 초안: '{draft[:50]}...'",
            extra={"work_id": work_id_for_log})
        return length_ok and kw_count >= MIN_KEYWORD_COUNT


    async def _save_workflow_state_to_redis(self, work_id: str, state_dict: Dict[str, Any]):
        self.logger.debug(f"_save_workflow_state_to_redis 시작. work_id: {work_id}.", extra={"work_id": work_id})
        redis_key = REDIS_WORKFLOW_KEY_TMPL.format(work_id=work_id)
        try:
            loggable_state_dict_keys = list(state_dict.keys())
            self.logger.debug(f"Redis에 상태 저장 시도. Key: {redis_key}. 상태 Dict 키: {loggable_state_dict_keys}",
                              extra={"work_id": work_id})
            await self.redis.set(redis_key, state_dict, expire=60 * 60 * 6)
            self.logger.info(f"Redis 상태 저장 완료. Key: {redis_key}.", extra={"work_id": work_id})
        except Exception as e:
            self.logger.error(f"Redis 상태 저장 중 오류 발생: {e}", exc_info=True, extra={"work_id": work_id})


    def _attach_log_context(self, work_id: str):
        self.current_work_id = work_id  # 멤버 변수에 work_id 저장
        filter_name = f"work_id_filter_for_{work_id}"

        handler_has_filter = False
        for handler in self.logger.handlers:
            for f_obj in handler.filters:  # filter_obj로 변경 (f는 float 내장 함수와 혼동 가능)
                if hasattr(f_obj, 'name') and f_obj.name == filter_name:
                    handler_has_filter = True
                    break
            if handler_has_filter:
                break

        if not handler_has_filter:
            log_filter = logging.Filter(name=filter_name)

            def filter_record(record):
                record.work_id = work_id
                return True

            log_filter.filter = filter_record

            for handler in self.logger.handlers:
                handler.addFilter(log_filter)
            # self.logger.debug(f"로그 핸들러에 work_id 필터 ('{filter_name}') 추가됨.", extra={"work_id": work_id}) # 이 로그는 필터 추가 전이라 work_id가 안찍힐 수 있음
            # 따라서, 필터 추가 직후의 로그는 print나 기본 로거 사용, 또는 extra를 통해 work_id 강제 주입 필요
            # print(f"DEBUG: 로그 핸들러에 work_id 필터 ('{filter_name}') 추가됨. work_id: {work_id}")
            # 또는 self.logger.info(f"Work_id 필터 '{filter_name}' 추가됨.", extra={"work_id": work_id}) # 이것이 더 적절