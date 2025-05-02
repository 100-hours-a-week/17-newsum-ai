# app/nodes/02_topic_analyzer_node.py

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 프로젝트 구성 요소 임포트 ---
from app.config.settings import settings            # 설정 객체 (재시도 횟수 등 참조)
from app.services.llm_server_client_v2 import LLMService # LLM 서비스 클라이언트
from app.services.database_con_client_v2 import DatabaseClientV2 # Redis 클라이언트 (캐시용)
# from app.services.langsmith_service_v2 import LangSmithService # LangSmith 로깅 필요시
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# --- 로거 설정 ---
logger = get_logger("TopicAnalyzerNode")

class TopicAnalyzerNode:
    """
    (Refactored) LLM을 사용하여 쿼리를 분석하고 키워드를 추출합니다.
    - LLMService (LLM 호출), DatabaseClientV2 (Redis 캐시) 사용.
    - 캐싱 및 재시도 메커니즘 포함.
    """
    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["initial_query", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["topic_analysis", "search_keywords", "processing_stats", "error_message"]

    # 서비스 클라이언트를 외부에서 주입
    def __init__(
        self,
        llm_client: LLMService,
        db_client: DatabaseClientV2,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        self.llm_client = llm_client
        self.db_client = db_client
        # self.langsmith_service = langsmith_service

    def _generate_cache_key(self, query: str, model_name: str) -> str:
        """캐시 키 생성"""
        key_string = f"topic_analysis::{query}::{model_name}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    async def _check_cache(self, key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Redis 캐시 확인"""
        cached_value = await self.db_client.get(key)
        if cached_value:
            # Redis에서 가져온 값은 JSON 문자열일 수 있으므로 파싱 시도 (db_client 설정에 따라 다름)
            # DatabaseClientV2가 decode_responses=True, JSON 직렬화/역직렬화를 내부적으로 처리한다면
            # 아래 isinstance 체크만으로 충분할 수 있음.
            if isinstance(cached_value, dict):
                logger.debug(f"Cache hit for key: {key}")
                return True, cached_value
            else:
                # 만약 문자열로 저장되었다면 여기서 json.loads 시도 필요
                try:
                    parsed_value = json.loads(cached_value)
                    if isinstance(parsed_value, dict):
                       logger.debug(f"Cache hit (after parsing str) for key: {key}")
                       return True, parsed_value
                    else:
                       logger.warning(f"Parsed cache value is not dict for key {key}. Type: {type(parsed_value)}")
                       return False, None
                except (json.JSONDecodeError, TypeError) as e:
                       logger.warning(f"Failed to parse cached value for key {key}: {e}. Value: '{str(cached_value)[:100]}...'")
                       return False, None
        logger.debug(f"Cache miss for key: {key}")
        return False, None

    async def _update_cache(self, key: str, value: Dict[str, Any], cache_ttl: int) -> None:
        """Redis 캐시 업데이트 (TTL 포함)"""
        # DatabaseClientV2가 내부적으로 JSON 직렬화를 처리한다고 가정
        success = await self.db_client.set(key, value, expire=cache_ttl)
        if success:
            logger.debug(f"Redis cache updated for key: {key} with TTL {cache_ttl}s")
        else:
            logger.warning(f"Failed to update Redis cache for key: {key}")

    # settings의 LLM_API_RETRIES 값으로 재시도 횟수 설정
    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # LLMService가 반환하는 특정 오류나 네트워크 오류 등 구체적 예외로 재시도 조건 강화 가능
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float,
                                   trace_id: Optional[str] = None, **kwargs) -> str:
        """LLMService.generate_text를 재시도 로직과 함께 호출"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.info(f"{log_prefix} Attempting LLM call via LLMService...")

        # LLMService 호출
        result = await self.llm_client.generate_text(
            prompt=prompt,
            temperature=temperature,
            **kwargs # response_format 등 추가 파라미터 전달
        )

        # 결과 처리
        if "error" in result:
            error_msg = result['error']
            logger.error(f"{log_prefix} LLMService returned error: {error_msg}")
            # 재시도 유발을 위해 예외 발생
            raise RuntimeError(f"LLMService error: {error_msg}")
        elif "generated_text" not in result or not isinstance(result["generated_text"], str):
            logger.error(f"{log_prefix} LLMService invalid response: {result}")
            raise ValueError("LLMService returned no valid generated_text")
        else:
            logger.info(f"{log_prefix} LLM call via LLMService successful.")
            return result["generated_text"] # 성공 시 텍스트 반환

    def _create_analysis_prompt_en(self, query: str) -> str:
        """주제 분석 프롬프트 생성 (이전과 동일)"""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant specialized in analyzing text queries to extract key information for content generation pipelines. Your task is to identify the main topic, key entities, and generate optimized search keywords. Respond ONLY with a valid JSON object. Ensure keys and string values are enclosed in double quotes.<|eot_id|><|start_header_id|>user<|end_header_id|>
Analyze the following query to identify the main topic, key entities, and effective search keywords.

[Query to Analyze]
{query}

[Analysis Instructions]
1.  **main_topic**: Identify the single, most central theme or subject of the query. Be concise.
2.  **entities**: Extract key named entities (e.g., people, organizations, locations, products, technologies, specific concepts). For each entity, specify its type and a confidence score (0.0 to 1.0) indicating its relevance and certainty. Include at least 2-3 important entities if present.
3.  **keywords_analyzed**: Generate a list of 3-5 distinct search keywords. These keywords should be optimized for finding relevant news articles and diverse opinions online (e.g., Twitter, Reddit, Blogs). They should cover different aspects or angles of the query if possible. Provide a confidence score (0.0 to 1.0) for each keyword's likely effectiveness in search.

[Required Output Format]
Respond ONLY with a single, valid JSON object enclosed in ```json ... ```, following this exact structure:
```json
{{
  "main_topic": "string",
  "entities": [
    {{"name": "string", "type": "string", "confidence": float}}
  ],
  "keywords_analyzed": [
    {{"keyword": "string", "confidence": float}}
  ]
}}
```<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
        return prompt

    def _parse_llm_response(self, response_str: str, trace_id: Optional[str]) -> Tuple[Optional[Dict], Optional[List]]:
        """LLM JSON 응답 파싱 및 검증 (이전과 동일, 로깅 강화)"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        try:
            logger.debug(f"{log_prefix} Raw LLM response for parsing: {response_str[:500]}...") # 로그 길이를 적절히 조절
            # ```json ``` 블록 제거 개선
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_str, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1)
            else:
                json_str = response_str.strip()
                # 간단한 괄호 보정 (만능은 아님)
                if not json_str.startswith("{") and json_str.endswith("}"): json_str = "{" + json_str
                elif json_str.startswith("{") and not json_str.endswith("}"): json_str = json_str + "}"

            logger.debug(f"{log_prefix} Cleaned JSON string for parsing: {json_str[:500]}...")
            analysis_result = json.loads(json_str)

            # 구조 검증 강화
            if not isinstance(analysis_result, dict): raise ValueError("Root is not a dictionary.")
            if not isinstance(analysis_result.get("main_topic"), str): raise ValueError("'main_topic' is missing or not a string.")
            if not isinstance(analysis_result.get("entities"), list): raise ValueError("'entities' is missing or not a list.")
            if not isinstance(analysis_result.get("keywords_analyzed"), list): raise ValueError("'keywords_analyzed' is missing or not a list.")

            topic_analysis = analysis_result

            # 키워드 추출 및 필터링
            keywords_analyzed = analysis_result.get("keywords_analyzed", [])
            search_keywords = []
            seen_keywords = set()
            # 키워드 신뢰도 임계값 (config에서 가져오도록 개선 가능)
            keyword_confidence_threshold = 0.5

            for item in keywords_analyzed:
                if isinstance(item, dict):
                    keyword = item.get("keyword")
                    confidence = item.get("confidence", 0.0)
                    # 키워드가 문자열이고, 신뢰도가 임계값 이상이며, 공백이 아닌 경우
                    if isinstance(keyword, str) and keyword.strip() and \
                       isinstance(confidence, (int, float)) and confidence >= keyword_confidence_threshold:
                        kw_clean = keyword.strip()
                        kw_lower = kw_clean.lower()
                        if kw_lower not in seen_keywords:
                            search_keywords.append(kw_clean)
                            seen_keywords.add(kw_lower)
                else:
                    logger.warning(f"{log_prefix} Invalid item in 'keywords_analyzed': {item}")

            if not search_keywords: logger.warning(f"{log_prefix} No valid search keywords extracted.")

            return topic_analysis, search_keywords

        except (json.JSONDecodeError, ValueError, TypeError) as e: # 구체적인 예외 처리
            logger.error(f"{log_prefix} Failed parsing/validating LLM response: {e}. Response fragment: '{response_str[:200]}...'")
            return None, None
        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error parsing LLM response: {e}. Response fragment: '{response_str[:200]}...'")
            return None, None


    async def run(self, state: ComicState) -> Dict[str, Any]:
        """주제 분석 노드 실행 로직 (Redis 캐시 및 LLMService 사용)"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing TopicAnalyzerNode...")

        # 상태에서 필요한 값 가져오기
        initial_query = state.initial_query or ""
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        # 설정값 로드 (InitializeNode가 settings에서 가져와 config에 넣은 값)
        llm_model_name = config.get("llm_model", "unknown_model") # 캐시키 생성용
        llm_temperature = float(config.get("llm_temperature_analysis", 0.2))
        cache_ttl = int(config.get("topic_analyzer_cache_ttl", 1800))

        # 입력 검증
        if not initial_query:
            logger.error(f"{log_prefix} Initial query is missing.")
            return {"error_message": "Initial query is missing for topic analysis"}

        logger.info(f"{log_prefix} Starting topic analysis for query: '{initial_query}'")

        topic_analysis: Optional[Dict] = None
        search_keywords: Optional[List] = None
        error_message: Optional[str] = None

        try:
            # --- 캐싱 확인 (Redis) ---
            cache_key = self._generate_cache_key(initial_query, llm_model_name)
            cache_hit, cached_data = await self._check_cache(cache_key)

            if cache_hit and cached_data:
                logger.info(f"{log_prefix} Cache hit found. Reusing previous analysis.")
                # 캐시 데이터 유효성 검사 (선택적이지만 권장)
                if isinstance(cached_data.get("topic_analysis"), dict) and \
                   isinstance(cached_data.get("search_keywords"), list):
                    topic_analysis = cached_data["topic_analysis"]
                    search_keywords = cached_data["search_keywords"]
                else:
                    logger.warning(f"{log_prefix} Cached data is invalid/incomplete. Forcing LLM call.")
                    cache_hit = False # 유효하지 않으면 다시 호출

            # --- 캐시 미스 또는 무효 캐시 시 LLM 호출 ---
            if not cache_hit:
                logger.info(f"{log_prefix} Cache miss or invalid cache. Calling LLM.")
                prompt = self._create_analysis_prompt_en(initial_query)

                # LLM 호출 (response_format 등 필요한 kwargs 추가)
                llm_kwargs = {}
                # 예시: LLM 서비스가 JSON 모드를 지원하는 경우
                # llm_kwargs['response_format'] = {"type": "json_object"}

                llm_response_str = await self._call_llm_with_retry(
                    prompt=prompt,
                    temperature=llm_temperature,
                    trace_id=state.trace_id,
                    **llm_kwargs
                )

                # 응답 파싱
                topic_analysis, search_keywords = self._parse_llm_response(llm_response_str, state.trace_id)

                if topic_analysis is None or search_keywords is None:
                    error_message = "Failed to parse or validate LLM response for topic analysis"
                    logger.error(f"{log_prefix} {error_message}")
                    # 오류가 발생했지만, 후속 노드에서 처리할 수 있도록 일단 진행할 수도 있음
                    # 여기서는 오류 메시지만 설정하고 반환 (그래프 흐름 제어는 별도)
                else:
                    # 성공 시 캐시 업데이트
                    await self._update_cache(cache_key, {
                        "topic_analysis": topic_analysis,
                        "search_keywords": search_keywords
                    }, cache_ttl)
                    logger.info(f"{log_prefix} Topic analysis successful. Extracted {len(search_keywords)} keywords.")

        except Exception as e:
            # 재시도 실패 또는 기타 예외 처리
            error_message = f"Topic analysis failed: {str(e)}"
            logger.exception(f"{log_prefix} {error_message} for query '{initial_query}'")
            # 오류 발생 시 결과는 None 유지

        # --- 처리 시간 및 결과 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        # processing_stats는 딕셔너리이므로 직접 업데이트
        processing_stats['topic_analyzer_node_time'] = node_processing_time
        logger.info(f"{log_prefix} TopicAnalyzerNode finished in {node_processing_time:.2f} seconds. Error: {error_message is not None}")

        # TODO: LangSmith 로깅 (필요시)
        # if self.langsmith_service: ...

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "topic_analysis": topic_analysis, # 성공 시 값, 실패 시 이전 값 또는 None
            "search_keywords": search_keywords, # 성공 시 값, 실패 시 이전 값 또는 None
            "processing_stats": processing_stats, # 업데이트된 통계
            "error_message": error_message # 오류 메시지 (성공 시 None)
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}