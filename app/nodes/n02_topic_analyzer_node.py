# app/nodes/02_topic_analyzer_node.py (Improved Version)

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings            # 설정 객체 (기본값 참조용)
from app.services.llm_server_client_v2 import LLMService # LLM 서비스 클라이언트
from app.services.database_con_client_v2 import DatabaseClientV2 # Redis 클라이언트 (캐시용)
# from app.services.langsmith_service_v2 import LangSmithService # LangSmith 로깅 필요시
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# 로거 설정
logger = get_logger(__name__)

class TopicAnalyzerNode:
    """
    LLM을 사용하여 쿼리를 분석하고 키워드를 추출합니다.
    - LLMService (LLM 호출), DatabaseClientV2 (Redis 캐시) 사용.
    - 캐싱 및 재시도 메커니즘 포함.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """
    # 상태 입력/출력 정의 (참고용)
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
        logger.info("TopicAnalyzerNode initialized.")

    def _generate_cache_key(self, query: str, model_name: str) -> str:
        """캐시 키 생성"""
        key_string = f"topic_analysis::{query}::{model_name}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    async def _check_cache(self, key: str, trace_id: Optional[str]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Redis 캐시 확인"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        try:
            cached_value = await self.db_client.get(key)
            if cached_value:
                # DatabaseClientV2가 JSON 직렬화/역직렬화를 내부적으로 처리한다고 가정
                if isinstance(cached_value, dict):
                    logger.debug(f"{log_prefix} Cache hit for key: {key}")
                    return True, cached_value
                else:
                    # 만약 문자열로 저장되었다면 여기서 json.loads 시도
                    try:
                        parsed_value = json.loads(cached_value)
                        if isinstance(parsed_value, dict):
                           logger.debug(f"{log_prefix} Cache hit (after parsing str) for key: {key}")
                           return True, parsed_value
                        else:
                           logger.warning(f"{log_prefix} Parsed cache value is not dict for key {key}. Type: {type(parsed_value)}")
                           return False, None
                    except (json.JSONDecodeError, TypeError) as e:
                           logger.warning(f"{log_prefix} Failed to parse cached value for key {key}: {e}. Value: '{str(cached_value)[:100]}...'")
                           return False, None
            logger.debug(f"{log_prefix} Cache miss for key: {key}")
            return False, None
        except Exception as e:
             logger.error(f"{log_prefix} Error checking cache for key {key}: {e}", exc_info=True)
             return False, None # 캐시 확인 중 오류 발생 시 캐시 미스로 처리

    async def _update_cache(self, key: str, value: Dict[str, Any], cache_ttl: int, trace_id: Optional[str]) -> None:
        """Redis 캐시 업데이트 (TTL 포함)"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        try:
            # DatabaseClientV2가 내부적으로 JSON 직렬화를 처리한다고 가정
            success = await self.db_client.set(key, value, expire=cache_ttl)
            if success:
                logger.debug(f"{log_prefix} Redis cache updated for key: {key} with TTL {cache_ttl}s")
            else:
                logger.warning(f"{log_prefix} Failed to update Redis cache for key: {key}")
        except Exception as e:
            logger.error(f"{log_prefix} Error updating cache for key {key}: {e}", exc_info=True)

    # LLM 호출 재시도 설정
    # 참고: retry_if_exception_type(Exception)은 너무 광범위할 수 있습니다.
    # LLMService가 네트워크 오류, API 오류 등 특정 예외를 발생시킨다면
    # 해당 예외 타입들로 제한하는 것이 더 좋습니다. (예: (LLMTimeoutError, LLMAPIError))
    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float,
                                   trace_id: Optional[str] = None, **kwargs) -> str:
        """LLMService.generate_text를 재시도 로직과 함께 호출"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.info(f"{log_prefix} Attempting LLM call via LLMService...")

        result = await self.llm_client.generate_text(
            prompt=prompt,
            temperature=temperature,
            **kwargs # response_format 등 추가 파라미터 전달
        )

        if "error" in result:
            error_msg = result['error']
            logger.error(f"{log_prefix} LLMService returned error: {error_msg}")
            raise RuntimeError(f"LLMService error: {error_msg}") # 재시도 유발
        elif "generated_text" not in result or not isinstance(result["generated_text"], str):
            logger.error(f"{log_prefix} LLMService invalid response: {result}")
            raise ValueError("LLMService returned no valid generated_text") # 재시도 유발
        else:
            logger.info(f"{log_prefix} LLM call via LLMService successful.")
            return result["generated_text"] # 성공 시 텍스트 반환

    def _create_analysis_prompt_en(self, query: str) -> str:
        """주제 분석 프롬프트 생성"""
        # 프롬프트 내용은 이전과 동일
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

    def _parse_llm_response(self, response_str: str, keyword_confidence_threshold: float, trace_id: Optional[str]) -> Tuple[Optional[Dict], Optional[List]]:
        """LLM JSON 응답 파싱 및 검증"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        try:
            logger.debug(f"{log_prefix} Raw LLM response for parsing: {response_str[:500]}...")
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_str, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            else:
                json_str = response_str.strip()
                if not json_str.startswith("{") and json_str.endswith("}"): json_str = "{" + json_str
                elif json_str.startswith("{") and not json_str.endswith("}"): json_str = json_str + "}"

            logger.debug(f"{log_prefix} Cleaned JSON string for parsing: {json_str[:500]}...")
            analysis_result = json.loads(json_str)

            if not isinstance(analysis_result, dict): raise ValueError("Root is not a dictionary.")
            if not isinstance(analysis_result.get("main_topic"), str): raise ValueError("'main_topic' is missing or not a string.")
            if not isinstance(analysis_result.get("entities"), list): raise ValueError("'entities' is missing or not a list.")
            if not isinstance(analysis_result.get("keywords_analyzed"), list): raise ValueError("'keywords_analyzed' is missing or not a list.")

            topic_analysis = analysis_result
            keywords_analyzed = analysis_result.get("keywords_analyzed", [])
            search_keywords = []
            seen_keywords = set()

            for item in keywords_analyzed:
                if isinstance(item, dict):
                    keyword = item.get("keyword")
                    confidence = item.get("confidence", 0.0)
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

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"{log_prefix} Failed parsing/validating LLM response: {e}. Response fragment: '{response_str[:200]}...'")
            return None, None
        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error parsing LLM response: {e}. Response fragment: '{response_str[:200]}...'")
            return None, None

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """주제 분석 노드 실행 로직 (Redis 캐시 및 LLMService 사용)"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id # 초기화 노드에서 반드시 설정됨
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing TopicAnalyzerNode...")

        # 상태에서 필요한 값 가져오기
        initial_query = state.initial_query or ""
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        # --- 설정값 로드 (state.config 우선, 없으면 settings 기본값 사용) ---
        # 참고: 이 노드가 실행되기 전에 state.config에 필요한 값이 설정되어 있어야 함
        #       또는 settings에 해당 기본값이 정의되어 있어야 함.
        llm_model_name = config.get("llm_model", settings.DEFAULT_LLM_MODEL) # 캐시키 생성용
        llm_temperature = float(config.get("llm_temperature_analysis", settings.DEFAULT_LLM_TEMP_ANALYSIS))
        cache_ttl = int(config.get("topic_analyzer_cache_ttl", settings.DEFAULT_CACHE_TTL_TOPIC))
        keyword_confidence_threshold = float(config.get("keyword_confidence_threshold", settings.DEFAULT_KEYWORD_CONF_THRESHOLD))
        # ---------------------------------------------------------------

        if not initial_query:
            logger.error(f"{log_prefix} Initial query is missing.")
            processing_stats['topic_analyzer_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"error_message": "Initial query is missing for topic analysis", "processing_stats": processing_stats}

        logger.info(f"{log_prefix} Starting topic analysis for query: '{initial_query}' using model '{llm_model_name}'")

        topic_analysis: Optional[Dict] = None
        search_keywords: Optional[List] = None
        error_message: Optional[str] = None

        try:
            # --- 캐싱 확인 (Redis) ---
            cache_key = self._generate_cache_key(initial_query, llm_model_name)
            cache_hit, cached_data = await self._check_cache(cache_key, trace_id)

            if cache_hit and cached_data:
                logger.info(f"{log_prefix} Cache hit found. Reusing previous analysis.")
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

                # LLM 호출 (JSON 응답 형식 요청)
                llm_kwargs = {"response_format": {"type": "json_object"}}

                try:
                    llm_response_str = await self._call_llm_with_retry(
                        prompt=prompt,
                        temperature=llm_temperature,
                        trace_id=trace_id,
                        **llm_kwargs
                    )
                    # 응답 파싱
                    topic_analysis, search_keywords = self._parse_llm_response(llm_response_str, keyword_confidence_threshold, trace_id)

                    if topic_analysis is None or search_keywords is None:
                        error_message = "Failed to parse or validate LLM response for topic analysis"
                        logger.error(f"{log_prefix} {error_message}")
                    else:
                        # 성공 시 캐시 업데이트
                        await self._update_cache(cache_key, {
                            "topic_analysis": topic_analysis,
                            "search_keywords": search_keywords
                        }, cache_ttl, trace_id)
                        logger.info(f"{log_prefix} Topic analysis successful. Extracted {len(search_keywords)} keywords.")

                except RetryError as e: # 모든 재시도 실패
                    error_message = f"LLM call failed after multiple retries: {e}"
                    logger.error(f"{log_prefix} {error_message}")
                except Exception as e: # LLM 호출 또는 파싱 중 예상치 못한 오류
                    error_message = f"Error during LLM call or parsing: {str(e)}"
                    logger.exception(f"{log_prefix} {error_message}")

        except Exception as e:
            # 캐시 확인/업데이트 등 로직 자체의 예외 처리
            error_message = f"Topic analysis failed due to an unexpected error: {str(e)}"
            logger.exception(f"{log_prefix} {error_message} for query '{initial_query}'")

        # --- 처리 시간 및 결과 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['topic_analyzer_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} TopicAnalyzerNode finished in {processing_stats['topic_analyzer_node_time']:.2f} seconds. Error: {error_message is not None}")

        # TODO: LangSmith 로깅 (필요시)

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "topic_analysis": topic_analysis, # 성공 시 값, 실패 시 이전 값 또는 None
            "search_keywords": search_keywords, # 성공 시 값, 실패 시 이전 값 또는 None
            "processing_stats": processing_stats, # 업데이트된 통계
            "error_message": error_message # 오류 메시지 (성공 시 None)
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}