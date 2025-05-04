# app/nodes/02_topic_analyzer_node.py (Refactored)

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.services.llm_server_client_v2 import LLMService
from app.services.database_con_client_v2 import DatabaseClientV2
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

class TopicAnalyzerNode:
    """
    LLM을 사용하여 쿼리를 분석하고 키워드를 추출합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["initial_query", "trace_id", "comic_id", "config"] # MODIFIED: Added comic_id
    outputs: List[str] = ["topic_analysis", "search_keywords", "node2_processing_stats", "error_message"] # MODIFIED: Renamed node1_ -> node2_

    def __init__(self, llm_client: LLMService, db_client: DatabaseClientV2):
        self.llm_client = llm_client
        self.db_client = db_client
        logger.info("TopicAnalyzerNode initialized.")

    def _generate_cache_key(self, query: str, model_name: str) -> str:
        key_string = f"topic_analysis::{query}::{model_name}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    async def _check_cache(self, key: str, trace_id: Optional[str], comic_id: Optional[str]) -> Tuple[bool, Optional[Dict[str, Any]]]: # MODIFIED: Added comic_id
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        try:
            cached_value = await self.db_client.get(key)
            if cached_value:
                if isinstance(cached_value, dict):
                    logger.debug(f"Cache hit for key: {key}", extra=extra_log_data) # MODIFIED
                    return True, cached_value
                else:
                    try:
                        parsed_value = json.loads(cached_value)
                        if isinstance(parsed_value, dict):
                           logger.debug(f"Cache hit (after parsing str) for key: {key}", extra=extra_log_data) # MODIFIED
                           return True, parsed_value
                        else:
                           logger.warning(f"Parsed cache value is not dict for key {key}. Type: {type(parsed_value)}", extra=extra_log_data) # MODIFIED
                           return False, None
                    except (json.JSONDecodeError, TypeError) as e:
                           logger.warning(f"Failed to parse cached value for key {key}: {e}. Value: '{str(cached_value)[:100]}...'", extra=extra_log_data) # MODIFIED
                           return False, None
            logger.debug(f"Cache miss for key: {key}", extra=extra_log_data) # MODIFIED
            return False, None
        except Exception as e:
             logger.error(f"Error checking cache for key {key}: {e}", exc_info=True, extra=extra_log_data) # MODIFIED
             return False, None

    async def _update_cache(self, key: str, value: Dict[str, Any], cache_ttl: int, trace_id: Optional[str], comic_id: Optional[str]) -> None: # MODIFIED: Added comic_id
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        try:
            success = await self.db_client.set(key, value, expire=cache_ttl)
            if success:
                logger.debug(f"Redis cache updated for key: {key} with TTL {cache_ttl}s", extra=extra_log_data) # MODIFIED
            else:
                logger.warning(f"Failed to update Redis cache for key: {key}", extra=extra_log_data) # MODIFIED
        except Exception as e:
            logger.error(f"Error updating cache for key {key}: {e}", exc_info=True, extra=extra_log_data) # MODIFIED

    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float,
                                   trace_id: Optional[str] = None, comic_id: Optional[str] = None, **kwargs) -> str: # MODIFIED: Added comic_id
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        logger.info(f"Attempting LLM call via LLMService...", extra=extra_log_data) # MODIFIED

        # Ensure llm_client.generate_text can accept trace_id and comic_id if needed for its internal logging/tracing
        result = await self.llm_client.generate_text(
            prompt=prompt,
            temperature=temperature,
            # Pass trace_id and comic_id if the client supports them, e.g.:
            # trace_id=trace_id,
            # comic_id=comic_id,
            **kwargs
        )

        if "error" in result:
            error_msg = result['error']
            logger.error(f"LLMService returned error: {error_msg}", extra=extra_log_data) # MODIFIED
            raise RuntimeError(f"LLMService error: {error_msg}")
        elif "generated_text" not in result or not isinstance(result["generated_text"], str):
            logger.error(f"LLMService invalid response: {result}", extra=extra_log_data) # MODIFIED
            raise ValueError("LLMService returned no valid generated_text")
        else:
            logger.info(f"LLM call via LLMService successful.", extra=extra_log_data) # MODIFIED
            return result["generated_text"]

    def _create_analysis_prompt_en(self, query: str) -> str:
        """주제 분석 프롬프트 생성 (English version)"""
#         # [... existing prompt string ...]
#         # No changes needed in the prompt itself based on the request
#         prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant specialized in analyzing text queries to extract key information for content generation pipelines. Your task is to identify the main topic, key entities, and generate optimized search keywords. Respond ONLY with a valid JSON object. Ensure keys and string values are enclosed in double quotes.<|eot_id|><|start_header_id|>user<|end_header_id|>
# Analyze the following query to identify the main topic, key entities, and effective search keywords.
#
# [Query to Analyze]
# {query}
#
# [Analysis Instructions]
# 1.  **main_topic**: Identify the single, most central theme or subject of the query. Be concise.
# 2.  **entities**: Extract key named entities (e.g., people, organizations, locations, products, technologies, specific concepts). For each entity, specify its type and a confidence score (0.0 to 1.0) indicating its relevance and certainty. Include at least 2-3 important entities if present.
# 3.  **keywords_analyzed**: Generate a list of 3-5 distinct search keywords. These keywords should be optimized for finding relevant news articles and diverse opinions online (e.g., Twitter, Reddit, Blogs). They should cover different aspects or angles of the query if possible. Provide a confidence score (0.0 to 1.0) for each keyword's likely effectiveness in search.
#
# [Required Output Format]
# Respond ONLY with a single, valid JSON object enclosed in ```json ... ```, following this exact structure:
# ```json
# {{
#   "main_topic": "string",
#   "entities": [
#     {{"name": "string", "type": "string", "confidence": float}}
#   ],
#   "keywords_analyzed": [
#     {{"keyword": "string", "confidence": float}}
#   ]
# }}
# ```<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# ```json
# """
#         return prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an assistant tasked with concise topic analysis and keyword extraction.<|eot_id|>
        
            <|start_header_id|>user<|end_header_id|>
            Analyze the query: '{query}'
        
            - Identify the main topic (max 5 words).
            - Extract 3 important entities.
            - Generate 5 concise keywords highly related to recent news and discussions.
        
            Respond exactly in this format:
            Main topic: <topic>
            Entities: entity1, entity2, entity3
            Keywords: keyword1, keyword2, keyword3, keyword4, keyword5
            <|eot_id|>
        
            <|start_header_id|>assistant<|end_header_id|>
            Main topic:"""
        return prompt


    def _parse_llm_response(self, response_str: str, keyword_confidence_threshold: float, trace_id: Optional[str], comic_id: Optional[str]) -> Tuple[Optional[Dict], Optional[List]]: # MODIFIED: Added comic_id
        """LLM JSON 응답 파싱 및 검증"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        try:
            logger.debug(f"Raw LLM response for parsing: {response_str[:500]}...", extra=extra_log_data) # MODIFIED
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_str, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            else:
                # Attempt to handle cases where the backticks might be missing
                json_str = response_str.strip()
                if json_str.startswith("{") and json_str.endswith("}"):
                    pass # Looks like a valid JSON object already
                elif json_str.startswith("{") and not json_str.endswith("}"):
                     # Try to find the last closing brace
                     last_brace_index = json_str.rfind('}')
                     if last_brace_index != -1:
                          json_str = json_str[:last_brace_index+1]
                     else:
                          logger.warning("Could not reliably find end of JSON object.", extra=extra_log_data) # MODIFIED
                          # Fallback or raise? For now, proceed, json.loads will likely fail
                else:
                     # If it doesn't start with { or end with }, it's unlikely to be the JSON we want directly
                     logger.warning("LLM response does not appear to be the expected JSON object.", extra=extra_log_data) # MODIFIED
                     # Try finding JSON within the string anyway, might be embedded differently
                     json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                     if json_match:
                          json_str = json_match.group(0)
                     else:
                          raise ValueError("Could not extract JSON object from LLM response.")


            logger.debug(f"Cleaned JSON string for parsing: {json_str[:500]}...", extra=extra_log_data) # MODIFIED
            analysis_result = json.loads(json_str)

            # --- Validation ---
            if not isinstance(analysis_result, dict): raise ValueError("Root is not a dictionary.")
            if not isinstance(analysis_result.get("main_topic"), str) or not analysis_result["main_topic"].strip():
                 logger.warning("Validation failed: 'main_topic' is missing, not a string, or empty.", extra=extra_log_data) # MODIFIED
                 raise ValueError("'main_topic' is missing or not a non-empty string.")
            if not isinstance(analysis_result.get("entities"), list):
                 logger.warning("Validation failed: 'entities' is missing or not a list.", extra=extra_log_data) # MODIFIED
                 raise ValueError("'entities' is missing or not a list.")
            if not isinstance(analysis_result.get("keywords_analyzed"), list):
                 logger.warning("Validation failed: 'keywords_analyzed' is missing or not a list.", extra=extra_log_data) # MODIFIED
                 raise ValueError("'keywords_analyzed' is missing or not a list.")
            # Deeper validation (optional but recommended)
            # for entity in analysis_result.get("entities", []):
            #     if not isinstance(entity.get("name"), str) or not isinstance(entity.get("type"), str) or not isinstance(entity.get("confidence"), (int, float)):
            #         logger.warning(f"Invalid entity format found: {entity}", extra=extra_log_data)
            #         # Decide whether to raise error or just skip invalid item
            # for kw_item in analysis_result.get("keywords_analyzed", []):
            #      if not isinstance(kw_item.get("keyword"), str) or not isinstance(kw_item.get("confidence"), (int, float)):
            #           logger.warning(f"Invalid keyword format found: {kw_item}", extra=extra_log_data)
            #           # Decide whether to raise error or just skip invalid item

            topic_analysis = analysis_result # Use the validated result
            keywords_analyzed = analysis_result.get("keywords_analyzed", [])
            search_keywords = []
            seen_keywords = set()

            for item in keywords_analyzed:
                if isinstance(item, dict):
                    keyword = item.get("keyword")
                    confidence = item.get("confidence", 0.0)
                    # Ensure keyword is a non-empty string and confidence meets threshold
                    if isinstance(keyword, str) and keyword.strip() and \
                       isinstance(confidence, (int, float)) and confidence >= keyword_confidence_threshold:
                        kw_clean = keyword.strip()
                        kw_lower = kw_clean.lower()
                        if kw_lower not in seen_keywords:
                            search_keywords.append(kw_clean)
                            seen_keywords.add(kw_lower)
                        else:
                             logger.debug(f"Skipping duplicate keyword: {kw_clean}", extra=extra_log_data) # MODIFIED
                    elif isinstance(keyword, str) and keyword.strip():
                         logger.debug(f"Keyword '{keyword}' below confidence threshold ({confidence} < {keyword_confidence_threshold}). Skipping.", extra=extra_log_data) # MODIFIED
                    else:
                         logger.warning(f"Invalid keyword item format or empty keyword in 'keywords_analyzed': {item}", extra=extra_log_data) # MODIFIED
                else:
                    logger.warning(f"Non-dict item found in 'keywords_analyzed': {item}", extra=extra_log_data) # MODIFIED

            if not search_keywords:
                logger.warning(f"No valid search keywords extracted after parsing and filtering.", extra=extra_log_data) # MODIFIED
                # Depending on requirements, you might want to return None here or an empty list.
                # Returning empty list allows workflow to proceed, but might yield no results later.

            return topic_analysis, search_keywords

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed parsing/validating LLM response: {e}. Response fragment: '{response_str[:200]}...'", extra=extra_log_data) # MODIFIED
            return None, None
        except Exception as e:
            logger.exception(f"Unexpected error parsing LLM response: {e}. Response fragment: '{response_str[:200]}...'", extra=extra_log_data) # MODIFIED
            return None, None

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """주제 분석 노드 실행 로직"""
        start_time = datetime.now(timezone.utc)
        # --- MODIFIED: Get trace_id and comic_id safely ---
        comic_id = getattr(state, 'comic_id', 'unknown_comic')
        trace_id = getattr(state, 'trace_id', comic_id) # Fallback to comic_id if trace_id not set
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        # -------------------------------------------------

        node_class_name = self.__class__.__name__
        # --- ADDED: Start Logging ---
        logger.info(f"--- Executing {node_class_name} ---", extra=extra_log_data)
        logger.debug(f"Entering state:\n{summarize_for_logging(state, is_state_object=True)}", extra=extra_log_data)
        # --------------------------

        initial_query = getattr(state, 'initial_query', None) # Safe access
        config = getattr(state, 'config', {}) or {} # Safe access, ensure dict

        # --- ADDED: Input Validation ---
        if not initial_query or not initial_query.strip():
            error_message = "Initial query is missing or empty for topic analysis."
            logger.error(error_message, extra=extra_log_data)
            end_time = datetime.now(timezone.utc)
            node2_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "topic_analysis": None,
                "search_keywords": [], # Return empty list for consistency
                "node2_processing_stats": node2_processing_stats,
                "error_message": error_message
            }
            # --- ADDED: End Logging (Error Case) ---
            logger.debug(f"Returning updates on error:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Error: Missing Query) --- (Elapsed: {node2_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        # 설정값 로드
        llm_model_name = config.get("llm_model", settings.DEFAULT_LLM_MODEL)
        llm_temperature = float(config.get("llm_temperature_analysis", settings.LLM_TEMPERATURE_ANALYSIS))
        cache_ttl = int(config.get("topic_analyzer_cache_ttl", settings.DEFAULT_CACHE_TTL_TOPIC))
        keyword_confidence_threshold = float(config.get("keyword_confidence_threshold", settings.DEFAULT_KEYWORD_CONF_THRESHOLD))

        logger.info(f"Starting topic analysis for query: '{initial_query}' using model '{llm_model_name}'", extra=extra_log_data)

        topic_analysis: Optional[Dict] = None
        search_keywords: Optional[List] = None # Keep Optional for clarity, but return [] on error
        error_message: Optional[str] = None

        try:
            cache_key = self._generate_cache_key(initial_query, llm_model_name)
            # MODIFIED: Pass comic_id to cache functions
            cache_hit, cached_data = await self._check_cache(cache_key, trace_id, comic_id)

            if cache_hit and cached_data:
                logger.info(f"Cache hit found. Reusing previous analysis.", extra=extra_log_data)
                # Validate cached data structure before using
                if isinstance(cached_data.get("topic_analysis"), dict) and \
                   isinstance(cached_data.get("search_keywords"), list):
                    topic_analysis = cached_data["topic_analysis"]
                    search_keywords = cached_data["search_keywords"]
                    # Optionally re-verify keyword confidence if threshold changed? Unlikely needed.
                else:
                    logger.warning(f"Cached data is invalid/incomplete. Forcing LLM call.", extra=extra_log_data)
                    cache_hit = False

            if not cache_hit:
                logger.info(f"Cache miss or invalid cache. Calling LLM.", extra=extra_log_data)
                prompt = self._create_analysis_prompt_en(initial_query)
                llm_kwargs = {"response_format": {"type": "json_object"}}

                try:
                    llm_response_str = await self._call_llm_with_retry(
                        prompt=prompt,
                        temperature=llm_temperature,
                        trace_id=trace_id,
                        comic_id=comic_id, # Pass IDs
                        **llm_kwargs
                    )
                    # MODIFIED: Pass comic_id to parser
                    topic_analysis, search_keywords = self._parse_llm_response(
                        llm_response_str, keyword_confidence_threshold, trace_id, comic_id
                    )

                    if topic_analysis is None or search_keywords is None:
                        # Parsing/Validation failed (already logged in _parse_llm_response)
                        error_message = "Failed to parse or validate LLM response for topic analysis."
                        # Reset to default values to avoid returning partial data from failed parse
                        topic_analysis = None
                        search_keywords = None
                    else:
                        # Success - Update cache
                        await self._update_cache(cache_key, {
                            "topic_analysis": topic_analysis,
                            "search_keywords": search_keywords
                        }, cache_ttl, trace_id, comic_id) # Pass IDs
                        logger.info(f"Topic analysis successful. Extracted {len(search_keywords)} keywords.", extra=extra_log_data)

                except RetryError as e:
                    error_message = f"LLM call failed after multiple retries: {e}"
                    logger.error(error_message, extra=extra_log_data) # Error already includes details
                    topic_analysis = None # Ensure reset on failure
                    search_keywords = None
                except Exception as e:
                    error_message = f"Error during LLM call or parsing: {str(e)}"
                    logger.exception(f"{error_message}", extra=extra_log_data) # Use exception for traceback
                    topic_analysis = None
                    search_keywords = None

        except Exception as e:
            error_message = f"Topic analysis failed due to an unexpected error: {str(e)}"
            logger.exception(f"{error_message} for query '{initial_query}'", extra=extra_log_data) # Use exception
            topic_analysis = None # Ensure reset on failure
            search_keywords = None

        # --- 결과 반환 ---
        end_time = datetime.now(timezone.utc)
        node2_processing_stats = (end_time - start_time).total_seconds()

        # Final state update data
        update_data: Dict[str, Any] = {
            "topic_analysis": topic_analysis,
            "search_keywords": search_keywords if search_keywords is not None else [], # Return [] if None
            "node2_processing_stats": node2_processing_stats,
            "error_message": error_message # Will be None on success
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if error_message else logger.info
        log_level(f"Topic analysis result: {len(update_data['search_keywords'])} keywords extracted. Error: {error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node2_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}