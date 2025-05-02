# app/nodes/14_idea_generator_node.py (Improved Version)

import asyncio
import re
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 기본값 참조용
from app.services.llm_server_client_v2 import LLMService # LLM 서비스 클라이언트
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# 로거 설정
logger = get_logger(__name__)

class IdeaGeneratorNode:
    """
    최종 요약과 트렌드 점수를 기반으로 4컷 만화 아이디어를 생성합니다.
    - LLMService를 사용하여 아이디어 생성 및 자체 평가 수행 (JSON 형식).
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["final_summary", "trend_scores", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["comic_ideas", "processing_stats", "error_message"]

    # LLMService 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        llm_client: LLMService,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        if not llm_client: raise ValueError("LLMService is required for IdeaGeneratorNode")
        self.llm_client = llm_client
        # self.langsmith_service = langsmith_service
        logger.info("IdeaGeneratorNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.llm_temp_creative = float(config.get("llm_temperature_creative", settings.DEFAULT_LLM_TEMP_CREATIVE))
        self.max_tokens_idea = int(config.get("llm_max_tokens_idea", settings.DEFAULT_MAX_TOKENS_IDEA))
        self.top_n_trends_for_prompt = int(config.get("trends_report_top_n", settings.DEFAULT_TRENDS_REPORT_TOP_N))

        logger.debug(f"Runtime config loaded. LLM Temp: {self.llm_temp_creative}, Max Tokens: {self.max_tokens_idea}, Top Trends: {self.top_n_trends_for_prompt}")


    # --- LLM 호출 래퍼 (재시도 적용) ---
    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float, max_tokens: int,
                                   trace_id: Optional[str] = None, **kwargs) -> str:
        """LLMService.generate_text를 재시도 로직과 함께 호출"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Calling LLMService (Temp: {temperature}, MaxTokens: {max_tokens})...")

        result = await self.llm_client.generate_text(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        if "error" in result:
            error_msg = result['error']
            logger.error(f"{log_prefix} LLMService call failed: {error_msg}")
            raise RuntimeError(f"LLMService error: {error_msg}")
        elif "generated_text" not in result or not isinstance(result.get("generated_text"), str) or not result["generated_text"].strip():
            logger.error(f"{log_prefix} LLMService returned invalid or empty text. Response: {result}")
            raise ValueError("LLMService returned invalid or empty text")
        else:
            logger.debug(f"{log_prefix} LLMService call successful.")
            return result["generated_text"].strip()

    # --- 프롬프트 생성 ---
    def _prepare_trend_info_for_prompt(self, trend_scores: List[Dict[str, Any]]) -> str:
        """LLM 프롬프트용 상위 트렌드 정보 포맷팅"""
        if not trend_scores: return "No specific trending topics identified recently."

        # Node 12에서 이미 점수 내림차순 정렬됨
        top_trends = [t for t in trend_scores if t.get('score', 0) > 0][:self.top_n_trends_for_prompt]

        if not top_trends: return "No significant trending topics identified recently."

        trend_lines = [f"- \"{trend.get('keyword', 'N/A')}\" (Score: {trend.get('score', 0):.1f})" for trend in top_trends]
        return "Recent potentially relevant trending keywords:\n" + "\n".join(trend_lines)

    def _create_idea_prompt_en(self, summary: str, trend_info: str) -> str:
        """아이디어 생성 프롬프트 (JSON 출력 요구)"""
        # 프롬프트 내용은 이전과 동일, JSON 출력 형식 명시
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a creative assistant tasked with generating compelling 4-panel comic ideas based on news summaries and trending topics. Provide exactly 5 diverse ideas in the specified JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>
Generate 5 distinct 4-panel comic ideas inspired by the following news summary and trending keywords. For each idea, provide a catchy title, a brief concept description, and a self-assessed creativity score (0.0 to 1.0).

[News Summary]
{summary}

[Trending Keywords Context]
{trend_info}

[Instructions]
1.  Create exactly 5 different comic ideas.
2.  Each idea should be suitable for a 4-panel comic format.
3.  Ideas should creatively connect the news summary and/or trending keywords.
4.  For each idea, include:
    * `idea_title`: A short, engaging title (string).
    * `concept`: A 1-2 sentence description of the comic's premise or storyline (string).
    * `creative_score`: Your assessment of the idea's creativity and potential interest (float between 0.0 and 1.0).
5.  Respond ONLY with a valid JSON list containing the 5 idea objects. Ensure keys and string values use double quotes.

[Required JSON Output Format]
```json
[
  {{
    "idea_title": "Example Title 1",
    "concept": "Example concept description 1.",
    "creative_score": 0.85
  }},
  {{
    "idea_title": "Example Title 2",
    "concept": "Example concept description 2.",
    "creative_score": 0.70
  }},
  {{
    "idea_title": "Example Title 3",
    "concept": "Example concept description 3.",
    "creative_score": 0.90
  }},
  {{
    "idea_title": "Example Title 4",
    "concept": "Example concept description 4.",
    "creative_score": 0.65
  }},
  {{
    "idea_title": "Example Title 5",
    "concept": "Example concept description 5.",
    "creative_score": 0.80
  }}
]
```<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```json
"""
        return prompt

    # --- 응답 파싱 ---
    def _parse_llm_response(self, response_json: str, trace_id: Optional[str]) -> List[Dict[str, Any]]:
        """LLM의 아이디어 JSON 응답 파싱 및 검증"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        try:
            logger.debug(f"{log_prefix} Raw LLM response for ideas: {response_json[:500]}...")
            match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_json, re.DOTALL | re.IGNORECASE)
            if match: json_str = match.group(1)
            else:
                json_str = response_json.strip()
                if not json_str.startswith('['): json_str = '[' + json_str
                if not json_str.endswith(']'): json_str = json_str + ']'

            logger.debug(f"{log_prefix} Cleaned JSON for parsing: {json_str[:500]}...")
            ideas = json.loads(json_str)

            if not isinstance(ideas, list): raise ValueError("LLM response is not a JSON list.")

            validated_ideas = []
            required_keys = {"idea_title", "concept", "creative_score"}
            for idx, idea in enumerate(ideas):
                if isinstance(idea, dict) and required_keys.issubset(idea.keys()):
                    title = idea.get('idea_title')
                    concept = idea.get('concept')
                    score_val = idea.get('creative_score')
                    if isinstance(title, str) and title.strip() and \
                       isinstance(concept, str) and concept.strip() and \
                       isinstance(score_val, (float, int)):
                        score = max(0.0, min(1.0, float(score_val)))
                        validated_ideas.append({
                            "idea_title": title.strip(),
                            "concept": concept.strip(),
                            "creative_score": round(score, 3)
                        })
                    else: logger.warning(f"{log_prefix} Idea #{idx+1} has invalid data: {idea}")
                else: logger.warning(f"{log_prefix} Idea #{idx+1} format invalid: {idea}")

            if len(validated_ideas) < 5: logger.warning(f"{log_prefix} LLM returned fewer than 5 valid ideas ({len(validated_ideas)}).")
            elif len(validated_ideas) > 5: logger.warning(f"{log_prefix} LLM returned more than 5 ideas. Using first 5.")

            return validated_ideas[:5] # 최대 5개 반환

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"{log_prefix} Failed parsing/validating LLM idea response: {e}. Response: '{response_json[:200]}...'")
            return []
        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error parsing LLM idea response: {e}")
            return []

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """아이디어 생성 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing IdeaGeneratorNode...")

        final_summary = state.final_summary or ""
        trend_scores = state.trend_scores or []
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        if not final_summary:
            logger.error(f"{log_prefix} Final summary is missing. Cannot generate ideas.")
            processing_stats['idea_generator_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"comic_ideas": [], "processing_stats": processing_stats, "error_message": "Final summary is required."}

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        # --------------------------

        logger.info(f"{log_prefix} Starting comic idea generation...")
        error_message: Optional[str] = None
        comic_ideas: List[Dict[str, Any]] = []

        try:
            # --- LLM 입력 준비 및 호출 ---
            trend_info = self._prepare_trend_info_for_prompt(trend_scores)
            prompt = self._create_idea_prompt_en(final_summary, trend_info)
            llm_kwargs = {"response_format": {"type": "json_object"}} # JSON 모드 요청

            response_str = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_creative,
                max_tokens=self.max_tokens_idea,
                trace_id=trace_id,
                **llm_kwargs
            )
            comic_ideas = self._parse_llm_response(response_str, trace_id)

            if not comic_ideas:
                 error_message = "Failed to generate or parse any valid comic ideas from LLM."
                 logger.error(f"{log_prefix} {error_message}")
            else:
                 logger.info(f"{log_prefix} Successfully generated {len(comic_ideas)} comic ideas.")

        except RetryError as e: # 모든 재시도 실패
            error_message = f"LLM call failed after multiple retries: {e}"
            logger.error(f"{log_prefix} {error_message}")
        except Exception as e: # 기타 예외
            error_message = f"Idea generation failed: {str(e)}"
            logger.exception(f"{log_prefix} {error_message}")
            comic_ideas = [] # 실패 시 빈 리스트

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['idea_generator_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} IdeaGeneratorNode finished in {processing_stats['idea_generator_node_time']:.2f} seconds.")

        # ComicState 업데이트
        update_data: Dict[str, Any] = {
            "comic_ideas": comic_ideas,
            "processing_stats": processing_stats,
            "error_message": error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}