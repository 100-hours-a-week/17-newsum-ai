# app/nodes/08_news_summarizer_node.py

import asyncio
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# --- 프로젝트 구성 요소 임포트 ---
from app.config.settings import settings            # 설정 객체 (재시도 횟수 등 참조)
from app.services.llm_server_client_v2 import LLMService # 실제 LLM 서비스 클라이언트
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger             # 로거 유틸리티
from app.workflows.state import ComicState          # 워크플로우 상태 모델

# --- 로거 설정 ---
logger = get_logger("NewsSummarizerNode")

class NewsSummarizerNode:
    """
    (Refactored) 뉴스 기사를 요약하고 FEQA를 사용하여 사실성을 검증합니다.
    - LLMService를 사용하여 모든 LLM 호출 수행.
    - 2단계 요약 (추출 -> 요약) 및 FEQA (QA 생성 -> 검증) 프로세스 포함.
    - 설정은 상태의 config 딕셔너리에서 로드.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["articles", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["news_summaries", "processing_stats", "error_message"]

    # LLMService 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        llm_client: LLMService,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        self.llm_client = llm_client
        # self.langsmith_service = langsmith_service
        logger.info("NewsSummarizerNode initialized with LLMService.")

    # --- LLM 호출 래퍼 (재시도 적용) ---
    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES), # 설정에서 재시도 횟수 사용
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception), # 재시도 조건 (개선 가능)
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float, max_tokens: int,
                                   trace_id: Optional[str] = None, **kwargs) -> str:
        """(Refactored) LLMService.generate_text를 재시도 로직과 함께 호출"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        # max_tokens 등의 파라미터를 kwargs로 전달 가능하도록 수정
        # 모델명은 LLMService 내부 설정 또는 kwargs를 통해 관리될 수 있음
        logger.debug(f"{log_prefix} Calling LLMService (Temp: {temperature}, MaxTokens: {max_tokens})...")

        result = await self.llm_client.generate_text(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs # response_format 등 추가 인자 전달
        )

        if "error" in result:
            error_msg = result['error']
            logger.error(f"{log_prefix} LLMService call failed: {error_msg}")
            raise RuntimeError(f"LLMService error: {error_msg}") # 재시도 유발
        elif "generated_text" not in result or not isinstance(result.get("generated_text"), str) or not result["generated_text"].strip():
            logger.error(f"{log_prefix} LLMService returned invalid or empty text. Response: {result}")
            raise ValueError("LLMService returned invalid or empty text") # 재시도 유발 또는 오류 처리
        else:
            logger.debug(f"{log_prefix} LLMService call successful.")
            return result["generated_text"].strip() # 공백 제거 후 반환

    # --- Stage 1: 키포인트 추출 ---
    def _create_extraction_prompt_en(self, title: str, text: str, max_text_len: int) -> str:
        """키포인트 추출 프롬프트 생성 (텍스트 길이 제한 적용)"""
        truncated_text = text[:max_text_len] + ("..." if len(text) > max_text_len else "")
        # 프롬프트 내용은 이전과 동일
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert text analyst. Your task is to extract the most important and factual sentences or key phrases from the provided news article text. Focus on conveying the core information accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>
Extract the 6-8 most important key sentences or factual phrases from the following news article. Present them as a numbered list. Use the original wording from the text as much as possible.

[Article Title]
{title}

[Article Text]
{truncated_text}

[Instructions]
- Extract 6 to 8 key points.
- Maintain original wording.
- Output as a numbered list (e.g., "1. Sentence one.").
- Focus on facts and core information presented.

[Extraction Result]
1. <|eot_id|><|start_header_id|>assistant<|end_header_id|>
1. """
        return prompt

    async def _extract_key_points(self, article: Dict[str, Any], config: Dict, trace_id: Optional[str]) -> List[str]:
        """LLM을 사용하여 기사에서 키포인트 추출"""
        title = article.get('title', 'N/A')
        text = article.get('text', '')
        if not text: return []

        # 설정값 로드
        llm_model = config.get("llm_model", "default_model") # 모델명은 참고용, 실제 사용 모델은 LLMService 설정 따름
        temperature = config.get("llm_temperature_extract", 0.1)
        max_tokens = config.get("max_tokens_extract", 512)
        max_text_len = config.get("max_article_text_len", 3500)

        prompt = self._create_extraction_prompt_en(title, text, max_text_len)
        try:
            response = await self._call_llm_with_retry(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, trace_id=trace_id
            )
            # 번호 매겨진 리스트 파싱 로직 개선
            lines = response.strip().split('\n')
            extracted_points = []
            for line in lines:
                # 정규표현식으로 더 정확하게 추출 (숫자, 점/괄호, 공백 후 내용)
                match = re.match(r'^\s*\d+[\.\)]\s*(.+)', line.strip())
                if match:
                    point = match.group(1).strip()
                    if point: extracted_points.append(point) # 빈 문자열 제외

            # 파싱 실패 시 간단한 분리 시도
            if not extracted_points and response.strip():
                 extracted_points = [line.strip() for line in response.strip().split('\n') if line.strip() and not line.strip().isnumeric()]

            logger.debug(f"[{trace_id}] Extracted {len(extracted_points)} key points for article: {article.get('url')}")
            return extracted_points[:8] # 최대 8개 제한
        except Exception as e:
            logger.error(f"[{trace_id}] Failed to extract key points for {article.get('url')}: {e}")
            return []

    # --- Stage 2: 요약 생성 ---
    def _create_summary_prompt_en(self, title: str, key_points: List[str]) -> str:
        """키포인트 기반 요약 생성 프롬프트"""
        key_points_text = "\n".join(f"- {point}" for point in key_points)
        # 프롬프트 내용은 이전과 동일
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a concise news summarizer. Your task is to write a brief, factual summary based *only* on the provided key points and title.<|eot_id|><|start_header_id|>user<|end_header_id|>
Based ONLY on the following key points and the article title, write a concise and factual summary of about 100-150 words. Do not add any information not present in the key points. Ensure the summary flows naturally.

[Article Title]
{title}

[Key Points]
{key_points_text}

[Instructions]
- Use ONLY the information from the key points and title.
- Write a summary of approximately 100-150 words.
- Maintain a neutral, factual tone.
- Combine the points into a coherent paragraph.

[Summary]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    async def _generate_summary(self, article: Dict[str, Any], key_points: List[str], config: Dict, trace_id: Optional[str]) -> str:
        """키포인트를 기반으로 요약 생성"""
        if not key_points: return ""
        title = article.get('title', 'N/A')

        # 설정값 로드
        llm_model = config.get("llm_model", "default_model")
        temperature = config.get("llm_temperature_summarize", 0.3)
        max_tokens = config.get("max_tokens_summarize", 300)

        prompt = self._create_summary_prompt_en(title, key_points)
        try:
            summary = await self._call_llm_with_retry(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, trace_id=trace_id
            )
            logger.debug(f"[{trace_id}] Generated summary (length: {len(summary)}) for article: {article.get('url')}")
            return summary # _call_llm_with_retry 에서 이미 strip() 처리됨
        except Exception as e:
            logger.error(f"[{trace_id}] Failed to generate summary for {article.get('url')}: {e}")
            return ""

    # --- FEQA: Step 1 - QA 쌍 생성 ---
    def _create_qa_generation_prompt_en(self, summary: str) -> str:
        """요약 기반 QA 쌍 생성 프롬프트"""
        # 프롬프트 내용은 이전과 동일
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are tasked with creating Question-Answer pairs to verify the factual content of a text. Generate questions whose answers are explicitly stated in the provided summary text.<|eot_id|><|start_header_id|>user<|end_header_id|>
Generate exactly 3 distinct factual question-answer pairs based ONLY on the following summary text. The answer for each question must be directly extractable from the summary text.

[Summary Text]
{summary}

[Instructions]
- Generate exactly 3 question-answer pairs.
- Questions should be factual and answerable directly from the text.
- Answers should be concise and directly quoted or derived from the text.
- Format the output strictly as follows:
Q1: [Question 1]
A1: [Answer 1]

Q2: [Question 2]
A2: [Answer 2]

Q3: [Question 3]
A3: [Answer 3]

[Generated QA Pairs]
Q1: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
Q1: """
        return prompt

    async def _generate_qa_pairs(self, summary: str, config: Dict, trace_id: Optional[str]) -> List[Dict[str, str]]:
        """요약문에서 QA 쌍 생성"""
        if not summary: return []

        # 설정값 로드
        llm_model = config.get("llm_model", "default_model")
        temperature = config.get("llm_temperature_qa_gen", 0.1)
        max_tokens = config.get("max_tokens_qa_gen", 512)

        prompt = self._create_qa_generation_prompt_en(summary)
        try:
            response = await self._call_llm_with_retry(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, trace_id=trace_id
            )
            # QA 파싱 로직 개선
            qa_pairs = []
            # 정규표현식을 사용하여 Qn: ..., An: ... 패턴 매칭
            pattern = re.compile(r"Q\d+:\s*(.*?)\s*\nA\d+:\s*(.*)", re.IGNORECASE)
            matches = pattern.findall(response)
            for q, a in matches:
                 q = q.strip()
                 a = a.strip()
                 if q and a: # 질문과 답변 모두 내용이 있는지 확인
                      qa_pairs.append({"question": q, "answer": a})

            # 매칭 실패 시 이전 방식 시도 (fallback)
            if not qa_pairs:
                 lines = response.strip().split('\n')
                 current_q = None
                 for line in lines:
                      line = line.strip()
                      if not line: continue
                      if (line.startswith("Q") or line.startswith("q")) and ":" in line:
                           current_q = line.split(":", 1)[1].strip()
                      elif (line.startswith("A") or line.startswith("a")) and ":" in line and current_q:
                           answer = line.split(":", 1)[1].strip()
                           if answer: qa_pairs.append({"question": current_q, "answer": answer})
                           current_q = None

            logger.debug(f"[{trace_id}] Generated {len(qa_pairs)} QA pairs from summary.")
            return qa_pairs[:3] # 최대 3개 반환
        except Exception as e:
            logger.error(f"[{trace_id}] Failed to generate QA pairs: {e}")
            return []

    # --- FEQA: Step 2 - 원문 기반 답변 검증 ---
    def _create_answer_verification_prompt_en(self, question: str, article_text: str, max_text_len: int) -> str:
        """원문에서 답변 찾는 프롬프트 (텍스트 길이 제한 적용)"""
        truncated_text = article_text[:max_text_len] + ("..." if len(article_text) > max_text_len else "")
        # 프롬프트 내용은 이전과 동일
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert extractive Question Answering system. Your task is to find the answer to a given question based *only* on the provided article text. If the answer is not found, respond with "Answer not found".<|eot_id|><|start_header_id|>user<|end_header_id|>
Answer the following question based ONLY on the provided article text. If the answer is not explicitly stated in the text, respond with "Answer not found".

[Question]
{question}

[Article Text]
{truncated_text}

[Instructions]
- Read the question and article text carefully.
- Find the shortest possible answer segment in the text that directly answers the question.
- If no direct answer is found, output exactly "Answer not found".
- Do not infer or add information not present in the text.

[Answer]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        return prompt

    async def _verify_answer_with_llm(self, question: str, expected_answer: str, article_text: str, config: Dict, trace_id: Optional[str]) -> bool:
        """LLM을 사용하여 원문에서 답변 검증"""
        if not article_text: return False

        # 설정값 로드
        llm_model = config.get("llm_model", "default_model")
        temperature = config.get("llm_temperature_qa_verify", 0.0)
        max_tokens = config.get("max_tokens_qa_verify", 100)
        max_text_len = config.get("max_article_text_len", 3500)

        prompt = self._create_answer_verification_prompt_en(question, article_text, max_text_len)
        try:
            llm_answer = await self._call_llm_with_retry(
                prompt=prompt, temperature=temperature, max_tokens=max_tokens, trace_id=trace_id
            )

            logger.debug(f"[{trace_id}] FEQA Verify - Q: '{question}' | Expected: '{expected_answer}' | LLM Found: '{llm_answer}'")

            if "answer not found" in llm_answer.lower(): return False

            # 답변 유사성/포함 관계 확인 (간단 버전)
            # 더 정확한 비교를 위해 정규화 강화 (소문자, 구두점 제거, 연속 공백 제거)
            norm_expected = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', expected_answer.lower())).strip()
            norm_llm = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', llm_answer.lower())).strip()

            # LLM이 추출한 답변 내에 예상 답변의 핵심이 포함되는지 확인
            return bool(norm_expected and norm_llm and norm_expected in norm_llm)

        except Exception as e:
            logger.error(f"[{trace_id}] Error during LLM answer verification for Q '{question}': {e}")
            return False

    async def _calculate_feqa_score(self, qa_pairs: List[Dict[str, str]], article_text: str, config: Dict, trace_id: Optional[str]) -> float:
        """생성된 QA 쌍을 원문과 비교하여 FEQA 점수 계산"""
        if not qa_pairs: return 0.0

        verification_tasks = []
        for pair in qa_pairs:
             question = pair.get("question")
             expected_answer = pair.get("answer")
             if question and expected_answer:
                  verification_tasks.append(
                       self._verify_answer_with_llm(question, expected_answer, article_text, config, trace_id)
                  )

        if not verification_tasks: return 0.0

        verified_count = 0
        verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
        for result in verification_results:
             if isinstance(result, bool) and result is True:
                  verified_count += 1
             elif isinstance(result, Exception): # gather에서 반환된 예외 처리
                  logger.error(f"[{trace_id}] FEQA verification task failed with exception: {result}")

        score = verified_count / len(verification_tasks) # 실패한 태스크도 분모에 포함
        logger.info(f"[{trace_id}] FEQA Score: {score:.3f} ({verified_count} / {len(verification_tasks)} verified)")
        return score

    # --- 개별 기사 처리 로직 ---
    async def _process_article(self, article: Dict[str, Any], config: Dict, trace_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """단일 기사 처리: 키포인트 추출 -> 요약 생성 -> FEQA 계산"""
        original_url = article.get('url', 'Unknown URL')
        article_text = article.get('text', '')
        log_prefix = f"[{trace_id}][Article:{original_url}]"
        logger.info(f"{log_prefix} Starting processing...")

        # 설정 로드
        feqa_threshold = float(config.get("feqa_threshold", 0.5))

        if not article_text:
             logger.warning(f"{log_prefix} Article text is empty. Skipping.")
             return None

        try:
            # 1. 키포인트 추출
            key_points = await self._extract_key_points(article, config, trace_id)
            if not key_points:
                logger.warning(f"{log_prefix} Failed to extract key points. Skipping article.")
                return None

            # 2. 요약 생성
            summary_text = await self._generate_summary(article, key_points, config, trace_id)
            if not summary_text:
                logger.warning(f"{log_prefix} Failed to generate summary. Skipping article.")
                return None

            # 3. QA 쌍 생성 (FEQA용)
            qa_pairs = await self._generate_qa_pairs(summary_text, config, trace_id)
            if not qa_pairs: # QA 생성 실패 시 FEQA 점수 0으로 처리하고 진행할 수 있음
                 logger.warning(f"{log_prefix} Failed to generate QA pairs for FEQA. Score will be 0.")
                 feqa_score = 0.0
            else:
                 # 4. QA 검증 및 FEQA 점수 계산
                 feqa_score = await self._calculate_feqa_score(qa_pairs, article_text, config, trace_id)

            # FEQA 임계값 기반 필터링
            if feqa_score < feqa_threshold:
                 logger.warning(f"{log_prefix} Summary failed FEQA check (Score: {feqa_score:.3f} < Threshold: {feqa_threshold}). Discarding.")
                 return None # 임계값 미만 시 요약 폐기

            logger.info(f"{log_prefix} Successfully processed article (FEQA: {feqa_score:.3f})")
            # ComicState의 news_summaries 필드 형식에 맞춰 반환
            return {
                "original_url": original_url,
                "summary_text": summary_text,
                "feqa_score": round(feqa_score, 3) # 소수점 3자리까지 반올림
            }

        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error processing article: {e}")
            return None # 오류 발생 시 해당 기사 처리 중단

    # --- 노드 실행 메인 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """뉴스 요약 및 FEQA 검증 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing NewsSummarizerNode...")

        # 상태 및 설정 로드
        articles = state.articles or []
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        if not articles:
            logger.warning(f"{log_prefix} No articles found in state. Skipping summarization.")
            return {"news_summaries": [], "processing_stats": processing_stats}

        # 설정값 가져오기
        max_articles_to_summarize = config.get('max_articles', 5) # 요약할 최대 기사 수
        concurrency_limit = config.get('summarizer_concurrency', 3) # 동시 LLM 호출 수

        articles_to_process = articles[:max_articles_to_summarize]
        logger.info(f"{log_prefix} Starting summarization for {len(articles_to_process)} articles (max: {max_articles_to_summarize}). Concurrency: {concurrency_limit}")

        # 동시성 제어를 위한 세마포어
        semaphore = asyncio.Semaphore(concurrency_limit)

        # 비동기 작업 생성
        tasks = []
        for article in articles_to_process:
            # 각 기사 처리 작업을 세마포어로 감싸서 동시성 제어
            async def task_with_semaphore(art):
                 async with semaphore:
                      return await self._process_article(art, config, state.trace_id)
            tasks.append(task_with_semaphore(article))

        # 모든 작업 실행 및 결과 수집
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 처리
        successful_summaries: List[Dict[str, Any]] = []
        failed_count = 0
        for i, res in enumerate(results):
             original_url = articles_to_process[i].get('url', f'Article_{i}')
             if isinstance(res, dict) and res is not None: # 성공적으로 처리된 결과
                  successful_summaries.append(res)
             else:
                  failed_count += 1
                  if isinstance(res, Exception): # gather에서 반환된 예외 로깅
                       logger.error(f"{log_prefix} Summarization task failed for {original_url} with exception: {res}")
                  # None인 경우 (_process_article 내부에서 이미 로깅됨)

        logger.info(f"{log_prefix} News summarization complete. Generated {len(successful_summaries)} valid summaries. Failed/Skipped articles: {failed_count}.")

        # 처리 시간 기록 및 반환
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['news_summarizer_node_time'] = node_processing_time
        logger.info(f"{log_prefix} NewsSummarizerNode finished in {node_processing_time:.2f} seconds.")

        # TODO: LangSmith 로깅

        # ComicState 업데이트를 위한 결과 반환
        update_data: Dict[str, Any] = {
            "news_summaries": successful_summaries, # 성공한 요약 목록
            "processing_stats": processing_stats,  # 업데이트된 처리 통계
            "error_message": None # 부분적 실패는 오류 메시지로 간주하지 않음 (필요시 로깅 참고)
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}