# app/nodes/08_news_summarizer_node.py (Improved Version)

import asyncio
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
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

class NewsSummarizerNode:
    """
    뉴스 기사를 요약하고 FEQA를 사용하여 사실성을 검증합니다.
    - LLMService를 사용하여 모든 LLM 호출 수행.
    - 2단계 요약 (추출 -> 요약) 및 FEQA (QA 생성 -> 검증) 프로세스 포함.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
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
        if not llm_client: raise ValueError("LLMService is required for NewsSummarizerNode")
        self.llm_client = llm_client
        # self.langsmith_service = langsmith_service
        logger.info("NewsSummarizerNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        # LLM 호출 파라미터
        self.llm_temp_extract = float(config.get("llm_temperature_extract", settings.DEFAULT_LLM_TEMP_EXTRACT))
        self.max_tokens_extract = int(config.get("max_tokens_extract", settings.DEFAULT_MAX_TOKENS_EXTRACT))
        self.llm_temp_summarize = float(config.get("llm_temperature_summarize", settings.DEFAULT_LLM_TEMP_SUMMARIZE))
        self.max_tokens_summarize = int(config.get("max_tokens_summarize", settings.DEFAULT_MAX_TOKENS_SUMMARIZE))
        self.llm_temp_qa_gen = float(config.get("llm_temperature_qa_gen", settings.DEFAULT_LLM_TEMP_QAGEN))
        self.max_tokens_qa_gen = int(config.get("max_tokens_qa_gen", settings.DEFAULT_MAX_TOKENS_QAGEN))
        self.llm_temp_qa_verify = float(config.get("llm_temperature_qa_verify", settings.DEFAULT_LLM_TEMP_QAVERIFY))
        self.max_tokens_qa_verify = int(config.get("max_tokens_qa_verify", settings.DEFAULT_MAX_TOKENS_QAVERIFY))
        # 텍스트 길이 제한
        self.max_article_text_len = int(config.get("max_article_text_len", settings.MAX_ARTICLE_TEXT_LEN))
        # FEQA 임계값
        self.feqa_threshold = float(config.get("feqa_threshold", settings.DEFAULT_FEQA_THRESHOLD))
        # 동시성 및 처리 제한
        self.max_articles_to_summarize = int(config.get('max_articles_to_summarize', settings.DEFAULT_MAX_ARTICLES_SUMMARIZE))
        self.concurrency_limit = int(config.get('summarizer_concurrency', settings.DEFAULT_SUMMARIZER_CONCURRENCY))

        logger.debug(f"Runtime config loaded. Max articles: {self.max_articles_to_summarize}, Concurrency: {self.concurrency_limit}, FEQA Threshold: {self.feqa_threshold}")
        logger.debug(f"LLM Temps (E/S/QG/QV): {self.llm_temp_extract}/{self.llm_temp_summarize}/{self.llm_temp_qa_gen}/{self.llm_temp_qa_verify}")
        logger.debug(f"Max Tokens (E/S/QG/QV): {self.max_tokens_extract}/{self.max_tokens_summarize}/{self.max_tokens_qa_gen}/{self.max_tokens_qa_verify}")
        logger.debug(f"Max Article Text Length: {self.max_article_text_len}")


    # --- LLM 호출 래퍼 (재시도 적용) ---
    # 참고: retry_if_exception_type(Exception)은 광범위함. LLMService 관련 특정 예외로 제한 고려.
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

    # --- Stage 1: 키포인트 추출 ---
    def _create_extraction_prompt_en(self, title: str, text: str) -> str:
        """키포인트 추출 프롬프트 생성 (길이 제한 적용)"""
        truncated_text = text[:self.max_article_text_len] + ("..." if len(text) > self.max_article_text_len else "")
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

    async def _extract_key_points(self, article: Dict[str, Any], trace_id: Optional[str]) -> List[str]:
        """LLM을 사용하여 기사에서 키포인트 추출"""
        title = article.get('title', 'N/A')
        text = article.get('text', '')
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not text: return []

        prompt = self._create_extraction_prompt_en(title, text)
        try:
            response = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_extract,
                max_tokens=self.max_tokens_extract,
                trace_id=trace_id
            )
            # 번호 매겨진 리스트 파싱 로직 개선
            lines = response.strip().split('\n')
            extracted_points = []
            for line in lines:
                match = re.match(r'^\s*\d+[\.\)]\s*(.+)', line.strip())
                if match:
                    point = match.group(1).strip()
                    if point: extracted_points.append(point)

            if not extracted_points and response.strip():
                 extracted_points = [line.strip() for line in response.strip().split('\n') if line.strip() and not line.strip().isnumeric()]

            logger.debug(f"{log_prefix} Extracted {len(extracted_points)} key points for article: {article.get('url')}")
            return extracted_points[:8] # 최대 8개 제한
        except RetryError as e:
            logger.error(f"{log_prefix} Failed to extract key points after retries for {article.get('url')}: {e}")
            return []
        except Exception as e:
            logger.error(f"{log_prefix} Error extracting key points for {article.get('url')}: {e}", exc_info=True)
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

    async def _generate_summary(self, article: Dict[str, Any], key_points: List[str], trace_id: Optional[str]) -> str:
        """키포인트를 기반으로 요약 생성"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not key_points: return ""
        title = article.get('title', 'N/A')

        prompt = self._create_summary_prompt_en(title, key_points)
        try:
            summary = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_summarize,
                max_tokens=self.max_tokens_summarize,
                trace_id=trace_id
            )
            logger.debug(f"{log_prefix} Generated summary (length: {len(summary)}) for article: {article.get('url')}")
            return summary
        except RetryError as e:
            logger.error(f"{log_prefix} Failed to generate summary after retries for {article.get('url')}: {e}")
            return ""
        except Exception as e:
            logger.error(f"{log_prefix} Error generating summary for {article.get('url')}: {e}", exc_info=True)
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

    async def _generate_qa_pairs(self, summary: str, trace_id: Optional[str]) -> List[Dict[str, str]]:
        """요약문에서 QA 쌍 생성"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not summary: return []

        prompt = self._create_qa_generation_prompt_en(summary)
        try:
            response = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_qa_gen,
                max_tokens=self.max_tokens_qa_gen,
                trace_id=trace_id
            )
            # QA 파싱 로직 개선
            qa_pairs = []
            pattern = re.compile(r"Q\d+:\s*(.*?)\s*\nA\d+:\s*(.*)", re.IGNORECASE | re.DOTALL)
            matches = pattern.findall(response)
            for q, a in matches:
                 q = q.strip()
                 a = a.strip()
                 if q and a: qa_pairs.append({"question": q, "answer": a})

            if not qa_pairs: # Fallback
                 lines = response.strip().split('\n')
                 current_q = None
                 for line in lines:
                      line = line.strip()
                      if not line: continue
                      q_match = re.match(r"^[Qq]\d?[:\.]?\s*(.*)", line)
                      a_match = re.match(r"^[Aa]\d?[:\.]?\s*(.*)", line)
                      if q_match: current_q = q_match.group(1).strip()
                      elif a_match and current_q:
                           answer = a_match.group(1).strip()
                           if answer: qa_pairs.append({"question": current_q, "answer": answer})
                           current_q = None # Reset after finding answer

            logger.debug(f"{log_prefix} Generated {len(qa_pairs)} QA pairs from summary.")
            return qa_pairs[:3] # 최대 3개 반환
        except RetryError as e:
            logger.error(f"{log_prefix} Failed to generate QA pairs after retries: {e}")
            return []
        except Exception as e:
            logger.error(f"{log_prefix} Error generating QA pairs: {e}", exc_info=True)
            return []

    # --- FEQA: Step 2 - 원문 기반 답변 검증 ---
    def _create_answer_verification_prompt_en(self, question: str, article_text: str) -> str:
        """원문에서 답변 찾는 프롬프트 (길이 제한 적용)"""
        truncated_text = article_text[:self.max_article_text_len] + ("..." if len(article_text) > self.max_article_text_len else "")
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

    async def _verify_answer_with_llm(self, question: str, expected_answer: str, article_text: str, trace_id: Optional[str]) -> bool:
        """LLM을 사용하여 원문에서 답변 검증"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not article_text: return False

        prompt = self._create_answer_verification_prompt_en(question, article_text)
        try:
            llm_answer = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_qa_verify,
                max_tokens=self.max_tokens_qa_verify,
                trace_id=trace_id
            )

            logger.debug(f"{log_prefix} FEQA Verify - Q: '{question[:50]}...' | Expected: '{expected_answer[:50]}...' | LLM Found: '{llm_answer[:50]}...'")

            if "answer not found" in llm_answer.lower(): return False

            # 답변 유사성/포함 관계 확인 (간단 버전)
            norm_expected = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', expected_answer.lower())).strip()
            norm_llm = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', llm_answer.lower())).strip()

            return bool(norm_expected and norm_llm and norm_expected in norm_llm)

        except RetryError as e:
             logger.error(f"{log_prefix} LLM answer verification failed after retries for Q '{question[:50]}...': {e}")
             return False
        except Exception as e:
            logger.error(f"{log_prefix} Error during LLM answer verification for Q '{question[:50]}...': {e}", exc_info=True)
            return False

    async def _calculate_feqa_score(self, qa_pairs: List[Dict[str, str]], article_text: str, trace_id: Optional[str]) -> float:
        """생성된 QA 쌍을 원문과 비교하여 FEQA 점수 계산"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not qa_pairs: return 0.0

        verification_tasks = []
        for pair in qa_pairs:
             question = pair.get("question")
             expected_answer = pair.get("answer")
             if question and expected_answer:
                  verification_tasks.append(
                       self._verify_answer_with_llm(question, expected_answer, article_text, trace_id)
                  )

        if not verification_tasks: return 0.0

        verified_count = 0
        verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
        for result in verification_results:
             if isinstance(result, bool) and result is True:
                  verified_count += 1
             elif isinstance(result, Exception):
                  logger.error(f"{log_prefix} FEQA verification task failed with exception: {result}")

        score = verified_count / len(verification_tasks) if verification_tasks else 0.0
        logger.info(f"{log_prefix} FEQA Score: {score:.3f} ({verified_count} / {len(verification_tasks)} verified)")
        return score

    # --- 개별 기사 처리 로직 ---
    async def _process_article(self, article: Dict[str, Any], trace_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """단일 기사 처리: 키포인트 추출 -> 요약 생성 -> FEQA 계산"""
        original_url = article.get('url', article.get('original_url_from_source', 'Unknown URL'))
        article_text = article.get('text', '')
        log_prefix = f"[{trace_id}][Article:{original_url}]"
        logger.info(f"{log_prefix} Processing article...")

        if not article_text:
             logger.warning(f"{log_prefix} Article text is empty. Skipping.")
             return None

        try:
            # 1. 키포인트 추출
            key_points = await self._extract_key_points(article, trace_id)
            if not key_points:
                logger.warning(f"{log_prefix} Failed to extract key points. Skipping article.")
                return None

            # 2. 요약 생성
            summary_text = await self._generate_summary(article, key_points, trace_id)
            if not summary_text:
                logger.warning(f"{log_prefix} Failed to generate summary. Skipping article.")
                return None

            # 3. QA 쌍 생성 (FEQA용)
            qa_pairs = await self._generate_qa_pairs(summary_text, trace_id)
            feqa_score = 0.0 # 기본값
            if not qa_pairs:
                 logger.warning(f"{log_prefix} Failed to generate QA pairs for FEQA. Score set to 0.")
            else:
                 # 4. QA 검증 및 FEQA 점수 계산
                 feqa_score = await self._calculate_feqa_score(qa_pairs, article_text, trace_id)

            # FEQA 임계값 기반 필터링
            if feqa_score < self.feqa_threshold:
                 logger.warning(f"{log_prefix} Summary failed FEQA check (Score: {feqa_score:.3f} < Threshold: {self.feqa_threshold}). Discarding.")
                 return None

            logger.info(f"{log_prefix} Successfully processed article (FEQA: {feqa_score:.3f})")
            # ComicState의 news_summaries 필드 형식에 맞춰 반환
            return {
                "original_url": original_url, # 식별용 원본 URL
                "summary_text": summary_text,
                "feqa_score": round(feqa_score, 3) # 소수점 3자리
                # 필요시 key_points, qa_pairs 등 추가 정보 포함 가능
            }

        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error processing article: {e}")
            return None

    # --- 노드 실행 메인 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """뉴스 요약 및 FEQA 검증 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing NewsSummarizerNode...")

        articles = state.articles or []
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        if not articles:
            logger.warning(f"{log_prefix} No articles found in state. Skipping summarization.")
            processing_stats['news_summarizer_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"news_summaries": [], "processing_stats": processing_stats}

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        # --------------------------

        articles_to_process = articles[:self.max_articles_to_summarize]
        logger.info(f"{log_prefix} Starting summarization for {len(articles_to_process)} articles (Max: {self.max_articles_to_summarize}, Concurrency: {self.concurrency_limit})...")

        tasks = []
        for article in articles_to_process:
            async def task_with_semaphore(art):
                 async with semaphore:
                      return await self._process_article(art, trace_id)
            tasks.append(task_with_semaphore(article))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_summaries: List[Dict[str, Any]] = []
        failed_count = 0
        task_errors: List[str] = []

        for i, res in enumerate(results):
             original_url = articles_to_process[i].get('url', f'Article_{i}')
             if isinstance(res, dict) and res is not None:
                  successful_summaries.append(res)
             else:
                  failed_count += 1
                  if isinstance(res, Exception):
                       err_msg = f"Summarization task failed for {original_url}: {res}"
                       logger.error(f"{log_prefix} {err_msg}")
                       task_errors.append(err_msg)
                  # else: res is None (_process_article 내부 로깅)

        logger.info(f"{log_prefix} News summarization complete. Generated {len(successful_summaries)} valid summaries. Failed/Skipped: {failed_count}.")

        end_time = datetime.now(timezone.utc)
        processing_stats['news_summarizer_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} NewsSummarizerNode finished in {processing_stats['news_summarizer_node_time']:.2f} seconds.")

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"{log_prefix} Some errors occurred during news summarization: {final_error_message}")

        # ComicState 업데이트를 위한 결과 반환
        update_data: Dict[str, Any] = {
            "news_summaries": successful_summaries,
            "processing_stats": processing_stats,
            "error_message": final_error_message # 부분 실패 요약
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}