# app/nodes/08_news_summarizer_node.py (Refactored)

import asyncio
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.services.llm_server_client_v2 import LLMService
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

class NewsSummarizerNode:
    """
    뉴스 기사를 요약하고 FEQA를 사용하여 사실성을 검증합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["articles", "trace_id", "comic_id", "config"] # MODIFIED: Added comic_id
    outputs: List[str] = ["news_summaries", "node8_processing_stats", "error_message"]

    def __init__(self, llm_client: LLMService):
        if not llm_client: raise ValueError("LLMService is required for NewsSummarizerNode")
        self.llm_client = llm_client
        logger.info("NewsSummarizerNode initialized.")

    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict): # MODIFIED: Added extra_log_data
        self.llm_temp_extract = float(config.get("llm_temperature_extract", settings.LLM_TEMPERATURE_EXTRACT))
        self.max_tokens_extract = int(config.get("max_tokens_extract", settings.MAX_TOKENS_EXTRACT))
        self.llm_temp_summarize = float(config.get("llm_temperature_summarize", settings.LLM_TEMPERATURE_SUMMARIZE))
        self.max_tokens_summarize = int(config.get("max_tokens_summarize", settings.MAX_TOKENS_SUMMARIZE))
        self.llm_temp_qa_gen = float(config.get("llm_temperature_qa_gen", settings.LLM_TEMPERATURE_QA_GEN))
        self.max_tokens_qa_gen = int(config.get("max_tokens_qa_gen", settings.MAX_TOKENS_QA_GEN))
        self.llm_temp_qa_verify = float(config.get("llm_temperature_qa_verify", settings.LLM_TEMPERATURE_QA_VERIFY))
        self.max_tokens_qa_verify = int(config.get("max_tokens_qa_verify", settings.MAX_TOKENS_QA_VERIFY))

        self.max_article_text_len = int(config.get("max_article_text_len", settings.MAX_ALT_TEXT_LEN))
        self.feqa_threshold = float(config.get("feqa_threshold", settings.FEQA_THRESHOLD))
        self.max_articles_to_summarize = int(config.get('max_articles_to_summarize', settings.MAX_ARTICLES_SUMMARIZE))

        self.concurrency_limit = int(config.get('summarizer_concurrency', settings.SUMMARIZER_CONCURRENCY))

        logger.debug(f"Runtime config loaded. Max articles: {self.max_articles_to_summarize}, Concurrency: {self.concurrency_limit}, FEQA Threshold: {self.feqa_threshold}", extra=extra_log_data) # MODIFIED
        logger.debug(f"LLM Temps (E/S/QG/QV): {self.llm_temp_extract}/{self.llm_temp_summarize}/{self.llm_temp_qa_gen}/{self.llm_temp_qa_verify}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Max Tokens (E/S/QG/QV): {self.max_tokens_extract}/{self.max_tokens_summarize}/{self.max_tokens_qa_gen}/{self.max_tokens_qa_verify}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Max Article Text Length: {self.max_article_text_len}", extra=extra_log_data) # MODIFIED

    @tenacity.retry(
        stop=stop_after_attempt(settings.LLM_API_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_llm_with_retry(self, prompt: str, temperature: float, max_tokens: int,
                                   trace_id: Optional[str] = None, comic_id: Optional[str] = None, **kwargs) -> str: # MODIFIED: Added comic_id
        """LLMService.generate_text를 재시도 로직과 함께 호출"""
        # Combine IDs for logging
        llm_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        logger.debug(f"Calling LLMService (Temp: {temperature}, MaxTokens: {max_tokens})...", extra=llm_log_data) # MODIFIED

        # Pass IDs if client supports them
        result = await self.llm_client.generate_text(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens,
            # trace_id=trace_id, comic_id=comic_id, # Pass if supported
            **kwargs
        )

        if "error" in result:
            error_msg = result['error']
            logger.error(f"LLMService call failed: {error_msg}", extra=llm_log_data) # MODIFIED
            raise RuntimeError(f"LLMService error: {error_msg}")
        elif "generated_text" not in result or not isinstance(result.get("generated_text"), str) or not result["generated_text"].strip():
            logger.error(f"LLMService returned invalid or empty text. Response: {result}", extra=llm_log_data) # MODIFIED
            raise ValueError("LLMService returned invalid or empty text")
        else:
            logger.debug(f"LLMService call successful.", extra=llm_log_data) # MODIFIED
            return result["generated_text"].strip()

    def _create_extraction_prompt_en(self, title: str, text: str) -> str:
        truncated_text = text[:self.max_article_text_len] + ("..." if len(text) > self.max_article_text_len else "")
        # [... existing prompt ...]
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

    async def _extract_key_points(self, article: Dict[str, Any], trace_id: Optional[str], comic_id: Optional[str]) -> List[str]: # MODIFIED: Added comic_id
        title = article.get('title', 'N/A')
        text = article.get('text', '')
        url = article.get('url', 'Unknown URL')
        kp_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': url} # MODIFIED
        if not text: return []

        prompt = self._create_extraction_prompt_en(title, text)
        try:
            # Pass IDs to LLM call
            response = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_extract,
                max_tokens=self.max_tokens_extract,
                trace_id=trace_id,
                comic_id=comic_id # Pass ID
            )
            lines = response.strip().split('\n')
            extracted_points = []
            for line in lines:
                match = re.match(r'^\s*\d+[\.\)]\s*(.+)', line.strip())
                if match:
                    point = match.group(1).strip()
                    if point: extracted_points.append(point)
            if not extracted_points and response.strip(): # Fallback
                 extracted_points = [line.strip() for line in response.strip().split('\n') if line.strip() and not line.strip().isnumeric()]

            logger.debug(f"Extracted {len(extracted_points)} key points.", extra=kp_log_data) # MODIFIED
            return extracted_points[:8]
        except RetryError as e:
            logger.error(f"Failed to extract key points after retries: {e}", extra=kp_log_data) # MODIFIED
            return []
        except Exception as e:
            logger.error(f"Error extracting key points: {e}", exc_info=True, extra=kp_log_data) # MODIFIED
            return []

    def _create_summary_prompt_en(self, title: str, key_points: List[str]) -> str:
        key_points_text = "\n".join(f"- {point}" for point in key_points)
        # [... existing prompt ...]
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

    async def _generate_summary(self, article: Dict[str, Any], key_points: List[str], trace_id: Optional[str], comic_id: Optional[str]) -> str: # MODIFIED: Added comic_id
        sum_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': article.get('url', 'Unknown URL')} # MODIFIED
        if not key_points: return ""
        title = article.get('title', 'N/A')

        prompt = self._create_summary_prompt_en(title, key_points)
        try:
            summary = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_summarize,
                max_tokens=self.max_tokens_summarize,
                trace_id=trace_id,
                comic_id=comic_id # Pass ID
            )
            logger.debug(f"Generated summary (length: {len(summary)}).", extra=sum_log_data) # MODIFIED
            return summary
        except RetryError as e:
            logger.error(f"Failed to generate summary after retries: {e}", extra=sum_log_data) # MODIFIED
            return ""
        except Exception as e:
            logger.error(f"Error generating summary: {e}", exc_info=True, extra=sum_log_data) # MODIFIED
            return ""

    def _create_qa_generation_prompt_en(self, summary: str) -> str:
        # [... existing prompt ...]
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

    async def _generate_qa_pairs(self, summary: str, trace_id: Optional[str], comic_id: Optional[str]) -> List[Dict[str, str]]: # MODIFIED: Added comic_id
        qa_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not summary: return []

        prompt = self._create_qa_generation_prompt_en(summary)
        try:
            response = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_qa_gen,
                max_tokens=self.max_tokens_qa_gen,
                trace_id=trace_id,
                comic_id=comic_id # Pass ID
            )
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
                           current_q = None
            logger.debug(f"Generated {len(qa_pairs)} QA pairs from summary.", extra=qa_log_data) # MODIFIED
            return qa_pairs[:3]
        except RetryError as e:
            logger.error(f"Failed to generate QA pairs after retries: {e}", extra=qa_log_data) # MODIFIED
            return []
        except Exception as e:
            logger.error(f"Error generating QA pairs: {e}", exc_info=True, extra=qa_log_data) # MODIFIED
            return []

    def _create_answer_verification_prompt_en(self, question: str, article_text: str) -> str:
        truncated_text = article_text[:self.max_article_text_len] + ("..." if len(article_text) > self.max_article_text_len else "")
        # [... existing prompt ...]
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

    async def _verify_answer_with_llm(self, question: str, expected_answer: str, article_text: str, trace_id: Optional[str], comic_id: Optional[str]) -> bool: # MODIFIED: Added comic_id
        """LLM을 사용하여 원문에서 답변 검증"""
        feqa_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'question': question[:50]} # MODIFIED
        if not article_text: return False

        prompt = self._create_answer_verification_prompt_en(question, article_text)
        try:
            llm_answer = await self._call_llm_with_retry(
                prompt=prompt,
                temperature=self.llm_temp_qa_verify,
                max_tokens=self.max_tokens_qa_verify,
                trace_id=trace_id,
                comic_id=comic_id # Pass ID
            )
            logger.debug(f"FEQA Verify - Expected: '{expected_answer[:50]}...' | LLM Found: '{llm_answer[:50]}...'", extra=feqa_log_data) # MODIFIED

            if "answer not found" in llm_answer.lower(): return False

            norm_expected = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', expected_answer.lower())).strip()
            norm_llm = re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', llm_answer.lower())).strip()
            return bool(norm_expected and norm_llm and norm_expected in norm_llm)
        except RetryError as e:
             logger.error(f"LLM answer verification failed after retries: {e}", extra=feqa_log_data) # MODIFIED
             return False
        except Exception as e:
            logger.error(f"Error during LLM answer verification: {e}", exc_info=True, extra=feqa_log_data) # MODIFIED
            return False

    async def _calculate_feqa_score(self, qa_pairs: List[Dict[str, str]], article_text: str, trace_id: Optional[str], comic_id: Optional[str]) -> float: # MODIFIED: Added comic_id
        """생성된 QA 쌍을 원문과 비교하여 FEQA 점수 계산"""
        feqa_calc_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not qa_pairs: return 0.0

        verification_tasks = []
        for pair in qa_pairs:
             question = pair.get("question")
             expected_answer = pair.get("answer")
             if question and expected_answer:
                  # Pass comic_id
                  verification_tasks.append(
                       self._verify_answer_with_llm(question, expected_answer, article_text, trace_id, comic_id)
                  )

        if not verification_tasks: return 0.0

        verified_count = 0
        verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
        for result in verification_results:
             if isinstance(result, bool) and result is True:
                  verified_count += 1
             elif isinstance(result, Exception):
                  logger.error(f"FEQA verification task failed with exception: {result}", extra=feqa_calc_log_data) # MODIFIED

        score = verified_count / len(verification_tasks) if verification_tasks else 0.0
        logger.info(f"FEQA Score: {score:.3f} ({verified_count} / {len(verification_tasks)} verified)", extra=feqa_calc_log_data) # MODIFIED
        return score

    async def _process_article(self, article: Dict[str, Any], trace_id: Optional[str], comic_id: Optional[str]) -> Optional[Dict[str, Any]]: # MODIFIED: Added comic_id
        """단일 기사 처리: 키포인트 추출 -> 요약 생성 -> FEQA 계산. Returns processed data or None"""
        original_url = article.get('url', article.get('original_url_from_source', 'Unknown URL'))
        article_text = article.get('text', '')
        # Combine IDs and URL for specific article processing logs
        process_log_data = {'trace_id': trace_id, 'comic_id': comic_id, 'url': original_url} # MODIFIED
        logger.info("Processing article...", extra=process_log_data) # MODIFIED

        if not article_text:
             logger.warning("Article text is empty. Skipping.", extra=process_log_data) # MODIFIED
             return None

        try:
            # Pass comic_id down the chain
            key_points = await self._extract_key_points(article, trace_id, comic_id)
            if not key_points:
                logger.warning("Failed to extract key points. Skipping article.", extra=process_log_data) # MODIFIED
                return None

            summary_text = await self._generate_summary(article, key_points, trace_id, comic_id)
            if not summary_text:
                logger.warning("Failed to generate summary. Skipping article.", extra=process_log_data) # MODIFIED
                return None

            qa_pairs = await self._generate_qa_pairs(summary_text, trace_id, comic_id)
            feqa_score = 0.0
            if not qa_pairs:
                 logger.warning("Failed to generate QA pairs for FEQA. Score set to 0.", extra=process_log_data) # MODIFIED
            else:
                 feqa_score = await self._calculate_feqa_score(qa_pairs, article_text, trace_id, comic_id)

            if feqa_score < self.feqa_threshold:
                 logger.warning(f"Summary failed FEQA check (Score: {feqa_score:.3f} < Threshold: {self.feqa_threshold}). Discarding.", extra=process_log_data) # MODIFIED
                 return None

            logger.info(f"Successfully processed article (FEQA: {feqa_score:.3f})", extra=process_log_data) # MODIFIED
            return {
                "original_url": original_url,
                "summary_text": summary_text,
                "feqa_score": round(feqa_score, 3)
                # "key_points": key_points, # Optionally include for debugging
                # "qa_pairs": qa_pairs     # Optionally include for debugging
            }
        except Exception as e:
            logger.exception("Unexpected error processing article.", extra=process_log_data) # MODIFIED use exception
            return None

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """뉴스 요약 및 FEQA 검증 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        # --- MODIFIED: Get trace_id and comic_id safely ---
        comic_id = getattr(state, 'comic_id', 'unknown_comic')
        trace_id = getattr(state, 'trace_id', comic_id)
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        # -------------------------------------------------

        node_class_name = self.__class__.__name__
        # --- ADDED: Start Logging ---
        logger.info(f"--- Executing {node_class_name} ---", extra=extra_log_data)
        logger.debug(f"Entering state:\n{summarize_for_logging(state, is_state_object=True)}", extra=extra_log_data)
        # --------------------------

        articles = getattr(state, 'articles', []) # Safe access
        config = getattr(state, 'config', {}) or {}

        # --- ADDED: Input Validation ---
        if not articles:
            error_message = "No articles found in state for summarization."
            logger.warning(error_message, extra=extra_log_data) # Warning is ok
            end_time = datetime.now(timezone.utc)
            node8_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "news_summaries": [],
                "node8_processing_stats": node8_processing_stats,
                "error_message": error_message
            }
            # --- ADDED: End Logging (Early Exit) ---
            logger.debug(f"Returning updates (no articles):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (No Articles) --- (Elapsed: {node8_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        # --------------------------

        articles_to_process = articles[:self.max_articles_to_summarize]
        logger.info(f"Starting summarization for {len(articles_to_process)} articles (Max: {self.max_articles_to_summarize}, Concurrency: {self.concurrency_limit})...", extra=extra_log_data)

        tasks = []
        for article in articles_to_process:
            async def task_with_semaphore(art):
                 async with semaphore:
                      # Pass comic_id
                      return await self._process_article(art, trace_id, comic_id)
            tasks.append(task_with_semaphore(article))

        # --- MODIFIED: Use return_exceptions=True ---
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # ------------------------------------------

        successful_summaries: List[Dict[str, Any]] = []
        failed_count = 0
        task_errors: List[str] = []

        # --- MODIFIED: Process results carefully ---
        for i, res in enumerate(results):
             original_url = articles_to_process[i].get('url', f'Article_{i}')
             if isinstance(res, dict) and res is not None:
                  successful_summaries.append(res)
             else:
                  failed_count += 1
                  if res is None:
                       # Failed/Filtered in _process_article (already logged)
                       # task_errors.append(f"Processing failed or filtered for {original_url}") # Optional: Add summary error
                       pass
                  elif isinstance(res, Exception):
                       # Exception from gather or outside _process_article's try/except
                       err_msg = f"Summarization task exception for {original_url}: {res}"
                       logger.error(err_msg, exc_info=res, extra=extra_log_data) # Log details
                       task_errors.append(f"Exception for {original_url}: {res}") # Summary error
                  else:
                       # Unexpected type
                       logger.warning(f"Unexpected result type for {original_url}: {type(res)}", extra=extra_log_data)
                       task_errors.append(f"Unknown error for {original_url}")
        # --------------------------------------------

        logger.info(f"News summarization complete. Generated {len(successful_summaries)} valid summaries. Failed/Skipped/Exception: {failed_count}.", extra=extra_log_data)

        end_time = datetime.now(timezone.utc)
        node8_processing_stats = (end_time - start_time).total_seconds()

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"Some errors occurred during news summarization task execution: {final_error_message}", extra=extra_log_data)

        update_data: Dict[str, Any] = {
            "news_summaries": successful_summaries,
            "node8_processing_stats": node8_processing_stats,
            "error_message": final_error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if final_error_message else logger.info
        log_level(f"News summarization result: {len(successful_summaries)} summaries generated. Errors: {final_error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node8_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}