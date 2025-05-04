# app/nodes/11_evaluate_summary_node.py (Refactored)

import asyncio
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import os

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

ROUGE_SCORE_AVAILABLE = False
try:
    from rouge_score import rouge_scorer
    ROUGE_SCORE_AVAILABLE = True
    logger.info("rouge-score library found.")
except ImportError:
    logger.warning("rouge-score library not installed. ROUGE evaluation will return 0.0.")

BERTSCORE_AVAILABLE = False
try:
    from bert_score import score as bert_score_calculate
    BERTSCORE_AVAILABLE = True
    logger.info("bert-score library found.")
except ImportError:
    logger.warning("bert-score library (or its dependencies) not installed. BERTScore evaluation will return 0.0.")
except Exception as e:
    logger.error(f"Error loading bert-score library: {e}. BERTScore evaluation disabled.", exc_info=True)
    BERTSCORE_AVAILABLE = False

class EvaluateSummaryNode:
    """
    자동 측정 지표(ROUGE-L, BERTScore, Topic Coverage)를 사용하여 최종 요약의 품질을 평가하고,
    결과에 따라 다음 단계를 결정합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["final_summary", "topic_analysis", "articles", "news_summaries", "trace_id", "comic_id", "config"] # MODIFIED: Added comic_id
    outputs: List[str] = ["decision", "evaluation_metrics", "node11_processing_stats", "error_message"]

    def __init__(self):
        self.rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if ROUGE_SCORE_AVAILABLE else None
        logger.info("EvaluateSummaryNode initialized.")
        if not ROUGE_SCORE_AVAILABLE: logger.warning("ROUGE scorer disabled.")
        if not BERTSCORE_AVAILABLE: logger.warning("BERTScore calculator disabled.")

    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict): # MODIFIED: Added extra_log_data
        # Use .get with defaults from settings for robustness
        self.eval_thresholds = config.get('evaluation_thresholds', settings.DEFAULT_EVALUATION_THRESHOLDS)
        self.decision_thresholds = config.get('decision_logic_thresholds', settings.DEFAULT_DECISION_THRESHOLDS)
        self.bertscore_lang = config.get('bertscore_lang', settings.DEFAULT_BERTSCORE_LANG)
        self.feqa_threshold_for_ref = float(config.get("feqa_threshold", settings.DEFAULT_FEQA_THRESHOLD))
        self.max_summaries_for_ref = int(config.get("max_news_summaries_for_synthesis", settings.DEFAULT_MAX_SUMMARIES_SYNTHESIS))

        # Add checks for threshold dictionaries
        if not isinstance(self.eval_thresholds, dict):
            logger.warning("evaluation_thresholds is not a dict. Using default.", extra=extra_log_data) # MODIFIED
            self.eval_thresholds = settings.DEFAULT_EVALUATION_THRESHOLDS
        if not isinstance(self.decision_thresholds, dict):
            logger.warning("decision_logic_thresholds is not a dict. Using default.", extra=extra_log_data) # MODIFIED
            self.decision_thresholds = settings.DEFAULT_DECISION_THRESHOLDS

        logger.debug(f"Runtime config loaded. Eval Thresh: {self.eval_thresholds}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Decision Thresh: {self.decision_thresholds}", extra=extra_log_data) # MODIFIED
        logger.debug(f"BERTScore Lang: {self.bertscore_lang}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Ref Text Params - FEQA Thresh: {self.feqa_threshold_for_ref}, Max Summaries: {self.max_summaries_for_ref}", extra=extra_log_data) # MODIFIED

    def _prepare_reference_text(self, articles: List[Dict[str, Any]], news_summaries: List[Dict[str, Any]], trace_id: Optional[str], comic_id: Optional[str]) -> str: # MODIFIED: Added comic_id
        """ROUGE 및 BERTScore 계산을 위한 참조 텍스트 준비"""
        ref_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        reference_texts = []

        if news_summaries:
             # Ensure summaries are dicts with text and float scores
             valid_summaries = [
                 s for s in news_summaries
                 if isinstance(s, dict) and isinstance(s.get("summary_text"), str) and s["summary_text"].strip() and isinstance(s.get("feqa_score"), (int, float))
             ]

             high_quality_summaries = sorted(
                [s for s in valid_summaries if s.get("feqa_score", 0.0) >= self.feqa_threshold_for_ref],
                key=lambda x: x.get("feqa_score", 0.0), reverse=True
             )[:self.max_summaries_for_ref]

             selected_texts = [s['summary_text'] for s in high_quality_summaries]

             if selected_texts:
                 reference_texts = selected_texts
                 logger.debug(f"Using {len(reference_texts)} high-quality news summaries as reference.", extra=ref_log_data) # MODIFIED
             elif valid_summaries: # Fallback
                 all_valid_sorted = sorted(valid_summaries, key=lambda x: x.get("feqa_score", 0.0), reverse=True)
                 selected_texts = [s['summary_text'] for s in all_valid_sorted[:self.max_summaries_for_ref]]
                 if selected_texts:
                      reference_texts = selected_texts
                      logger.debug(f"No high-quality summaries found. Using {len(reference_texts)} top summaries as reference.", extra=ref_log_data) # MODIFIED

        if not reference_texts and articles:
            # Ensure articles are dicts with titles
            valid_titles = [a.get("title", "") for a in articles if isinstance(a, dict) and isinstance(a.get("title"), str) and a["title"].strip()]
            if valid_titles:
                 reference_texts = valid_titles[:5]
                 logger.debug(f"No usable summaries. Using {len(reference_texts)} article titles as reference.", extra=ref_log_data) # MODIFIED

        if not reference_texts:
            logger.warning("Could not prepare reference text. Evaluation scores might be 0.", extra=ref_log_data) # MODIFIED
            return ""

        return "\n\n".join(reference_texts).strip()

    def _calculate_rouge_l(self, summary: str, reference_text: str, trace_id: Optional[str], comic_id: Optional[str]) -> float: # MODIFIED: Added comic_id
        """ROUGE-L F1 점수 계산"""
        rouge_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not ROUGE_SCORE_AVAILABLE or not self.rouge_l_scorer:
            logger.warning("ROUGE scorer unavailable.", extra=rouge_log_data) # MODIFIED
            return 0.0
        if not summary or not reference_text:
             logger.warning("Cannot calculate ROUGE-L: missing summary or reference text.", extra=rouge_log_data) # MODIFIED
             return 0.0
        try:
            scores = self.rouge_l_scorer.score(reference_text, summary)
            f1_score = scores['rougeL'].fmeasure
            logger.debug(f"ROUGE-L F1 calculated: {f1_score:.4f}", extra=rouge_log_data) # MODIFIED
            return round(f1_score, 4)
        except Exception as e:
            logger.exception("Error calculating ROUGE-L score.", extra=rouge_log_data) # MODIFIED use exception
            return 0.0

    def _calculate_bert_score(self, summary: str, reference_text: str, trace_id: Optional[str], comic_id: Optional[str]) -> float: # MODIFIED: Added comic_id
        """BERTScore F1 점수 계산"""
        bert_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not BERTSCORE_AVAILABLE:
            logger.warning("BERTScore library unavailable.", extra=bert_log_data) # MODIFIED
            return 0.0
        if not summary or not reference_text:
            logger.warning("Cannot calculate BERTScore: missing summary or reference text.", extra=bert_log_data) # MODIFIED
            return 0.0
        try:
            # Ensure calculation happens in a way that doesn't block event loop if very heavy
            # Consider ProcessPoolExecutor if bert_score is synchronous and slow
            precision, recall, f1 = bert_score_calculate(
                [summary], [reference_text], lang=self.bertscore_lang, rescale_with_baseline=True, verbose=False
            )
            f1_score = f1.mean().item()
            logger.debug(f"BERTScore F1 (lang={self.bertscore_lang}) calculated: {f1_score:.4f}", extra=bert_log_data) # MODIFIED
            return round(f1_score, 4)
        except Exception as e:
            logger.exception("Error calculating BERTScore.", extra=bert_log_data) # MODIFIED use exception
            return 0.0

    def _calculate_topic_coverage(self, summary: str, topic_analysis: Dict[str, Any], trace_id: Optional[str], comic_id: Optional[str]) -> float: # MODIFIED: Added comic_id
        """요약의 토픽 커버리지 계산"""
        cov_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not summary or not topic_analysis or not isinstance(topic_analysis, dict): # Added type check
            logger.warning("Cannot calculate topic coverage: missing summary or invalid topic_analysis.", extra=cov_log_data) # MODIFIED
            return 0.0

        summary_lower = summary.lower()
        all_terms: Set[str] = set()

        main_topic = topic_analysis.get("main_topic", "")
        if isinstance(main_topic, str) and main_topic.strip():
            all_terms.add(main_topic.strip().lower())

        entities = topic_analysis.get("entities", [])
        if isinstance(entities, list):
            for entity in entities:
                name = entity.get("name") if isinstance(entity, dict) else None
                if isinstance(name, str) and name.strip():
                    all_terms.add(name.strip().lower())

        keywords_analyzed = topic_analysis.get("keywords_analyzed", []) # Use correct key
        if isinstance(keywords_analyzed, list):
            for kw_item in keywords_analyzed:
                 keyword = kw_item.get("keyword") if isinstance(kw_item, dict) else None
                 if isinstance(keyword, str) and keyword.strip():
                      all_terms.add(keyword.strip().lower())

        if not all_terms:
            logger.warning("No valid terms found in topic_analysis for coverage check.", extra=cov_log_data) # MODIFIED
            return 0.0

        covered_count = 0
        covered_terms_list = []
        for term in all_terms:
            try: # Add try-except for regex errors with complex terms
                 if re.search(r'\b' + re.escape(term) + r'\b', summary_lower):
                      covered_count += 1
                      covered_terms_list.append(term)
            except re.error as re_err:
                 logger.warning(f"Regex error checking term '{term}': {re_err}", extra=cov_log_data) # MODIFIED

        coverage_ratio = covered_count / len(all_terms) if all_terms else 0.0
        logger.debug(f"Topic Coverage: {coverage_ratio:.4f} ({covered_count}/{len(all_terms)} terms covered). Terms: {list(all_terms)}", extra=cov_log_data) # MODIFIED
        logger.debug(f"Covered terms: {covered_terms_list}", extra=cov_log_data) # MODIFIED
        return round(coverage_ratio, 4)

    def _make_decision(self, scores: Dict[str, float], trace_id: Optional[str], comic_id: Optional[str]) -> str: # MODIFIED: Added comic_id
        """점수 및 설정된 임계값 기반 의사결정"""
        decision_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        rouge = scores.get('rouge_l', 0.0)
        bert = scores.get('bert_score', 0.0)
        coverage = scores.get('topic_coverage', 0.0)

        # Safely get thresholds from the potentially invalid dicts
        thresh_rouge = self.eval_thresholds.get('rouge_l', 0.3) if isinstance(self.eval_thresholds, dict) else 0.3
        thresh_bert = self.eval_thresholds.get('bert_score', 0.7) if isinstance(self.eval_thresholds, dict) else 0.7
        thresh_coverage = self.eval_thresholds.get('topic_coverage', 0.6) if isinstance(self.eval_thresholds, dict) else 0.6

        decision_vr = self.decision_thresholds.get('very_low_rouge', 0.1) if isinstance(self.decision_thresholds, dict) else 0.1
        decision_vb = self.decision_thresholds.get('very_low_bertscore', 0.5) if isinstance(self.decision_thresholds, dict) else 0.5
        decision_vc = self.decision_thresholds.get('very_low_coverage', 0.3) if isinstance(self.decision_thresholds, dict) else 0.3
        decision_lchm = self.decision_thresholds.get('low_coverage_high_metrics', 0.7) if isinstance(self.decision_thresholds, dict) else 0.7

        log_msg = f"Scores R={rouge:.3f}(T={thresh_rouge:.2f}), B={bert:.3f}(T={thresh_bert:.2f}), C={coverage:.3f}(T={thresh_coverage:.2f})."

        if rouge >= thresh_rouge and bert >= thresh_bert and coverage >= thresh_coverage:
            logger.info(f"Decision: proceed (All metrics meet thresholds). {log_msg}", extra=decision_log_data) # MODIFIED
            return "proceed"
        if rouge < decision_vr or bert < decision_vb or coverage < decision_vc:
            logger.warning(f"Decision: research_again (One or more metrics very low). {log_msg}", extra=decision_log_data) # MODIFIED
            return "research_again"
        if coverage < thresh_coverage and rouge >= decision_lchm and bert >= decision_lchm:
            logger.warning(f"Decision: refine_topic (Coverage low, R/B acceptable). {log_msg}", extra=decision_log_data) # MODIFIED
            return "refine_topic"

        logger.info(f"Decision: proceed (Default - below threshold but not triggering retry/refine). {log_msg}", extra=decision_log_data) # MODIFIED
        return "proceed"

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """요약 평가 프로세스 실행"""
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

        final_summary = getattr(state, 'final_summary', None) # Safe access
        topic_analysis = getattr(state, 'topic_analysis', {}) or {}
        articles = getattr(state, 'articles', []) or []
        news_summaries = getattr(state, 'news_summaries', []) or []
        config = getattr(state, 'config', {}) or {}

        # --- ADDED: Input Validation ---
        if not final_summary or not isinstance(final_summary, str) or not final_summary.strip():
            error_message = "Final summary is missing or empty. Evaluation cannot proceed."
            logger.error(error_message, extra=extra_log_data)
            metrics = {"rouge_l": 0.0, "bert_score": 0.0, "topic_coverage": 0.0}
            decision = "research_again" # Decide to retry if summary failed
            end_time = datetime.now(timezone.utc)
            node11_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "decision": decision,
                "evaluation_metrics": metrics,
                "node11_processing_stats": node11_processing_stats,
                "error_message": error_message
            }
            # --- ADDED: End Logging (Error Case) ---
            logger.debug(f"Returning updates (no summary):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Error: Missing Summary) --- (Elapsed: {node11_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        # --------------------------

        logger.info("Starting final summary evaluation...", extra=extra_log_data)
        evaluation_metrics = {"rouge_l": 0.0, "bert_score": 0.0, "topic_coverage": 0.0}
        error_message = None

        try:
            # --- Reference Text Preparation ---
            # Pass IDs
            reference_text = self._prepare_reference_text(articles, news_summaries, trace_id, comic_id)

            # --- Metric Calculation ---
            # Run potentially slow calculations sequentially for simplicity here
            # Consider asyncio.to_thread or ProcessPoolExecutor for CPU/GPU bound tasks if performance is critical
            # Pass IDs
            rouge_l_score = self._calculate_rouge_l(final_summary, reference_text, trace_id, comic_id)
            bert_score = self._calculate_bert_score(final_summary, reference_text, trace_id, comic_id)
            topic_coverage_score = self._calculate_topic_coverage(final_summary, topic_analysis, trace_id, comic_id)

            evaluation_metrics = {
                "rouge_l": rouge_l_score,
                "bert_score": bert_score,
                "topic_coverage": topic_coverage_score
            }
            logger.info(f"Calculated Metrics: {evaluation_metrics}", extra=extra_log_data)

        except Exception as e:
            logger.exception("Error during metric calculation.", extra=extra_log_data) # Use exception
            error_message = f"Evaluation metric calculation failed: {e}"
            # Keep default 0.0 scores

        # --- Decision Making ---
        # Pass IDs
        decision = self._make_decision(evaluation_metrics, trace_id, comic_id)

        end_time = datetime.now(timezone.utc)
        node11_processing_stats = (end_time - start_time).total_seconds()

        update_data: Dict[str, Any] = {
            "decision": decision,
            "evaluation_metrics": evaluation_metrics,
            "node11_processing_stats": node11_processing_stats,
            "error_message": error_message # Will be None if calculation succeeded
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if decision != "proceed" else logger.info
        log_level(f"Evaluation result: Decision='{decision}'. Errors: {error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node11_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}