# app/nodes/12_trend_analyzer_node.py (Refactored)

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import numpy as np

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.tools.trends.google_trends import GoogleTrendsTool
from app.tools.trends.twitter_counts import TwitterCountsTool
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

class TrendAnalyzerNode:
    """
    검색 키워드에 대한 트렌드 점수를 분석합니다 (Google Trends, Twitter Counts).
    [... existing docstring ...]
    """
    inputs: List[str] = ["search_keywords", "trace_id", "comic_id", "config"] # MODIFIED: Added comic_id
    outputs: List[str] = ["trend_scores", "node12_processing_stats", "error_message"]

    def __init__(
        self,
        google_trends_tool: GoogleTrendsTool,
        # --- MODIFIED: twitter_counts_tool을 Optional로 변경 ---
        twitter_counts_tool: Optional[TwitterCountsTool] = None,
    ):
        if not google_trends_tool:
            raise ValueError("GoogleTrendsTool is required.")
        # --- MODIFIED: twitter_counts_tool이 없어도 오류 발생 안 함 ---
        # if not twitter_counts_tool: raise ValueError("TwitterCountsTool is required.")

        self.google_trends = google_trends_tool
        self.twitter_counts = twitter_counts_tool

        try:
             self.google_weight = float(settings.TREND_GOOGLE_WEIGHT)
             self.twitter_weight = float(settings.TREND_TWITTER_WEIGHT)
             if not (0 <= self.google_weight <= 1 and 0 <= self.twitter_weight <= 1):
                  raise ValueError("Weights must be between 0 and 1")
        except (AttributeError, ValueError, TypeError) as e:
             logger.error(f"Invalid trend weights in settings ({e}). Using default weights (0.6, 0.4).")
             self.google_weight = 0.6
             self.twitter_weight = 0.4

        self._normalize_weights()
        logger.info("TrendAnalyzerNode initialized.")
        logger.info(f"Trend weights - Google: {self.google_weight:.2f}, Twitter: {self.twitter_weight:.2f}")

    def _normalize_weights(self):
        total_weight = self.google_weight + self.twitter_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Initial trend weights ({self.google_weight:.2f}, {self.twitter_weight:.2f}) do not sum to 1. Normalizing.")
            if total_weight > 1e-9:
                self.google_weight /= total_weight
                self.twitter_weight /= total_weight
            else:
                logger.warning("Both trend weights are zero. Resetting to default (0.6, 0.4).")
                self.google_weight = 0.6
                self.twitter_weight = 0.4

    def _normalize_score(self, value: float, min_val: float, max_val: float, scale_max: float = 100.0) -> float:
        if max_val <= min_val:
             # If min/max are equal, return mid-point if value > 0, else 0
             return scale_max / 2.0 if value > 0 else 0.0
        normalized = ((value - min_val) / (max_val - min_val)) * scale_max
        # Clamp the value between 0 and scale_max
        return round(max(0.0, min(scale_max, normalized)), 2) # Added rounding

    def _combine_scores(
            self,
            keywords: List[str],
            google_scores: Dict[str, float],
            twitter_scores: Dict[str, float],
            trace_id: Optional[str],
            comic_id: Optional[str] # MODIFIED: Added comic_id
    ) -> List[Dict[str, Any]]:
        """Google 및 Twitter 점수를 정규화하고 가중 평균하여 결합"""
        combine_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        logger.info("Combining and normalizing trend scores...", extra=combine_log_data) # MODIFIED
        combined_results = []

        g_norm = {k: 0.0 for k in keywords}
        g_vals = [v for v in google_scores.values() if isinstance(v, (int, float))]
        if g_vals:
            g_min, g_max = min(g_vals), max(g_vals)
            g_norm = {k: self._normalize_score(google_scores.get(k, 0.0), g_min, g_max)
                      for k in keywords if isinstance(google_scores.get(k), (int, float))} # Ensure key exists and is number
            logger.debug(f"Google scores normalized (Min: {g_min}, Max: {g_max})", extra=combine_log_data) # MODIFIED
        else:
            logger.debug("No valid Google scores to normalize.", extra=combine_log_data) # MODIFIED

        t_norm = {k: 0.0 for k in keywords}
        positive_t_scores = {k: v for k, v in twitter_scores.items() if isinstance(v, (int, float)) and v > 0}
        if positive_t_scores:
            log_scores = {k: np.log1p(v) for k, v in positive_t_scores.items()}
            log_vals = list(log_scores.values())
            if log_vals:
                t_min_log, t_max_log = min(log_vals), max(log_vals)
                if not np.isclose(t_max_log, t_min_log): # Use isclose for float comparison
                    for k in keywords:
                        t_norm[k] = self._normalize_score(log_scores.get(k, t_min_log), t_min_log, t_max_log) if k in log_scores else 0.0
                elif t_max_log > 0: # All positive scores were the same
                     for k in keywords: t_norm[k] = 50.0 if k in log_scores else 0.0
                logger.debug(f"Twitter scores log-normalized (MinLog: {t_min_log:.2f}, MaxLog: {t_max_log:.2f})", extra=combine_log_data) # MODIFIED
            else:
                 logger.debug("No positive Twitter scores after log transform.", extra=combine_log_data) # MODIFIED
        else:
            logger.debug("No valid Twitter scores > 0 to normalize.", extra=combine_log_data) # MODIFIED

        for keyword in keywords:
            norm_g = g_norm.get(keyword, 0.0)
            norm_t = t_norm.get(keyword, 0.0)
            final_score = (norm_g * self.google_weight) + (norm_t * self.twitter_weight)
            final_score = round(max(0.0, min(100.0, final_score)), 2)

            combined_results.append({
                'keyword': keyword,
                'score': final_score,
                'source': 'combined',
                'details': {
                    'google_raw': round(google_scores.get(keyword, 0.0), 2),
                    'google_norm': round(norm_g, 2),
                    'twitter_raw': twitter_scores.get(keyword, 0.0),
                    'twitter_norm': round(norm_t, 2),
                    'weights': {'google': round(self.google_weight, 3), 'twitter': round(self.twitter_weight, 3)}
                }
            })
        logger.info("Score combination complete.", extra=combine_log_data) # MODIFIED
        return combined_results

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """트렌드 분석 워크플로우 실행"""
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

        search_keywords = getattr(state, 'search_keywords', []) # Safe access
        config = getattr(state, 'config', {}) or {}

        # --- ADDED: Input Validation ---
        if not search_keywords:
            error_message = "Search keywords are missing for trend analysis."
            logger.error(error_message, extra=extra_log_data)
            end_time = datetime.now(timezone.utc)
            node12_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "trend_scores": [],
                "node12_processing_stats": node12_processing_stats,
                "error_message": error_message
            }
            # --- ADDED: End Logging (Error Case) ---
            logger.debug(f"Returning updates (no keywords):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Error: Missing Keywords) --- (Elapsed: {node12_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        logger.info(f"Starting trend analysis for {len(search_keywords)} keywords...", extra=extra_log_data)
        error_message: Optional[str] = None
        trend_scores: List[Dict[str, Any]] = []
        task_errors: List[str] = [] # Collect specific tool errors

        try:
            # Pass trace_id and comic_id if tools support them
            google_task = self.google_trends.get_trends(search_keywords, trace_id) # Pass trace_id
            twitter_task = self.twitter_counts.get_counts_for_keywords(search_keywords, trace_id) # Pass trace_id

            # MODIFIED: Use return_exceptions=True
            results = await asyncio.gather(google_task, twitter_task, return_exceptions=True)

            google_scores: Dict[str, float] = {}
            twitter_scores: Dict[str, float] = {}

            if isinstance(results[0], dict):
                 google_scores = results[0]
            elif isinstance(results[0], Exception):
                 msg = f"Google Trends analysis failed: {results[0]}"
                 logger.error(msg, exc_info=results[0], extra=extra_log_data) # Log exception details
                 task_errors.append(f"Google Trends Failed: {results[0]}")
                 google_scores = {kw: 0.0 for kw in search_keywords}
            else: # Unexpected result type
                 logger.error(f"Unexpected result type from Google Trends: {type(results[0])}", extra=extra_log_data)
                 task_errors.append(f"Google Trends Unexpected Result Type: {type(results[0])}")
                 google_scores = {kw: 0.0 for kw in search_keywords}

            if isinstance(results[1], dict):
                 twitter_scores = results[1]
            elif isinstance(results[1], Exception):
                 msg = f"Twitter Counts analysis failed: {results[1]}"
                 logger.error(msg, exc_info=results[1], extra=extra_log_data) # Log exception details
                 task_errors.append(f"Twitter Counts Failed: {results[1]}")
                 twitter_scores = {kw: 0.0 for kw in search_keywords}
            else: # Unexpected result type
                 logger.error(f"Unexpected result type from Twitter Counts: {type(results[1])}", extra=extra_log_data)
                 task_errors.append(f"Twitter Counts Unexpected Result Type: {type(results[1])}")
                 twitter_scores = {kw: 0.0 for kw in search_keywords}

            # Combine scores (pass IDs for logging)
            combined_scores = self._combine_scores(
                search_keywords, google_scores, twitter_scores, trace_id, comic_id
            )
            trend_scores = sorted(combined_scores, key=lambda x: x.get('score', 0.0), reverse=True)

            final_error_message = "; ".join(task_errors) if task_errors else None
            if final_error_message:
                 logger.warning(f"Errors occurred during trend tool execution: {final_error_message}", extra=extra_log_data)
                 error_message = final_error_message # Set node error message

        except Exception as e:
            logger.exception("Unexpected error in TrendAnalyzerNode run.", extra=extra_log_data) # Use exception
            error_message = f"Unexpected error during trend analysis: {str(e)}"
            trend_scores = []

        end_time = datetime.now(timezone.utc)
        node12_processing_stats = (end_time - start_time).total_seconds()

        update_data: Dict[str, Any] = {
            "trend_scores": trend_scores,
            "node12_processing_stats": node12_processing_stats,
            "error_message": error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if error_message else logger.info
        log_level(f"Trend analysis result: {len(trend_scores)} scores calculated. Errors: {error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node12_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}