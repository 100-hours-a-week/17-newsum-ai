# app/nodes/12_trend_analyzer_node.py (Improved Version)

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import os

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 가중치 등 기본 설정 참조
from app.tools.trends.google_trends import GoogleTrendsTool
from app.tools.trends.twitter_counts import TwitterCountsTool
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# 로거 설정
logger = get_logger(__name__)

class TrendAnalyzerNode:
    """
    검색 키워드에 대한 트렌드 점수를 분석합니다 (Google Trends, Twitter Counts).
    점수를 정규화하고 가중 평균하여 통합 점수를 계산합니다.
    - 가중치는 `settings`에서 로드합니다 (전역 설정으로 간주).
    """

    # 상태 입력/출력 정의
    inputs: List[str] = ["search_keywords", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["trend_scores", "processing_stats", "error_message"]

    # 트렌드 분석 도구 및 가중치 초기화
    def __init__(
        self,
        google_trends_tool: GoogleTrendsTool,
        twitter_counts_tool: TwitterCountsTool,
        # langsmith_service: Optional[LangSmithService] = None
    ):
        if not google_trends_tool: raise ValueError("GoogleTrendsTool is required.")
        if not twitter_counts_tool: raise ValueError("TwitterCountsTool is required.")
        self.google_trends = google_trends_tool
        self.twitter_counts = twitter_counts_tool
        # self.langsmith = langsmith_service

        # 가중치 로드 (settings에서 직접)
        # settings에 해당 값이 정의되어 있어야 함
        try:
             self.google_weight = float(settings.TREND_GOOGLE_WEIGHT)
             self.twitter_weight = float(settings.TREND_TWITTER_WEIGHT)
             if not (0 <= self.google_weight <= 1 and 0 <= self.twitter_weight <= 1):
                  raise ValueError("Weights must be between 0 and 1")
        except (AttributeError, ValueError, TypeError) as e:
             logger.error(f"Invalid trend weights in settings ({e}). Using default weights (0.6, 0.4).")
             self.google_weight = 0.6
             self.twitter_weight = 0.4

        # 가중치 정규화
        self._normalize_weights()
        logger.info("TrendAnalyzerNode initialized.")
        logger.info(f"Trend weights - Google: {self.google_weight:.2f}, Twitter: {self.twitter_weight:.2f}")

    def _normalize_weights(self):
        """가중치 합이 1이 되도록 정규화"""
        total_weight = self.google_weight + self.twitter_weight
        # 부동소수점 비교 시 np.isclose 사용 권장
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Initial trend weights ({self.google_weight:.2f}, {self.twitter_weight:.2f}) do not sum to 1. Normalizing.")
            if total_weight > 1e-9: # 0으로 나누기 방지 (더 작은 값 사용)
                self.google_weight /= total_weight
                self.twitter_weight /= total_weight
            else: # 둘 다 0이면 기본값으로 재설정
                logger.warning("Both trend weights are zero. Resetting to default (0.6, 0.4).")
                self.google_weight = 0.6
                self.twitter_weight = 0.4

    # --- 점수 조합 및 정규화 ---
    def _normalize_score(self, value: float, min_val: float, max_val: float, scale_max: float = 100.0) -> float:
        """값을 0-scale_max 범위로 선형 정규화"""
        if max_val <= min_val:
            # 모든 값이 같으면, 값이 0보다 크면 중간값, 아니면 0 반환
            return scale_max / 2.0 if value > 0 else 0.0
        normalized = ((value - min_val) / (max_val - min_val)) * scale_max
        return max(0.0, min(scale_max, normalized)) # 0~scale_max 범위 보장

    def _combine_scores(
            self,
            keywords: List[str],
            google_scores: Dict[str, float],
            twitter_scores: Dict[str, float],
            trace_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Google 및 Twitter 점수를 정규화하고 가중 평균하여 결합"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.info(f"{log_prefix} Combining and normalizing trend scores...")
        combined_results = []

        # --- Google 점수 정규화 (선형) ---
        g_vals = [v for v in google_scores.values() if isinstance(v, (int, float))]
        g_norm = {k: 0.0 for k in keywords}
        if g_vals:
            g_min, g_max = min(g_vals), max(g_vals)
            g_norm = {k: self._normalize_score(google_scores.get(k, 0.0), g_min, g_max) for k in keywords}
            logger.debug(f"{log_prefix} Google scores normalized (Min: {g_min}, Max: {g_max})")
        else:
            logger.debug(f"{log_prefix} No valid Google scores to normalize.")

        # --- Twitter 점수 정규화 (로그 변환 + 선형 정규화) ---
        t_vals = [v for v in twitter_scores.values() if isinstance(v, (int, float))]
        t_norm = {k: 0.0 for k in keywords}
        # 로그 변환은 양수 값에만 의미가 있음
        positive_t_scores = {k: v for k, v in twitter_scores.items() if isinstance(v, (int, float)) and v > 0}
        if positive_t_scores:
            # log1p(x) = log(1+x) -> 0을 포함한 값 처리 가능
            log_scores = {k: np.log1p(v) for k, v in positive_t_scores.items()}
            log_vals = list(log_scores.values())
            if log_vals: # 로그 변환된 값이 있는 경우
                t_min_log, t_max_log = min(log_vals), max(log_vals)
                # 로그 변환 후에도 모든 값이 같지 않은 경우에만 정규화
                if t_max_log > t_min_log:
                    for k in keywords:
                        # 원래 점수가 0 이하였으면 로그 점수 없음 -> 정규화 0
                        # 원래 점수가 양수였으면 해당 로그 점수로 정규화
                        t_norm[k] = self._normalize_score(log_scores.get(k, t_min_log), t_min_log, t_max_log) if k in log_scores else 0.0
                elif t_max_log > 0: # 모든 양수 값이 로그 변환 후 동일하면 (원래 값들이 같음) 중간값 부여
                    for k in keywords: t_norm[k] = 50.0 if k in log_scores else 0.0
                # else: 모든 로그 값이 0 (원래 값들이 0)이면 0 유지
                logger.debug(f"{log_prefix} Twitter scores log-normalized (MinLog: {t_min_log:.2f}, MaxLog: {t_max_log:.2f})")
            else: # 양수 값 없으면 모두 0
                 logger.debug(f"{log_prefix} No positive Twitter scores to normalize.")
        else:
            logger.debug(f"{log_prefix} No valid Twitter scores > 0 to normalize.")


        # --- 가중 평균 계산 ---
        for keyword in keywords:
            norm_g = g_norm.get(keyword, 0.0)
            norm_t = t_norm.get(keyword, 0.0)
            # 가중 평균
            final_score = (norm_g * self.google_weight) + (norm_t * self.twitter_weight)
            final_score = round(max(0.0, min(100.0, final_score)), 2) # 0-100 범위 및 반올림

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
        logger.info(f"{log_prefix} Score combination complete.")
        return combined_results


    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """트렌드 분석 워크플로우 실행"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing TrendAnalyzerNode...")

        search_keywords = state.search_keywords or []
        # config는 현재 사용하지 않지만 로드 (향후 확장 가능성)
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        if not search_keywords:
            logger.error(f"{log_prefix} No search keywords provided. Skipping trend analysis.")
            processing_stats['trend_analyzer_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"trend_scores": [], "processing_stats": processing_stats, "error_message": "Search keywords are missing for trend analysis."}

        logger.info(f"{log_prefix} Starting trend analysis for {len(search_keywords)} keywords...")
        error_message: Optional[str] = None
        trend_scores: List[Dict[str, Any]] = []

        try:
            # 도구 호출 (동시 실행)
            google_task = self.google_trends.get_trends(search_keywords, trace_id)
            twitter_task = self.twitter_counts.get_counts_for_keywords(search_keywords, trace_id)

            results = await asyncio.gather(google_task, twitter_task, return_exceptions=True)

            google_scores: Dict[str, float] = {}
            twitter_scores: Dict[str, float] = {}
            task_errors: List[str] = []

            # 결과 처리 (오류 발생 시 기본값 할당)
            if isinstance(results[0], dict): google_scores = results[0]
            elif isinstance(results[0], Exception):
                 msg = f"Google Trends analysis failed: {results[0]}"
                 logger.error(f"{log_prefix} {msg}", exc_info=results[0])
                 task_errors.append(msg)
                 google_scores = {kw: 0.0 for kw in search_keywords} # 실패 시 0점

            if isinstance(results[1], dict): twitter_scores = results[1]
            elif isinstance(results[1], Exception):
                 msg = f"Twitter Counts analysis failed: {results[1]}"
                 logger.error(f"{log_prefix} {msg}", exc_info=results[1])
                 task_errors.append(msg)
                 twitter_scores = {kw: 0.0 for kw in search_keywords} # 실패 시 0점

            # 점수 결합 및 정규화
            combined_scores = self._combine_scores(search_keywords, google_scores, twitter_scores, trace_id)

            # 결과 정렬 (점수 내림차순)
            trend_scores = sorted(combined_scores, key=lambda x: x.get('score', 0.0), reverse=True)

            error_message = "; ".join(task_errors) if task_errors else None
            if error_message: logger.warning(f"{log_prefix} Errors occurred during trend analysis: {error_message}")

        except Exception as e:
            logger.exception(f"{log_prefix} Unexpected error in TrendAnalyzerNode run: {e}")
            error_message = f"Unexpected error during trend analysis: {str(e)}"
            trend_scores = [] # 전체 실패 시 빈 리스트

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['trend_analyzer_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} TrendAnalyzerNode finished in {processing_stats['trend_analyzer_node_time']:.2f} seconds. Analyzed {len(trend_scores)} keywords.")

        # ComicState 업데이트
        update_data: Dict[str, Any] = {
            "trend_scores": trend_scores,
            "processing_stats": processing_stats,
            "error_message": error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}