# app/nodes/12_trend_analyzer_node.py

import asyncio
# --- datetime, timezone 임포트 확인 ---
from datetime import datetime, timezone
# ------------------------------------
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import os # settings 로딩에 필요할 수 있으나 여기서는 config 사용

# --- 프로젝트 구성 요소 임포트 ---
from app.config.settings import settings # 가중치 등 직접 참조 시 필요
# 도구 클래스 임포트
from app.tools.trends.google_trends import GoogleTrendsTool
from app.tools.trends.twitter_counts import TwitterCountsTool
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# --- 로거 설정 ---
logger = get_logger("TrendAnalyzerNode")

class TrendAnalyzerNode:
    """
    (Refactored) 검색 키워드에 대한 트렌드 점수를 분석합니다.
    - GoogleTrendsTool과 TwitterCountsTool 사용.
    - 점수 정규화 및 가중치 기반 통합.
    """

    # 상태 입력/출력 정의
    inputs: List[str] = ["search_keywords", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["trend_scores", "processing_stats", "error_message"]

    # 트렌드 분석 도구를 외부에서 주입
    def __init__(
        self,
        google_trends_tool: GoogleTrendsTool,
        twitter_counts_tool: TwitterCountsTool,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        self.google_trends = google_trends_tool
        self.twitter_counts = twitter_counts_tool
        # self.langsmith = langsmith_service
        # 가중치는 settings에서 직접 로드 (전역 설정으로 간주)
        self.google_weight = settings.TREND_GOOGLE_WEIGHT
        self.twitter_weight = settings.TREND_TWITTER_WEIGHT
        # 가중치 정규화 (합이 1이 되도록)
        self._normalize_weights()
        logger.info("TrendAnalyzerNode initialized with GoogleTrendsTool and TwitterCountsTool.")
        logger.info(f"Trend weights - Google: {self.google_weight:.2f}, Twitter: {self.twitter_weight:.2f}")

    def _normalize_weights(self):
        """가중치 합이 1이 되도록 정규화"""
        total_weight = self.google_weight + self.twitter_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Initial trend weights ({self.google_weight}, {self.twitter_weight}) do not sum to 1. Normalizing.")
            if total_weight > 1e-6: # 0으로 나누기 방지
                self.google_weight /= total_weight
                self.twitter_weight /= total_weight
            else: # 둘 다 0이면 기본값 사용
                self.google_weight = 0.6
                self.twitter_weight = 0.4

    # --- 점수 조합 및 정규화 ---
    def _normalize_score(self, value: float, min_val: float, max_val: float, scale_max: float = 100.0) -> float:
        """값을 0-scale_max 범위로 정규화"""
        if max_val <= min_val: return scale_max / 2.0 if value > min_val else 0.0 # 모든 값이 같거나 범위 오류
        normalized = ((value - min_val) / (max_val - min_val)) * scale_max
        return max(0.0, min(scale_max, normalized)) # 범위 제한

    def _combine_scores(
            self,
            keywords: List[str],
            google_scores: Dict[str, float],
            twitter_scores: Dict[str, float],
            trace_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Google 및 Twitter 점수를 결합하고 정규화"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.info(f"{log_prefix} Combining and normalizing trend scores...")
        combined_results = []

        # Google 점수 정규화 (선형)
        g_vals = [v for v in google_scores.values() if isinstance(v, (int, float))] # 유효한 숫자만
        g_norm = {k: 0.0 for k in keywords}
        if g_vals:
            g_min, g_max = min(g_vals), max(g_vals)
            if g_max > g_min: # 값이 다른 경우에만 정규화
                 g_norm = {k: self._normalize_score(google_scores.get(k, 0.0), g_min, g_max) for k in keywords}
            elif g_max > 0: # 모든 값이 0 이상이고 같으면 중간값(50) 부여
                 g_norm = {k: 50.0 if google_scores.get(k, 0.0) > 0 else 0.0 for k in keywords}
            # else: 모든 값이 0이면 0 유지
            logger.debug(f"{log_prefix} Google scores normalized (Min: {g_min}, Max: {g_max})")
        else: logger.debug(f"{log_prefix} No valid Google scores to normalize.")

        # Twitter 점수 정규화 (로그 스케일 + 선형)
        t_vals = [v for v in twitter_scores.values() if isinstance(v, (int, float))]
        t_norm = {k: 0.0 for k in keywords}
        if t_vals and any(v > 0 for v in t_vals): # 0보다 큰 값이 하나라도 있을 때
            # 로그 변환 (log1p 사용: 0 처리)
            log_scores = {k: np.log1p(twitter_scores.get(k, 0.0)) for k in keywords}
            log_vals = list(log_scores.values())
            t_min_log, t_max_log = min(log_vals), max(log_vals)
            if t_max_log > t_min_log: # 로그 변환 후 값이 다를 때 정규화
                 t_norm = {k: self._normalize_score(log_scores[k], t_min_log, t_max_log) for k in keywords}
            elif t_max_log > 0: # 로그 변환 후 값이 0 이상이고 같으면 중간값(50) 부여
                 t_norm = {k: 50.0 if log_scores.get(k, 0.0) > 0 else 0.0 for k in keywords}
            # else: 모든 로그 값이 0이면 0 유지
            logger.debug(f"{log_prefix} Twitter scores log-normalized (MinLog: {t_min_log:.2f}, MaxLog: {t_max_log:.2f})")
        else: logger.debug(f"{log_prefix} No valid Twitter scores > 0 to normalize.")


        # 가중 평균 계산
        for keyword in keywords:
            norm_g = g_norm.get(keyword, 0.0)
            norm_t = t_norm.get(keyword, 0.0)
            final_score = (norm_g * self.google_weight) + (norm_t * self.twitter_weight)
            final_score = round(max(0.0, min(100.0, final_score)), 2) # 0-100 범위 및 반올림

            combined_results.append({
                'keyword': keyword,
                'score': final_score,
                'source': 'combined',
                'details': { # 상세 정보 포함
                    'google_raw': round(google_scores.get(keyword, 0.0), 2),
                    'google_norm': round(norm_g, 2),
                    'twitter_raw': twitter_scores.get(keyword, 0.0),
                    'twitter_norm': round(norm_t, 2),
                    'weights': {'google': self.google_weight, 'twitter': self.twitter_weight}
                }
            })
        logger.info(f"{log_prefix} Score combination complete.")
        return combined_results


    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """트렌드 분석 워크플로우 실행"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing TrendAnalyzerNode...")

        # 상태 및 설정 로드
        search_keywords = state.search_keywords or []
        config = state.config or {} # 현재 config는 사용하지 않지만 로드
        processing_stats = state.processing_stats or {}

        # 입력 유효성 검사
        if not search_keywords:
            logger.error(f"{log_prefix} No search keywords provided. Skipping trend analysis.")
            processing_stats['trend_analyzer_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"trend_scores": [], "processing_stats": processing_stats, "error_message": "Search keywords are missing for trend analysis."}

        logger.info(f"{log_prefix} Starting trend analysis for {len(search_keywords)} keywords...")
        error_message: Optional[str] = None # 오류 메시지 저장용

        try:
            # --- 각 소스에서 트렌드 분석 (도구 호출, 동시 실행) ---
            google_task = self.google_trends.get_trends(search_keywords, state.trace_id)
            twitter_task = self.twitter_counts.get_counts_for_keywords(search_keywords, state.trace_id)

            # 결과 기다리기
            results = await asyncio.gather(google_task, twitter_task, return_exceptions=True)

            # 결과 처리 및 오류 핸들링
            google_scores: Dict[str, float] = {}
            twitter_scores: Dict[str, float] = {}
            task_errors = []

            if isinstance(results[0], dict): google_scores = results[0]
            elif isinstance(results[0], Exception):
                 msg = f"Google Trends analysis failed: {results[0]}"
                 logger.error(f"{log_prefix} {msg}")
                 task_errors.append(msg)
                 google_scores = {kw: 0.0 for kw in search_keywords} # 오류 시 0점 처리

            if isinstance(results[1], dict): twitter_scores = results[1]
            elif isinstance(results[1], Exception):
                 msg = f"Twitter Counts analysis failed: {results[1]}"
                 logger.error(f"{log_prefix} {msg}")
                 task_errors.append(msg)
                 twitter_scores = {kw: 0.0 for kw in search_keywords} # 오류 시 0점 처리

            # --- 점수 결합 및 정규화 ---
            combined_scores = self._combine_scores(search_keywords, google_scores, twitter_scores, state.trace_id)

            # --- 결과 정렬 (점수 내림차순) ---
            trend_scores = sorted(combined_scores, key=lambda x: x['score'], reverse=True)

            # 최종 오류 메시지 설정
            error_message = "; ".join(task_errors) if task_errors else None
            if error_message: logger.warning(f"{log_prefix} Errors occurred during trend analysis: {error_message}")

        except Exception as e:
            # gather 외의 예외 처리 (예: _combine_scores)
            logger.exception(f"{log_prefix} Unexpected error in TrendAnalyzerNode run: {e}")
            error_message = f"Unexpected error during trend analysis: {str(e)}"
            trend_scores = [] # 실패 시 빈 리스트 반환

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['trend_analyzer_node_time'] = node_processing_time
        logger.info(f"{log_prefix} TrendAnalyzerNode finished in {node_processing_time:.2f} seconds. Analyzed {len(trend_scores)} keywords.")

        # TODO: LangSmith 로깅

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "trend_scores": trend_scores,
            "processing_stats": processing_stats,
            "error_message": error_message # 오류 발생 시 메시지 포함
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}