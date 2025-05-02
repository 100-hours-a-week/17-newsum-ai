# app/nodes/07_filter_node.py (Improved Version)

import os
import re
import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
from datetime import datetime, timezone

# 프로젝트 구성 요소 임포트
from app.utils.logger import get_logger
from app.workflows.state import ComicState
from app.config.settings import settings
from app.tools.analysis.language_detector import LanguageDetectionTool
# SpamDetectionService가 실제 서비스라고 가정
from app.services.spam_detector import SpamDetectionService
from app.tools.analysis.text_clusterer import TextClusteringTool

# 로거 설정
logger = get_logger(__name__)

# Simhash 동적 임포트
SIMHASH_AVAILABLE = False
try:
    from simhash import Simhash
    SIMHASH_AVAILABLE = True
except ImportError:
    Simhash = None # type: ignore
    logger.warning("simhash library not installed. Near-duplicate detection disabled.")


class FilterNode:
    """
    원본 의견 데이터를 필터링(언어, 스팸), 중복 제거(SimHash), 클러스터링하여 정리된 의견 목록 생성.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["opinions_raw", "trace_id", "comic_id", "config", "processing_stats"]
    outputs: List[str] = ["opinions_clean", "processing_stats", "error_message"]

    # 의존성 주입
    def __init__(
        self,
        language_tool: LanguageDetectionTool,
        spam_service: SpamDetectionService,
        clustering_tool: TextClusteringTool
        # langsmith_service: Optional[LangSmithService] = None
    ):
        self.language_tool = language_tool
        self.spam_service = spam_service
        self.clustering_tool = clustering_tool
        # self.langsmith = langsmith_service

        # SimHash 설정값은 settings에서 직접 로드 (전역 설정 간주)
        self.simhash_threshold = settings.SIMHASH_THRESHOLD
        self.simhash_width = settings.SIMHASH_TOKEN_WIDTH
        if SIMHASH_AVAILABLE:
             logger.info(f"SimHash configured (Threshold: {self.simhash_threshold}, Width: {self.simhash_width}).")

        # 도구 사용 가능 여부 로깅
        if not self.language_tool: logger.warning("LanguageDetectionTool not injected!")
        if not self.spam_service: logger.warning("SpamDetectionService not injected!")
        if not self.clustering_tool: logger.warning("TextClusteringTool not injected!")
        logger.info("FilterNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.allowed_languages = config.get('language_filter', settings.LANGUAGE_FILTER)
        logger.debug(f"Runtime config loaded. Allowed languages: {self.allowed_languages}")


    # --- Simhash 관련 로직 ---
    def _compute_simhash(self, text: str) -> Optional[Simhash]:
        """텍스트의 Simhash 계산"""
        if not SIMHASH_AVAILABLE or not text: return None
        try:
            # TODO: 한국어 등 다국어 지원을 위해 언어별 토크나이저 사용 고려 (예: Okt, Mecab)
            # 현재: 간단한 공백 및 구두점 기반 토큰화 (영어 중심)
            tokens = re.findall(r'\b\w+\b', text.lower())
            if not tokens: return None
            return Simhash(tokens, f=self.simhash_width)
        except Exception as e:
            logger.error(f"Error computing Simhash: {e}", exc_info=True)
            return None

    def _is_near_duplicate(self, current_simhash: Simhash, existing_hashes: Set[int]) -> bool:
        """Simhash 해밍 거리 기반 유사 중복 확인"""
        if not SIMHASH_AVAILABLE or current_simhash is None: return False
        current_value = current_simhash.value
        for existing_value in existing_hashes:
            distance = bin(current_value ^ existing_value).count('1') # 해밍 거리 계산
            if distance <= self.simhash_threshold:
                return True
        return False

    # --- 메인 실행 메서드 (run으로 이름 변경) ---
    # 이 노드는 CPU 바운드 작업 위주이므로 async 이점 적음. LangGraph 호환성을 위해 async 유지.
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """의견 필터링 및 클러스터링 파이프라인 실행"""
        start_time = datetime.now(timezone.utc)
        comic_id = state.comic_id
        trace_id = state.trace_id
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}
        log_prefix = f"[{trace_id}]"

        logger.info(f"{log_prefix} FilterNode starting...")

        opinions_raw = state.opinions_raw or []
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        if not opinions_raw:
            logger.warning(f"{log_prefix} No raw opinions to filter. Skipping.", extra=extra_log_data)
            processing_stats['filter_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"opinions_clean": [], "processing_stats": processing_stats}

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        # --------------------------

        logger.info(f"{log_prefix} Starting filtering for {len(opinions_raw)} raw opinions...", extra=extra_log_data)

        current_opinions = opinions_raw
        initial_count = len(current_opinions)
        stats_log = {"initial": initial_count}

        # 1. 언어 필터링
        if self.language_tool:
            filtered_by_lang = []
            for opinion in current_opinions:
                text = opinion.get('text', '')
                # 언어 감지 도구 호출 (trace_id, comic_id 전달)
                lang_code = self.language_tool.detect(text, trace_id, comic_id) if text else 'und'
                opinion['language'] = lang_code # 감지된 언어 정보 저장
                if lang_code in self.allowed_languages:
                    filtered_by_lang.append(opinion)
                # else: logger.debug(f"Opinion filtered (Language: {lang_code}). URL: {opinion.get('url')}", extra=extra_log_data)
            current_opinions = filtered_by_lang
            stats_log['after_lang'] = len(current_opinions)
        else:
            logger.warning(f"{log_prefix} Language detection tool unavailable. Skipping language filtering.", extra=extra_log_data)
            for opinion in current_opinions: opinion.setdefault('language', 'und') # 기본값 추가

        # 2. 스팸 필터링
        if self.spam_service:
            filtered_by_spam = []
            for opinion in current_opinions:
                text = opinion.get('text', '')
                # 스팸 탐지 서비스 호출 (trace_id, comic_id 전달)
                if not self.spam_service.is_spam(text, trace_id, comic_id):
                    filtered_by_spam.append(opinion)
                # else: logger.debug(f"Opinion filtered (Spam). URL: {opinion.get('url')}", extra=extra_log_data)
            current_opinions = filtered_by_spam
            stats_log['after_spam'] = len(current_opinions)
        else:
            logger.warning(f"{log_prefix} Spam detection service unavailable. Skipping spam filtering.", extra=extra_log_data)

        # 3. 유사 중복 제거 (Simhash)
        if SIMHASH_AVAILABLE:
            filtered_by_dupe = []
            seen_simhashes_values: Set[int] = set()
            logger.info(f"{log_prefix} Performing Simhash near-duplicate detection (Threshold: {self.simhash_threshold})...", extra=extra_log_data)
            skipped_computation = 0
            duplicate_count = 0

            for opinion in current_opinions:
                text = opinion.get('text', '')
                simhash_obj = self._compute_simhash(text)

                if simhash_obj:
                    if not self._is_near_duplicate(simhash_obj, seen_simhashes_values):
                        filtered_by_dupe.append(opinion)
                        seen_simhashes_values.add(simhash_obj.value)
                    else:
                        # logger.debug(f"Opinion filtered (Near-duplicate). URL: {opinion.get('url')}", extra=extra_log_data)
                        duplicate_count += 1
                else:
                    # Simhash 계산 불가 시 일단 통과
                    filtered_by_dupe.append(opinion)
                    skipped_computation += 1

            if skipped_computation > 0:
                 logger.warning(f"{log_prefix} Skipped Simhash computation for {skipped_computation} items due to errors.", extra=extra_log_data)
            current_opinions = filtered_by_dupe
            stats_log['after_simhash'] = len(current_opinions)
            logger.debug(f"{log_prefix} Simhash removed {duplicate_count} near-duplicates.")
        else:
            logger.warning(f"{log_prefix} Simhash library unavailable. Skipping near-duplicate detection.", extra=extra_log_data)


        # 4. 클러스터링 및 대표 의견 선정
        opinions_clean: List[Dict[str, Any]] = []
        if self.clustering_tool and current_opinions:
            logger.info(f"{log_prefix} Performing text clustering...", extra=extra_log_data)
            try:
                # 클러스터링 도구 호출 (CPU 바운드 작업은 도구 내부에서 처리 가정)
                # cluster_texts 메서드에 trace_id, comic_id 전달
                # 이 메서드는 cluster_id와 is_representative 필드를 추가하여 반환한다고 가정
                opinions_clean = self.clustering_tool.cluster_texts(current_opinions, trace_id, comic_id)
                cluster_summary = Counter(op.get('cluster_id', -1) for op in opinions_clean)
                rep_count = sum(1 for op in opinions_clean if op.get('is_representative'))
                stats_log['after_clustering'] = len(opinions_clean) # 클러스터링 후 개수 (보통 동일)
                logger.info(f"{log_prefix} Clustering complete. Clusters: {len(cluster_summary)}, Representatives: {rep_count}.", extra=extra_log_data)
                logger.debug(f"{log_prefix} Cluster distribution: {dict(cluster_summary)}", extra=extra_log_data)
            except Exception as cluster_err:
                 logger.exception(f"{log_prefix} Error during text clustering: {cluster_err}. Proceeding without clustering.", extra=extra_log_data)
                 # 클러스터링 실패 시, 필터링된 결과를 그대로 사용하고 기본값 할당
                 for op in current_opinions: op.update({'cluster_id': -1, 'is_representative': True})
                 opinions_clean = current_opinions
                 stats_log['after_clustering'] = len(opinions_clean)
        else:
            logger.warning(f"{log_prefix} Clustering skipped (tool unavailable or no opinions). Marking all as representative.", extra=extra_log_data)
            for op in current_opinions: op.update({'cluster_id': 0, 'is_representative': True})
            opinions_clean = current_opinions
            stats_log['after_clustering'] = len(opinions_clean)

        # 최종 필터링 통계 로깅
        logger.info(f"{log_prefix} Filtering stats: {stats_log}")

        # --- 시간 기록 및 상태 업데이트 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['filter_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} FilterNode finished. Elapsed time: {processing_stats['filter_node_time']:.2f} seconds. Final opinions: {len(opinions_clean)}", extra=extra_log_data)

        # 상태 업데이트 준비
        updates = {
            "opinions_clean": opinions_clean,
            "processing_stats": processing_stats,
            "error_message": None # 이 노드 자체의 심각한 오류 시에만 설정
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in updates.items() if k in valid_keys}