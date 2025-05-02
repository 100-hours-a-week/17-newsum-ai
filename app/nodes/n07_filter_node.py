# app/nodes/07_filter_node.py

import os
import re
import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
from datetime import datetime, timezone

# --- 리팩토링된 임포트 ---
from app.utils.logger import get_logger
from app.workflows.state import ComicState
from app.config.settings import settings
# 필요한 도구/서비스 임포트
from app.tools.analysis.language_detector import LanguageDetectionTool
from app.services.spam_detector import SpamDetectionService # Placeholder 또는 실제 서비스
from app.tools.analysis.text_clusterer import TextClusteringTool

# Simhash 동적 임포트 (노드 내 로직 유지)
SIMHASH_AVAILABLE = False
try:
    from simhash import Simhash
    SIMHASH_AVAILABLE = True
except ImportError:
    Simhash = None # type: ignore

logger = get_logger("FilterNode") # 로거 설정

if not SIMHASH_AVAILABLE:
    logger.warning("simhash 라이브러리가 설치되지 않았습니다. 유사 중복 탐지 기능이 비활성화됩니다.")


class FilterNode:
    """
    (리팩토링됨) 원본 의견 데이터를 필터링(언어, 스팸), 중복 제거(SimHash),
    클러스터링(TF-IDF, K-Means)하여 정리된 의견 목록을 생성합니다.
    주입된 도구(언어 감지, 스팸 탐지, 클러스터링)와 내부 로직(SimHash)을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["opinions_raw", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["opinions_clean", "processing_stats", "error_message"]

    def __init__(
        self,
        language_tool: LanguageDetectionTool,
        spam_service: SpamDetectionService, # 이름 변경 (Service)
        clustering_tool: TextClusteringTool
    ):
        """
        FilterNode 초기화. 분석 및 필터링 도구를 주입받습니다.

        Args:
            language_tool (LanguageDetectionTool): 언어 감지 도구.
            spam_service (SpamDetectionService): 스팸 탐지 서비스 (Placeholder 또는 실제 모델).
            clustering_tool (TextClusteringTool): 텍스트 클러스터링 도구.
        """
        self.language_tool = language_tool
        self.spam_service = spam_service
        self.clustering_tool = clustering_tool

        # SimHash 설정값 로드 (노드 내 로직용)
        self.simhash_threshold = settings.SIMHASH_THRESHOLD
        self.simhash_width = settings.SIMHASH_TOKEN_WIDTH
        logger.info(f"SimHash 임계값: {self.simhash_threshold}, 너비: {self.simhash_width}")

        # 도구 사용 가능 여부 로깅
        if not self.language_tool: logger.warning("LanguageDetectionTool이 주입되지 않았습니다!")
        if not self.spam_service: logger.warning("SpamDetectionService가 주입되지 않았습니다!")
        if not self.clustering_tool: logger.warning("TextClusteringTool이 주입되지 않았습니다!")


    # --- Simhash 관련 로직 (노드 내 유지) ---
    def _compute_simhash(self, text: str) -> Optional[Simhash]:
        """주어진 텍스트의 Simhash를 계산합니다."""
        if not SIMHASH_AVAILABLE or not text: return None
        try:
            # 간단한 토큰화: 비-알파벳/숫자 문자로 분리 후 소문자 변환
            # TODO: 한국어 등 언어 특화 토크나이저 사용 고려 (예: Okt, Mecab)
            tokens = re.findall(r'\w+', text.lower())
            if not tokens: return None
            # Simhash 객체 생성 (설정된 너비 사용)
            return Simhash(tokens, f=self.simhash_width)
        except Exception as e:
            logger.error(f"Simhash 계산 오류: {e}", exc_info=True)
            return None

    def _is_near_duplicate(self, current_simhash: Simhash, existing_hashes: Set[int]) -> bool:
        """현재 Simhash가 기존 해시 집합과 유사 중복인지 확인합니다."""
        if not SIMHASH_AVAILABLE or current_simhash is None: return False

        current_value = current_simhash.value # 현재 해시 값

        # 해밍 거리 비교 (임계값 이하이면 중복)
        for existing_value in existing_hashes:
            # Simhash 객체 재구성 없이 직접 비트 연산으로 거리 계산 가능
            # (또는 Simhash 라이브러리의 distance 메서드 사용)
            # distance = current_simhash.distance(Simhash(existing_value)) # 원본 방식
            # 비트 연산 방식 (더 효율적일 수 있음)
            xor_val = current_value ^ existing_value
            distance = bin(xor_val).count('1') # XOR 후 1의 개수 = 해밍 거리

            if distance <= self.simhash_threshold:
                return True # 임계값 이하이면 유사 중복
        return False # 유사 중복 없음


    # --- 메인 실행 메서드 ---
    # 이 노드는 주로 CPU-bound 작업이므로 async는 LangGraph 호환성 위주
    async def execute(self, state: ComicState) -> Dict[str, Any]:
        """
        의견 필터링 및 클러스터링 파이프라인을 실행합니다.

        Args:
            state (ComicState): 현재 워크플로우 상태 객체.

        Returns:
            Dict[str, Any]: 상태 업데이트를 위한 사전 (opinions_clean, processing_stats 등).
        """
        start_time = datetime.now(timezone.utc)
        comic_id = state.comic_id
        trace_id = state.trace_id or comic_id or "unknown_trace"
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id}

        logger.info("FilterNode 실행 시작...", extra=extra_log_data)

        # 상태에서 필요한 정보 로드
        opinions_raw = state.opinions_raw
        config = state.config
        processing_stats = state.processing_stats

        # 처리할 의견 없으면 종료
        if not opinions_raw:
            logger.warning("필터링할 원본 의견이 없습니다. FilterNode를 건너<0xEB><0x9C><0x95>니다.", extra=extra_log_data)
            return {"opinions_clean": [], "processing_stats": processing_stats}

        logger.info(f"{len(opinions_raw)}개의 원본 의견 필터링 시작...", extra=extra_log_data)

        # --- 파이프라인 단계 ---
        current_opinions = opinions_raw
        passed_count = len(current_opinions)

        # 1. 언어 필터링
        if self.language_tool:
            filtered_by_lang = []
            allowed_languages = config.get('language_filter', settings.LANGUAGE_FILTER)
            for opinion in current_opinions:
                text = opinion.get('text', '')
                lang_code = self.language_tool.detect(text, trace_id, comic_id) if text else 'und'
                opinion['language'] = lang_code # 언어 정보 추가
                if lang_code in allowed_languages:
                    filtered_by_lang.append(opinion)
                else:
                    logger.debug(f"의견 필터링됨 (언어: {lang_code}, 허용: {allowed_languages}). URL: {opinion.get('url')}", extra=extra_log_data)
            current_opinions = filtered_by_lang
            logger.info(f"언어 필터링 후: {len(current_opinions)}개 의견 남음 (-{passed_count - len(current_opinions)})", extra=extra_log_data)
            passed_count = len(current_opinions)
        else:
            logger.warning("언어 감지 도구 사용 불가. 언어 필터링 건너<0xEB><0x9C><0x95>뜀.", extra=extra_log_data)
            # 언어 정보 기본값 추가
            for opinion in current_opinions: opinion.setdefault('language', 'und')


        # 2. 스팸 필터링
        if self.spam_service:
            filtered_by_spam = []
            for opinion in current_opinions:
                text = opinion.get('text', '')
                # is_spam 호출 시 trace_id, comic_id 전달
                if not self.spam_service.is_spam(text, trace_id, comic_id):
                    filtered_by_spam.append(opinion)
                else:
                    logger.debug(f"의견 필터링됨 (스팸). URL: {opinion.get('url')}", extra=extra_log_data)
            current_opinions = filtered_by_spam
            logger.info(f"스팸 필터링 후: {len(current_opinions)}개 의견 남음 (-{passed_count - len(current_opinions)})", extra=extra_log_data)
            passed_count = len(current_opinions)
        else:
            logger.warning("스팸 탐지 서비스 사용 불가. 스팸 필터링 건너<0xEB><0x9C><0x95>뜀.", extra=extra_log_data)


        # 3. 유사 중복 제거 (Simhash) - 노드 내 로직
        if SIMHASH_AVAILABLE:
            filtered_by_dupe = []
            # Set으로 변경: 정수형 해시 값 저장
            seen_simhashes_values: Set[int] = set()
            logger.info(f"Simhash 유사 중복 탐지 시작 (임계값: {self.simhash_threshold})...", extra=extra_log_data)

            for opinion in current_opinions:
                text = opinion.get('text', '')
                simhash_obj = self._compute_simhash(text) # Simhash 계산

                if simhash_obj:
                    # 유사 중복 검사
                    if not self._is_near_duplicate(simhash_obj, seen_simhashes_values):
                        filtered_by_dupe.append(opinion) # 중복 아니면 추가
                        seen_simhashes_values.add(simhash_obj.value) # 해시 값 저장
                    else:
                        logger.debug(f"의견 필터링됨 (유사 중복). URL: {opinion.get('url')}", extra=extra_log_data)
                else:
                    # Simhash 계산 불가 시 일단 통과시킴 (또는 다른 처리 방식 선택)
                    filtered_by_dupe.append(opinion)
                    logger.warning(f"Simhash 계산 불가하여 중복 검사 건너<0xEB><0x9C><0x95>뜀. URL: {opinion.get('url')}", extra=extra_log_data)

            current_opinions = filtered_by_dupe
            logger.info(f"Simhash 중복 제거 후: {len(current_opinions)}개 의견 남음 (-{passed_count - len(current_opinions)})", extra=extra_log_data)
            passed_count = len(current_opinions)
        else:
            logger.warning("Simhash 라이브러리 사용 불가. 유사 중복 제거 건너<0xEB><0x9C><0x95>뜀.", extra=extra_log_data)


        # 4. 클러스터링 및 대표 의견 선정
        opinions_clean: List[Dict[str, Any]] = [] # 최종 결과 저장 리스트
        if self.clustering_tool and current_opinions: # 도구 있고 처리할 의견 있으면 실행
            logger.info("클러스터링 시작...", extra=extra_log_data)
            # 클러스터링 도구 호출 (CPU-bound 작업은 도구 내부에서 처리 가정)
            # cluster_texts는 동기 함수일 수 있음. 필요 시 executor 사용.
            # 현재 TextClusteringTool은 동기 함수로 가정.
            opinions_clean = self.clustering_tool.cluster_texts(current_opinions, trace_id, comic_id)
            # 클러스터링 결과 요약 로깅
            cluster_summary = Counter(op.get('cluster_id', -1) for op in opinions_clean)
            rep_count = sum(1 for op in opinions_clean if op.get('is_representative'))
            logger.info(f"클러스터링 완료. 클러스터: {len(cluster_summary)}개. 대표 의견: {rep_count}개.", extra=extra_log_data)
            logger.debug(f"클러스터 분포: {dict(cluster_summary)}", extra=extra_log_data)
        else:
            # 클러스터링 건너<0xEB><0x9C><0x95>뛴 경우 처리
            logger.warning("클러스터링 건너<0xEB><0x9C><0x94> (도구 없거나 처리할 의견 없음). 모든 의견을 대표로 지정.", extra=extra_log_data)
            # 기본 클러스터 ID 및 대표 플래그 할당
            for op in current_opinions: op.update({'cluster_id': 0, 'is_representative': True})
            opinions_clean = current_opinions


        # --- 시간 기록 및 상태 업데이트 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['node_07_filter_time'] = node_processing_time # 고유 키 사용
        logger.info(f"FilterNode 완료. 소요 시간: {node_processing_time:.2f} 초. 최종 의견 수: {len(opinions_clean)}", extra=extra_log_data)

        # 상태 업데이트 준비
        updates = {
            "opinions_clean": opinions_clean, # 최종 정리된 의견 목록
            "processing_stats": processing_stats # 업데이트된 처리 통계
        }
        # 이 노드 관련 이전 오류 메시지 상태 초기화
        current_error = state.error_message
        if current_error and "FilterNode" in current_error:
             updates["error_message"] = None # 성공 시 None으로 설정

        return updates