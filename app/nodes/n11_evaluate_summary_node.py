# app/nodes/11_evaluate_summary_node.py (Improved Version)

import asyncio
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import os

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 기본값 참조용
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# 로거 및 외부 라이브러리 임포트/확인
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
    # BERTScore는 torch, transformers 등 무거운 의존성이 있을 수 있음
    from bert_score import score as bert_score_calculate
    BERTSCORE_AVAILABLE = True
    logger.info("bert-score library found.")
except ImportError:
    logger.warning("bert-score library (or its dependencies) not installed. BERTScore evaluation will return 0.0.")
except Exception as e:
    # 예를 들어 CUDA 관련 오류 등 로딩 시 다른 문제가 발생할 수 있음
    logger.error(f"Error loading bert-score library: {e}. BERTScore evaluation disabled.", exc_info=True)
    BERTSCORE_AVAILABLE = False


class EvaluateSummaryNode:
    """
    자동 측정 지표(ROUGE-L, BERTScore, Topic Coverage)를 사용하여 최종 요약의 품질을 평가하고,
    결과에 따라 다음 단계를 결정합니다 ('proceed', 'research_again', 'refine_topic').
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = ["final_summary", "topic_analysis", "articles", "news_summaries", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["decision", "evaluation_metrics", "processing_stats", "error_message"]

    def __init__(self):
        """노드 초기화 (Scorer 인스턴스 생성)"""
        self.rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if ROUGE_SCORE_AVAILABLE else None
        logger.info("EvaluateSummaryNode initialized.")
        if not ROUGE_SCORE_AVAILABLE: logger.warning("ROUGE scorer disabled.")
        if not BERTSCORE_AVAILABLE: logger.warning("BERTScore calculator disabled.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        # 평가 임계값 (딕셔너리 전체 또는 개별 로드)
        self.eval_thresholds = config.get('evaluation_thresholds', settings.DEFAULT_EVALUATION_THRESHOLDS)
        self.decision_thresholds = config.get('decision_logic_thresholds', settings.DEFAULT_DECISION_THRESHOLDS)
        # BERTScore 언어 설정
        self.bertscore_lang = config.get('bertscore_lang', settings.DEFAULT_BERTSCORE_LANG)
        # 참조 텍스트 생성 관련 설정
        self.feqa_threshold_for_ref = float(config.get("feqa_threshold", settings.DEFAULT_FEQA_THRESHOLD))
        self.max_summaries_for_ref = int(config.get("max_news_summaries_for_synthesis", settings.DEFAULT_MAX_SUMMARIES_SYNTHESIS))

        # 설정값 로깅 (디버깅용)
        logger.debug(f"Runtime config loaded. Evaluation Thresholds: {self.eval_thresholds}")
        logger.debug(f"Decision Logic Thresholds: {self.decision_thresholds}")
        logger.debug(f"BERTScore Lang: {self.bertscore_lang}")
        logger.debug(f"Ref Text Params - FEQA Thresh: {self.feqa_threshold_for_ref}, Max Summaries: {self.max_summaries_for_ref}")


    def _prepare_reference_text(self, articles: List[Dict[str, Any]], news_summaries: List[Dict[str, Any]], trace_id: Optional[str]) -> str:
        """ROUGE 및 BERTScore 계산을 위한 참조 텍스트 준비 (FEQA 점수 높은 요약 우선)"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        reference_texts = []

        if news_summaries:
            # FEQA 점수 높은 요약 필터링 및 정렬
            high_quality_summaries = sorted(
                [s for s in news_summaries if s.get("feqa_score", 0.0) >= self.feqa_threshold_for_ref and s.get("summary_text")],
                key=lambda x: x.get("feqa_score", 0.0), reverse=True
            )[:self.max_summaries_for_ref] # 최대 개수 제한

            selected_texts = [s['summary_text'] for s in high_quality_summaries]

            if selected_texts:
                reference_texts = selected_texts
                logger.debug(f"{log_prefix} Using {len(reference_texts)} high-quality news summaries as reference.")
            else: # 고품질 요약 없으면 점수 무관 상위 N개 시도
                 all_valid_summaries = sorted(
                     [s for s in news_summaries if s.get("summary_text")],
                     key=lambda x: x.get("feqa_score", 0.0), reverse=True
                 )[:self.max_summaries_for_ref]
                 selected_texts = [s['summary_text'] for s in all_valid_summaries]
                 if selected_texts:
                      reference_texts = selected_texts
                      logger.debug(f"{log_prefix} No high-quality summaries found. Using {len(reference_texts)} top summaries as reference.")

        # 요약 없으면 기사 제목 사용 (Fallback)
        if not reference_texts and articles:
            valid_titles = [a.get("title", "") for a in articles if a.get("title")]
            if valid_titles:
                 reference_texts = valid_titles[:5] # 너무 많으면 제한 (예: 5개)
                 logger.debug(f"{log_prefix} No usable summaries. Using {len(reference_texts)} article titles as reference.")

        if not reference_texts:
            logger.warning(f"{log_prefix} Could not prepare reference text. Evaluation scores might be 0.")
            return ""

        return "\n\n".join(reference_texts).strip() # 단일 문자열로 결합

    def _calculate_rouge_l(self, summary: str, reference_text: str, trace_id: Optional[str]) -> float:
        """ROUGE-L F1 점수 계산"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not ROUGE_SCORE_AVAILABLE or not self.rouge_l_scorer or not summary or not reference_text:
            if not ROUGE_SCORE_AVAILABLE: logger.warning(f"{log_prefix} ROUGE scorer unavailable.")
            else: logger.warning(f"{log_prefix} Cannot calculate ROUGE-L: missing summary or reference text.")
            return 0.0
        try:
            # rouge_scorer.score(target, prediction) - target=reference, prediction=summary
            scores = self.rouge_l_scorer.score(reference_text, summary)
            f1_score = scores['rougeL'].fmeasure
            logger.debug(f"{log_prefix} ROUGE-L F1 calculated: {f1_score:.4f}")
            return round(f1_score, 4)
        except Exception as e:
            logger.exception(f"{log_prefix} Error calculating ROUGE-L score: {e}")
            return 0.0

    # 참고: BERTScore 계산은 모델 로딩 등으로 인해 시간이 걸릴 수 있음
    def _calculate_bert_score(self, summary: str, reference_text: str, trace_id: Optional[str]) -> float:
        """BERTScore F1 점수 계산"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not BERTSCORE_AVAILABLE or not summary or not reference_text:
            if not BERTSCORE_AVAILABLE: logger.warning(f"{log_prefix} BERTScore library unavailable.")
            else: logger.warning(f"{log_prefix} Cannot calculate BERTScore: missing summary or reference text.")
            return 0.0
        try:
            # bert_score 함수는 candidate와 reference 리스트를 인자로 받음
            precision, recall, f1 = bert_score_calculate(
                [summary], [reference_text], lang=self.bertscore_lang, rescale_with_baseline=True, verbose=False
            )
            # 결과는 tensor 형태이므로 .item()으로 스칼라 값 추출
            f1_score = f1.mean().item()
            logger.debug(f"{log_prefix} BERTScore F1 (lang={self.bertscore_lang}) calculated: {f1_score:.4f}")
            return round(f1_score, 4)
        except Exception as e:
            # 모델 다운로드 실패, 메모리 부족 등 다양한 오류 가능
            logger.exception(f"{log_prefix} Error calculating BERTScore: {e}")
            return 0.0

    def _calculate_topic_coverage(self, summary: str, topic_analysis: Dict[str, Any], trace_id: Optional[str]) -> float:
        """요약의 토픽 커버리지 계산 (토픽 분석 결과 기반)"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not summary or not topic_analysis: return 0.0

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

        keywords = topic_analysis.get("keywords_analyzed", [])
        if isinstance(keywords, list):
            for kw_item in keywords:
                 keyword = kw_item.get("keyword") if isinstance(kw_item, dict) else None
                 if isinstance(keyword, str) and keyword.strip():
                      all_terms.add(keyword.strip().lower())

        if not all_terms:
            logger.warning(f"{log_prefix} No valid terms found in topic_analysis for coverage check.")
            return 0.0

        covered_count = 0
        covered_terms_list = []
        for term in all_terms:
            # 단어 경계를 고려한 매칭 (더 정확함)
            if re.search(r'\b' + re.escape(term) + r'\b', summary_lower):
                covered_count += 1
                covered_terms_list.append(term)

        coverage_ratio = covered_count / len(all_terms) if all_terms else 0.0
        logger.debug(f"{log_prefix} Topic Coverage: {coverage_ratio:.4f} ({covered_count}/{len(all_terms)} terms covered). Terms: {list(all_terms)}")
        logger.debug(f"{log_prefix} Covered terms: {covered_terms_list}")
        return round(coverage_ratio, 4)

    def _make_decision(self, scores: Dict[str, float], trace_id: Optional[str]) -> str:
        """점수 및 설정된 임계값 기반 의사결정"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        rouge = scores.get('rouge_l', 0.0)
        bert = scores.get('bert_score', 0.0)
        coverage = scores.get('topic_coverage', 0.0)

        # 임계값 로드 (개별 값 접근)
        thresh_rouge = self.eval_thresholds.get('rouge_l', 0.3)
        thresh_bert = self.eval_thresholds.get('bert_score', 0.7)
        thresh_coverage = self.eval_thresholds.get('topic_coverage', 0.6)

        # 결정 로직 임계값 로드
        decision_vr = self.decision_thresholds.get('very_low_rouge', 0.1)
        decision_vb = self.decision_thresholds.get('very_low_bertscore', 0.5)
        decision_vc = self.decision_thresholds.get('very_low_coverage', 0.3)
        decision_lchm = self.decision_thresholds.get('low_coverage_high_metrics', 0.7) # ROUGE/BERT 기준

        log_msg = f"Scores R={rouge:.3f}(T={thresh_rouge:.2f}), B={bert:.3f}(T={thresh_bert:.2f}), C={coverage:.3f}(T={thresh_coverage:.2f})."

        # 1. 진행 조건 (모든 평가 임계값 만족)
        if rouge >= thresh_rouge and bert >= thresh_bert and coverage >= thresh_coverage:
            logger.info(f"{log_prefix} Decision: proceed (All metrics meet thresholds). {log_msg}")
            return "proceed"

        # 2. 재탐색 조건 (하나라도 매우 낮은 점수)
        if rouge < decision_vr or bert < decision_vb or coverage < decision_vc:
            logger.warning(f"{log_prefix} Decision: research_again (One or more metrics very low). {log_msg}")
            return "research_again"

        # 3. 토픽 재정의 조건 (커버리지 낮음, 다른 지표 양호)
        if coverage < thresh_coverage and rouge >= decision_lchm and bert >= decision_lchm:
            logger.warning(f"{log_prefix} Decision: refine_topic (Coverage low, R/B acceptable). {log_msg}")
            return "refine_topic"

        # 4. 기본 결정: 임계값 미달이지만 재탐색/토픽재정의 조건 미충족 시 일단 진행
        logger.info(f"{log_prefix} Decision: proceed (Default - below threshold but not triggering retry/refine). {log_msg}")
        return "proceed"


    # --- 메인 실행 메서드 ---
    # 참고: BERTScore 계산 등 CPU/GPU 바운드 작업은 asyncio 이점 적음
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """요약 평가 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing EvaluateSummaryNode...")

        final_summary = state.final_summary or ""
        topic_analysis = state.topic_analysis or {}
        articles = state.articles or []
        news_summaries = state.news_summaries or []
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        # --------------------------

        if not final_summary:
            logger.error(f"{log_prefix} Final summary is missing. Evaluation cannot proceed.")
            metrics = {"rouge_l": 0.0, "bert_score": 0.0, "topic_coverage": 0.0}
            processing_stats['evaluate_summary_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "decision": "research_again",
                "evaluation_metrics": metrics,
                "processing_stats": processing_stats,
                "error_message": "Final summary is missing for evaluation."
            }

        logger.info(f"{log_prefix} Starting final summary evaluation...")

        # --- 참조 텍스트 준비 ---
        reference_text = self._prepare_reference_text(articles, news_summaries, trace_id)

        # --- 메트릭 계산 ---
        # CPU/GPU 바운드 작업들은 순차 실행 (필요시 run_in_executor 사용)
        rouge_l_score = self._calculate_rouge_l(final_summary, reference_text, trace_id)
        # BERTScore는 특히 느릴 수 있음
        bert_score = self._calculate_bert_score(final_summary, reference_text, trace_id)
        topic_coverage_score = self._calculate_topic_coverage(final_summary, topic_analysis, trace_id)

        evaluation_metrics = {
            "rouge_l": rouge_l_score,
            "bert_score": bert_score,
            "topic_coverage": topic_coverage_score
        }
        logger.info(f"{log_prefix} Calculated Metrics: {evaluation_metrics}")

        # --- 의사 결정 ---
        decision = self._make_decision(evaluation_metrics, trace_id)

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['evaluate_summary_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} EvaluateSummaryNode finished in {processing_stats['evaluate_summary_node_time']:.2f} seconds. Decision: {decision}")

        # ComicState 업데이트
        update_data: Dict[str, Any] = {
            "decision": decision,
            "evaluation_metrics": evaluation_metrics,
            "processing_stats": processing_stats,
            "error_message": None
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}