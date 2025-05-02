# app/nodes/11_evaluate_summary_node.py

import asyncio
import re
# --- datetime, timezone 임포트 추가 ---
from datetime import datetime, timezone
# ------------------------------------
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np # numpy는 bert-score 등에서 내부적으로 사용될 수 있음
import os # bertscore_lang 로딩에 사용 (settings으로 대체 예정)

# --- 프로젝트 구성 요소 임포트 ---
# from app.config.settings import settings # 직접 사용하지 않음 (config 통해 받음)
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# --- 의존성 라이브러리 임포트 및 확인 ---
logger = get_logger("EvaluateSummaryNode") # 로거 먼저 정의

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
    logger.warning("bert-score library (or its dependencies like torch/transformers) not installed. BERTScore evaluation will return 0.0.")
# --- 의존성 임포트 종료 ---


class EvaluateSummaryNode:
    """
    (Refactored) 자동 측정 지표를 사용하여 최종 요약의 품질을 평가합니다.
    - 계산 지표: ROUGE-L, BERTScore, Topic Coverage.
    - 임계값 기반으로 의사결정 ('proceed', 'research_again', 'refine_topic').
    - 설정은 상태의 config 딕셔너리에서 로드.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["final_summary", "topic_analysis", "articles", "news_summaries", "trace_id", "config", "processing_stats"]
    outputs: List[str] = ["decision", "evaluation_metrics", "processing_stats", "error_message"]

    def __init__(self):
        """노드 초기화 (Scorer 인스턴스 생성)"""
        # rouge_scorer는 상태 비저장(stateless)이므로 인스턴스 변수로 저장 가능
        self.rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if ROUGE_SCORE_AVAILABLE else None
        # BERTScore는 모델 로딩 때문에 run 메서드 내에서 호출 시점에 처리
        # self.bert_score_lang 은 run 메서드에서 config 통해 로드

        logger.info("EvaluateSummaryNode initialized.")
        if not ROUGE_SCORE_AVAILABLE: logger.warning("ROUGE scorer disabled.")
        if not BERTSCORE_AVAILABLE: logger.warning("BERTScore calculator disabled.")


    def _prepare_reference_text(self, articles: List[Dict[str, Any]], news_summaries: List[Dict[str, Any]], config: Dict, trace_id: Optional[str]) -> str:
        """ROUGE 및 BERTScore 계산을 위한 참조 텍스트 준비"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        reference_texts = []
        feqa_threshold = float(config.get("feqa_threshold", 0.5)) # config에서 로드
        max_summaries = config.get("max_news_summaries_for_synthesis", 3) # 통합 요약에 사용된 개수만큼 참조

        # FEQA 점수 높은 뉴스 요약 우선 사용
        if news_summaries:
            high_quality_summaries = sorted(
                [s for s in news_summaries if s.get("feqa_score", 0.0) >= feqa_threshold and s.get("summary_text")],
                key=lambda x: x.get("feqa_score", 0.0), reverse=True
            )[:max_summaries]
            selected_texts = [s['summary_text'] for s in high_quality_summaries]

            if selected_texts:
                reference_texts = selected_texts
                logger.debug(f"{log_prefix} Using {len(reference_texts)} high-quality news summaries as reference.")
            else: # 고품질 요약 없으면 점수 무관 상위 N개 시도
                 all_valid_summaries = sorted(
                     [s for s in news_summaries if s.get("summary_text")],
                     key=lambda x: x.get("feqa_score", 0.0), reverse=True
                 )[:max_summaries]
                 selected_texts = [s['summary_text'] for s in all_valid_summaries]
                 if selected_texts:
                      reference_texts = selected_texts
                      logger.debug(f"{log_prefix} No high-quality summaries. Using {len(reference_texts)} top summaries as reference.")

        # 요약 없으면 기사 제목 사용 (Fallback)
        if not reference_texts and articles:
            valid_titles = [a.get("title", "") for a in articles if a.get("title")]
            if valid_titles:
                 # 너무 많으면 잘라낼 수 있음 (예: 상위 5개)
                 reference_texts = valid_titles[:5]
                 logger.debug(f"{log_prefix} No usable summaries. Using {len(reference_texts)} article titles as reference.")

        if not reference_texts:
            logger.warning(f"{log_prefix} Could not prepare reference text. Evaluation scores might be 0.")
            return ""

        # 단일 문자열로 결합
        return "\n\n".join(reference_texts)

    def _calculate_rouge_l(self, summary: str, reference_text: str, trace_id: Optional[str]) -> float:
        """ROUGE-L F1 점수 계산"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not self.rouge_l_scorer or not summary or not reference_text:
            if not self.rouge_l_scorer: logger.warning(f"{log_prefix} ROUGE scorer unavailable.")
            return 0.0
        try:
            scores = self.rouge_l_scorer.score(reference_text, summary) # target, prediction 순서 주의
            f1_score = scores['rougeL'].fmeasure
            logger.debug(f"{log_prefix} ROUGE-L F1 calculated: {f1_score:.4f}")
            return round(f1_score, 4) # 소수점 4자리 반올림
        except Exception as e:
            logger.exception(f"{log_prefix} Error calculating ROUGE-L score: {e}")
            return 0.0

    def _calculate_bert_score(self, summary: str, reference_text: str, lang: str, trace_id: Optional[str]) -> float:
        """BERTScore F1 점수 계산"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not BERTSCORE_AVAILABLE or not summary or not reference_text:
            if not BERTSCORE_AVAILABLE: logger.warning(f"{log_prefix} BERTScore library unavailable.")
            return 0.0
        try:
            # bert_score 함수는 리스트를 인자로 받음
            precision, recall, f1 = bert_score_calculate(
                [summary], [reference_text], lang=lang, rescale_with_baseline=True, verbose=False
            )
            f1_score = f1.mean().item() # 여러 참조가 있을 경우 평균 사용, 여기선 하나이므로 .item()
            logger.debug(f"{log_prefix} BERTScore F1 (lang={lang}) calculated: {f1_score:.4f}")
            return round(f1_score, 4)
        except Exception as e:
            logger.exception(f"{log_prefix} Error calculating BERTScore: {e}")
            return 0.0

    def _calculate_topic_coverage(self, summary: str, topic_analysis: Dict[str, Any], trace_id: Optional[str]) -> float:
        """요약의 토픽 커버리지 계산"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not summary or not topic_analysis: return 0.0

        summary_lower = summary.lower()
        all_terms: Set[str] = set()

        # 주요 토픽 추가
        main_topic = topic_analysis.get("main_topic", "")
        if isinstance(main_topic, str) and main_topic.strip():
            all_terms.add(main_topic.strip().lower())

        # 엔티티 이름 추가
        entities = topic_analysis.get("entities", [])
        if isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, dict) and isinstance(entity.get("name"), str) and entity["name"].strip():
                    all_terms.add(entity["name"].strip().lower())

        # 분석된 키워드 추가
        keywords = topic_analysis.get("keywords_analyzed", [])
        if isinstance(keywords, list):
            for kw_item in keywords:
                 if isinstance(kw_item, dict) and isinstance(kw_item.get("keyword"), str) and kw_item["keyword"].strip():
                      all_terms.add(kw_item["keyword"].strip().lower())

        if not all_terms:
            logger.warning(f"{log_prefix} No valid terms found in topic_analysis for coverage check.")
            return 0.0

        # 커버된 용어 확인 (단순 포함 여부)
        covered_count = 0
        covered_terms_list = []
        for term in all_terms:
            # 더 정확한 매칭을 위해 단어 경계 고려 가능 (r'\b' + re.escape(term) + r'\b')
            if term in summary_lower:
                covered_count += 1
                covered_terms_list.append(term)

        coverage_ratio = covered_count / len(all_terms) if all_terms else 0.0
        logger.debug(f"{log_prefix} Topic Coverage: {coverage_ratio:.4f} ({covered_count}/{len(all_terms)} terms covered). Covered terms: {covered_terms_list}")
        return round(coverage_ratio, 4)

    def _make_decision(self, scores: Dict[str, float], thresholds: Dict, decision_logic: Dict, trace_id: Optional[str]) -> str:
        """점수 및 임계값 기반 의사결정"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        rouge = scores.get('rouge_l', 0.0)
        bert = scores.get('bert_score', 0.0)
        coverage = scores.get('topic_coverage', 0.0)

        # 임계값 안전하게 가져오기 (기본값 설정)
        thresh_rouge = thresholds.get('rouge_l', 0.3)
        thresh_bert = thresholds.get('bert_score', 0.7)
        thresh_coverage = thresholds.get('topic_coverage', 0.6)

        decision_vr = decision_logic.get('very_low_rouge', 0.1)
        decision_vb = decision_logic.get('very_low_bertscore', 0.5)
        decision_vc = decision_logic.get('very_low_coverage', 0.3)
        decision_lchm = decision_logic.get('low_coverage_high_metrics', 0.7)

        log_msg = f"Scores R={rouge:.3f}(T={thresh_rouge:.2f}), B={bert:.3f}(T={thresh_bert:.2f}), C={coverage:.3f}(T={thresh_coverage:.2f})."

        # 1. 진행 조건
        if rouge >= thresh_rouge and bert >= thresh_bert and coverage >= thresh_coverage:
            logger.info(f"{log_prefix} Decision: proceed (All metrics meet thresholds). {log_msg}")
            return "proceed"

        # 2. 재탐색 조건 (매우 낮은 점수)
        if rouge < decision_vr or bert < decision_vb or coverage < decision_vc:
            logger.warning(f"{log_prefix} Decision: research_again (One or more metrics very low). {log_msg}")
            return "research_again"

        # 3. 토픽 재정의 조건 (커버리지 낮음, 다른 지표 양호)
        # 주의: low_coverage_high_metrics 임계값은 ROUGE/BERTScore에 대한 기준임
        if coverage < thresh_coverage and rouge >= decision_lchm and bert >= decision_lchm:
            logger.warning(f"{log_prefix} Decision: refine_topic (Coverage low, R/B acceptable). {log_msg}")
            return "refine_topic"

        # 4. 기본 결정 (임계값 미달이지만 특정 조건 미해당)
        # 이 경우, '진행'시키는 것이 합리적일 수 있으나 요구사항에 따라 변경 가능
        logger.info(f"{log_prefix} Decision: proceed (Default - below threshold but not triggering retry/refine). {log_msg}")
        return "proceed"


    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """요약 평가 프로세스 실행"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing EvaluateSummaryNode...")

        # 상태 및 설정 로드
        final_summary = state.final_summary or ""
        topic_analysis = state.topic_analysis or {}
        articles = state.articles or []
        news_summaries = state.news_summaries or []
        config = state.config or {}
        processing_stats = state.processing_stats or {}

        # 설정값 로드
        eval_thresholds = config.get('evaluation_thresholds', {})
        decision_thresholds = config.get('decision_logic_thresholds', {})
        bertscore_lang = config.get('bertscore_lang', 'en') # config에서 언어 로드

        # 입력 유효성 검사
        if not final_summary:
            logger.error(f"{log_prefix} Final summary is missing. Evaluation cannot proceed.")
            metrics = {"rouge_l": 0.0, "bert_score": 0.0, "topic_coverage": 0.0}
            processing_stats['evaluate_summary_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {
                "decision": "research_again", # 요약 없으면 재탐색 유도
                "evaluation_metrics": metrics,
                "processing_stats": processing_stats,
                "error_message": "Final summary is missing for evaluation."
            }

        logger.info(f"{log_prefix} Starting final summary evaluation...")
        logger.debug(f"{log_prefix} Summary length: {len(final_summary)} chars.")
        logger.debug(f"{log_prefix} Evaluation Thresholds: {eval_thresholds}")
        logger.debug(f"{log_prefix} Decision Logic Thresholds: {decision_thresholds}")
        logger.debug(f"{log_prefix} BERTScore Lang: {bertscore_lang}")


        # --- 참조 텍스트 준비 ---
        # 이 작업은 비동기 I/O가 없으므로 직접 실행
        reference_text = self._prepare_reference_text(articles, news_summaries, config, state.trace_id)

        # --- 메트릭 계산 ---
        # CPU 바운드 작업들이므로 비동기로 얻는 이득 적음. 순차 실행.
        # BERTScore는 모델 로딩 등으로 느릴 수 있음. 필요시 run_in_executor 사용 고려.
        rouge_l_score = self._calculate_rouge_l(final_summary, reference_text, state.trace_id)
        bert_score = self._calculate_bert_score(final_summary, reference_text, bertscore_lang, state.trace_id)
        topic_coverage_score = self._calculate_topic_coverage(final_summary, topic_analysis, state.trace_id)

        evaluation_metrics = {
            "rouge_l": rouge_l_score,
            "bert_score": bert_score,
            "topic_coverage": topic_coverage_score
        }
        logger.info(f"{log_prefix} Calculated Metrics: {evaluation_metrics}")

        # --- 의사 결정 ---
        decision = self._make_decision(evaluation_metrics, eval_thresholds, decision_thresholds, state.trace_id)

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['evaluate_summary_node_time'] = node_processing_time
        logger.info(f"{log_prefix} EvaluateSummaryNode finished in {node_processing_time:.2f} seconds. Decision: {decision}")

        # TODO: LangSmith 로깅 (결과 점수, 결정 등)
        # if self.langsmith: ...

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "decision": decision,
            "evaluation_metrics": evaluation_metrics,
            "processing_stats": processing_stats,
            "error_message": None # 이 노드 자체의 심각한 오류 시에만 설정
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}