# app/nodes/13_progress_report_node.py (Improved Version)

import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, Tuple

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 기본값 참조용
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# 로거 설정
logger = get_logger(__name__)

# Jinja2 의존성 처리
JINJA2_AVAILABLE = False
template_env_cache: Optional[Dict[str, 'jinja2.Environment']] = {} # 경로별 캐시
try:
    import jinja2
    JINJA2_AVAILABLE = True
    logger.info("jinja2 library found.")
except ImportError:
    jinja2 = None # type: ignore
    logger.error("jinja2 library not installed. Report generation disabled.")
except Exception as e:
    jinja2 = None # type: ignore
    logger.exception(f"Error importing jinja2: {e}")

class ProgressReportNode:
    """
    Jinja2 템플릿을 사용하여 중간 진행 보고서(Markdown, Template A)를 생성합니다.
    - 수집된 데이터, 평가 결과, 트렌드 점수 등을 요약합니다.
    - 설정은 `state.config`를 우선 사용하며, 없으면 `settings`의 기본값을 사용합니다.
    """

    # 상태 입력/출력 정의 (참고용)
    inputs: List[str] = [
        "fact_urls", "opinion_urls", "final_summary", "decision", "trend_scores",
        "evaluation_metrics", "timestamp", "trace_id", "used_links",
        "processing_stats", "config"
    ]
    outputs: List[str] = ["progress_report", "processing_stats", "error_message"]

    def __init__(self):
        """노드 초기화."""
        logger.info("ProgressReportNode initialized.")
        if not JINJA2_AVAILABLE:
             logger.error("Report generation disabled due to missing Jinja2 library.")

    def _setup_jinja_env(self, template_dir: str, trace_id: Optional[str]) -> Optional['jinja2.Environment']:
        """설정된 경로로 Jinja2 환경 로드 또는 캐시된 환경 반환"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not JINJA2_AVAILABLE or not template_dir: return None

        # 경로가 같고 캐시된 환경이 있으면 재사용
        if template_dir in template_env_cache:
            logger.debug(f"{log_prefix} Using cached Jinja2 environment for {template_dir}")
            return template_env_cache[template_dir]

        # 새로 로드
        try:
            if not os.path.isdir(template_dir):
                logger.error(f"{log_prefix} Jinja2 template directory not found: {template_dir}")
                return None

            loader = jinja2.FileSystemLoader(searchpath=template_dir)
            env = jinja2.Environment(
                loader=loader,
                autoescape=jinja2.select_autoescape(['html', 'xml', 'md']),
                undefined=jinja2.StrictUndefined # 정의되지 않은 변수 사용 시 오류 발생
            )
            logger.info(f"{log_prefix} Jinja2 environment loaded from: {template_dir}")
            template_env_cache[template_dir] = env # 캐시 저장
            return env
        except Exception as e:
            logger.exception(f"{log_prefix} Error initializing Jinja2 environment from {template_dir}: {e}")
            return None

    # --- 데이터 포맷팅 헬퍼 함수들 ---
    def _truncate_text(self, text: Optional[str], max_length: int) -> str:
        if not text: return ""
        return text[:max_length - 3] + "..." if len(text) > max_length else text

    def _format_metric(self, value: Any, precision: int = 2) -> str:
         try: return f"{float(value):.{precision}f}" if isinstance(value, (int, float)) else str(value)
         except (ValueError, TypeError): return str(value)

    def _prepare_url_info(self, url_list: List[Dict[str, Any]], used_links: List[Dict[str, Any]], url_type: str, trace_id: Optional[str]) -> List[Dict[str, Any]]:
        """URL 목록과 사용 상태를 병합하여 보고서용 데이터 생성"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        prepared_list = []
        if not url_list: return prepared_list

        used_link_map: Dict[str, Dict[str, Any]] = {
            link.get('url', ''): link for link in used_links if link.get('url')
        }

        for idx, url_info in enumerate(url_list):
            url = url_info.get('url', '')
            if not url: continue

            link_data = used_link_map.get(url)
            status = '?' # 기본값: 추적 안됨
            purpose = url_info.get('purpose', 'N/A') # 수집 시 purpose 사용

            if link_data:
                 link_status = link_data.get('status', 'tracked') # used_links의 상태 필드 사용
                 purpose = link_data.get('purpose', purpose) # used_links의 purpose 우선

                 if link_status == 'processed': status = '✓' # 처리됨
                 elif 'failed' in link_status: status = '!' # 실패
                 elif link_status == 'collected': status = '-' # 수집됨 (처리 전)
                 elif link_status == 'tracked': status = '-' # 추적됨 (상태 모호)
                 elif link_status == 'context_used': status = 'C' # 컨텍스트 사용됨 (Node 15에서 설정)

            prepared_list.append({
                'index': idx + 1,
                'url': self._truncate_text(url, 60),
                'status_symbol': status,
                'purpose': self._truncate_text(purpose, 40)
            })
        logger.debug(f"{log_prefix} Prepared {len(prepared_list)} {url_type} URL info entries.")
        return prepared_list

    def _prepare_top_trends(self, trend_scores: List[Dict[str, Any]], top_n: int, trace_id: Optional[str]) -> List[Dict[str, Any]]:
        """상위 N개 트렌드 점수 포맷팅"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not trend_scores:
            logger.warning(f"{log_prefix} No trend scores available for report.")
            return [{'rank': i, 'keyword': 'N/A', 'score': '-'} for i in range(1, top_n + 1)]

        top_trends = trend_scores[:top_n]
        prepared_trends = [
            {'rank': idx + 1,
             'keyword': item.get('keyword', 'N/A'),
             'score': self._format_metric(item.get('score', '-'), 1)}
            for idx, item in enumerate(top_trends)
        ]
        # 결과가 N개 미만일 경우 빈칸 채우기
        while len(prepared_trends) < top_n:
             prepared_trends.append({'rank': len(prepared_trends) + 1, 'keyword': 'N/A', 'score': '-'})
        logger.debug(f"{log_prefix} Prepared top {len(prepared_trends)} trend scores.")
        return prepared_trends

    def _determine_next_step(self, decision: Optional[str], trace_id: Optional[str]) -> str:
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if decision == "proceed": return "Evaluation passed or default. Proceeding to creative generation."
        elif decision == "research_again": return "**Action Recommended**: Evaluation failed thresholds. Re-run data collection/analysis."
        elif decision == "refine_topic": return "**Action Recommended**: Summary quality okay, topic coverage low. Refine topic/keywords."
        elif decision: return f"Evaluation Decision: '{decision}'. Further action depends on logic."
        else:
             logger.warning(f"{log_prefix} Evaluation decision missing in state.")
             return "Evaluation decision not available."

    def _prepare_template_data(self, state: ComicState, config: Dict, trace_id: Optional[str]) -> Dict[str, Any]:
        """Jinja2 템플릿 컨텍스트 데이터 준비"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Preparing data for progress report template...")

        timestamp_str = state.timestamp or datetime.now(timezone.utc).isoformat()
        formatted_timestamp = timestamp_str
        try: formatted_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S %Z")
        except ValueError: pass

        used_links = state.used_links or []
        fact_urls_info = self._prepare_url_info(state.fact_urls or [], used_links, "Fact", trace_id)
        opinion_urls_info = self._prepare_url_info(state.opinion_urls or [], used_links, "Opinion", trace_id)

        final_summary = state.final_summary
        final_summary_preview = self._truncate_text(final_summary, 300) or "[Summary not available]"
        decision = state.decision or "[Decision not available]"

        top_n = config.get("trends_report_top_n", settings.DEFAULT_TRENDS_REPORT_TOP_N)
        trend_scores_top_n = self._prepare_top_trends(state.trend_scores or [], top_n, trace_id)

        evaluation_metrics = state.evaluation_metrics or {}
        topic_coverage = self._format_metric(evaluation_metrics.get("topic_coverage", 0.0) * 100, 1) + "%"
        rouge_l = self._format_metric(evaluation_metrics.get("rouge_l", "-"), 3)
        bertscore = self._format_metric(evaluation_metrics.get("bert_score", "-"), 3)

        next_step = self._determine_next_step(state.decision, trace_id)

        processing_stats = state.processing_stats or {}
        # 노드별 시간 (키 이름 형식 통일 가정)
        time_keys = sorted([k for k in processing_stats if k.endswith('_node_time')])
        node_times = {k.replace('_node_time',''): self._format_metric(processing_stats[k], 1) for k in time_keys}
        total_elapsed = sum(v for k, v in processing_stats.items() if k.endswith('_node_time') and isinstance(v, (int, float)))

        context = {
            "timestamp": formatted_timestamp,
            "trace_id": state.trace_id,
            "fact_urls_info": fact_urls_info,
            "opinion_urls_info": opinion_urls_info,
            "final_summary_preview": final_summary_preview,
            "decision": decision,
            "trend_scores_top_n": trend_scores_top_n,
            "topic_match_percent": topic_coverage,
            "rouge_l_score": rouge_l,
            "bertscore_f1": bertscore,
            "next_step": next_step,
            "node_processing_times": node_times, # 노드별 시간 추가
            "total_elapsed_time": self._format_metric(total_elapsed, 1)
        }
        logger.debug(f"{log_prefix} Template data preparation complete.")
        return context

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """Jinja2 템플릿을 렌더링하여 진행 보고서 생성"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing ProgressReportNode...")

        config = state.config or {}
        processing_stats = state.processing_stats or {}
        report_content = "# Report Generation Failed\n\nJinja2 library not available or template configuration error."
        error_message: Optional[str] = None

        if not JINJA2_AVAILABLE:
            error_message = "Jinja2 library not available."
        else:
            template_dir = config.get("template_dir", settings.DEFAULT_TEMPLATE_DIR) # settings 기본값 사용
            template_name = config.get("progress_report_template_a_filename", settings.DEFAULT_TEMPLATE_A_FILENAME)

            if not template_dir or not template_name:
                error_message = "Template directory or filename not configured."
            else:
                jinja_env = self._setup_jinja_env(template_dir, trace_id)
                if jinja_env:
                    logger.info(f"{log_prefix} Generating progress report using template: {template_name}")
                    try:
                        template = jinja_env.get_template(template_name)
                        template_data = self._prepare_template_data(state, config, trace_id)
                        report_content = template.render(**template_data)
                        logger.info(f"{log_prefix} Progress report generated successfully.")
                    except jinja2.TemplateNotFound:
                        error_message = f"Template file '{template_name}' not found in '{template_dir}'."
                    except Exception as e:
                        error_message = f"Failed to prepare data or render template '{template_name}': {str(e)}"
                        logger.exception(f"{log_prefix} Template rendering error:", exc_info=e)
                else:
                    error_message = f"Failed to initialize Jinja2 environment from '{template_dir}'."

            if error_message:
                 logger.error(f"{log_prefix} Report generation failed: {error_message}")
                 report_content = f"# Report Generation Failed\n\n{error_message}"


        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['progress_report_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} ProgressReportNode finished in {processing_stats['progress_report_node_time']:.2f} seconds.")

        # ComicState 업데이트
        update_data: Dict[str, Any] = {
            "progress_report": report_content,
            "processing_stats": processing_stats,
            "error_message": error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}