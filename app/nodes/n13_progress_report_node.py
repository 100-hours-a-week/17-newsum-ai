# app/nodes/13_progress_report_node.py

import os
import re
# --- datetime, timezone 임포트 추가 ---
from datetime import datetime, timezone
# ------------------------------------
from typing import List, Dict, Any, Optional, Set, Tuple

# --- 프로젝트 구성 요소 임포트 ---
# from app.config.settings import settings # 직접 사용하지 않음 (config 통해 받음)
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# --- 로거 설정 ---
logger = get_logger("ProgressReportNode")

# --- Jinja2 의존성 처리 ---
JINJA2_AVAILABLE = False
template_env: Optional['jinja2.Environment'] = None # 타입 힌트 forward 참조
try:
    import jinja2
    JINJA2_AVAILABLE = True
    logger.info("jinja2 library found.")
    # Jinja2 환경(Environment)은 노드 실행 시 config의 경로를 사용하여 생성
except ImportError:
    logger.error("jinja2 library not installed. Report generation disabled.")
except Exception as e:
    logger.exception(f"Error importing jinja2: {e}")


class ProgressReportNode:
    """
    (Refactored) Jinja2 템플릿을 사용하여 중간 진행 보고서(Markdown)를 생성합니다.
    - 수집된 데이터, 평가 결과, 트렌드 점수 요약.
    - 설정(템플릿 경로/파일명 등)은 상태의 config 딕셔너리에서 로드.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = [
        "fact_urls", "opinion_urls", "final_summary", "decision",
        "trend_scores", "evaluation_metrics", "timestamp", "trace_id", "used_links",
        "processing_stats", "config" # config 추가
    ]
    outputs: List[str] = ["progress_report", "processing_stats", "error_message"]

    def __init__(self):
        """노드 초기화."""
        # Jinja2 환경은 run 메서드에서 config 기반으로 로드 시도
        self.template_env: Optional[jinja2.Environment] = None
        logger.info("ProgressReportNode initialized.")
        if not JINJA2_AVAILABLE:
             logger.error("Report generation disabled due to missing Jinja2 library.")

    def _setup_jinja_env(self, template_dir: str) -> bool:
        """설정된 경로로 Jinja2 환경 로드 시도."""
        if not JINJA2_AVAILABLE: return False
        # 이미 로드되었으면 재사용 (경로가 같다면)
        if self.template_env and self.template_env.loader.searchpath == [template_dir]:
             return True
        try:
            if not os.path.isdir(template_dir):
                logger.error(f"Jinja2 template directory not found: {template_dir}")
                self.template_env = None
                return False

            template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
            # autoescape 설정 (Markdown은 상대적으로 안전하지만 적용)
            self.template_env = jinja2.Environment(
                loader=template_loader,
                autoescape=jinja2.select_autoescape(['html', 'xml', 'md']),
                undefined=jinja2.StrictUndefined # 정의되지 않은 변수 사용 시 오류 발생
            )
            logger.info(f"Jinja2 environment loaded from: {template_dir}")
            return True
        except Exception as e:
            logger.exception(f"Error initializing Jinja2 environment from {template_dir}: {e}")
            self.template_env = None
            return False

    # --- 데이터 포맷팅 헬퍼 함수들 ---
    def _truncate_text(self, text: Optional[str], max_length: int) -> str:
        """텍스트 축약"""
        if not text: return ""
        return text[:max_length - 3] + "..." if len(text) > max_length else text

    def _format_metric(self, value: Any, precision: int = 2) -> str:
         """숫자 메트릭 포맷팅"""
         try: return f"{float(value):.{precision}f}" if isinstance(value, (int, float)) else str(value)
         except (ValueError, TypeError): return str(value)

    def _prepare_url_info(self, url_list: List[Dict[str, Any]], used_links: List[Dict[str, Any]], trace_id: Optional[str]) -> List[Dict[str, Any]]:
        """URL 정보와 사용 상태 병합"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        prepared_list = []
        if not url_list: return prepared_list

        used_link_map: Dict[str, Dict[str, str]] = {
            link.get('url', ''): link for link in used_links if link.get('url')
        }

        for idx, url_info in enumerate(url_list):
            url = url_info.get('url', '')
            if not url: continue
            link_data = used_link_map.get(url)
            # status 필드 확인 (스크래핑 노드에서 설정 가정)
            used_status = '✓ Processed' if link_data and link_data.get('status') == 'processed' else \
                          ('! Error' if link_data and 'error' in link_data.get('status', '') else \
                          ('- Tracked' if link_data else '? Untracked'))
            purpose = link_data.get('purpose', 'N/A') if link_data else 'N/A'

            prepared_list.append({
                'index': idx + 1,
                'url': self._truncate_text(url, 60),
                'used': used_status,
                'purpose': self._truncate_text(purpose, 40)
            })
        logger.debug(f"{log_prefix} Prepared {len(prepared_list)} URL info entries.")
        return prepared_list

    def _prepare_top_trends(self, trend_scores: List[Dict[str, Any]], top_n: int, trace_id: Optional[str]) -> List[Dict[str, Any]]:
        """상위 N개 트렌드 점수 포맷팅"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if not trend_scores:
            logger.warning(f"{log_prefix} No trend scores available for report.")
            return [{'rank': i, 'keyword': 'N/A', 'score': '-'} for i in range(1, top_n + 1)]

        # Node 12에서 이미 점수 내림차순 정렬됨
        top_trends = trend_scores[:top_n]
        prepared_trends = [
            {'rank': idx + 1,
             'keyword': item.get('keyword', 'N/A'),
             'score': self._format_metric(item.get('score', '-'), 1)} # 소수점 1자리
            for idx, item in enumerate(top_trends)
        ]
        logger.debug(f"{log_prefix} Prepared top {len(prepared_trends)} trend scores.")
        # 결과가 N개 미만일 경우 빈칸 채우기
        while len(prepared_trends) < top_n:
             prepared_trends.append({'rank': len(prepared_trends) + 1, 'keyword': 'N/A', 'score': '-'})
        return prepared_trends

    def _determine_next_step(self, decision: Optional[str], trace_id: Optional[str]) -> str:
        """평가 결정에 따른 다음 단계 설명"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        if decision == "proceed": return "Evaluation passed or default action. Proceeding to creative generation."
        elif decision == "research_again": return "Evaluation failed thresholds. Recommend re-running data collection/analysis."
        elif decision == "refine_topic": return "Summary quality okay, but topic coverage low. Recommend refining topic analysis or keywords."
        elif decision: return f"Evaluation Decision: '{decision}'. Next step depends on workflow logic."
        else:
             logger.warning(f"{log_prefix} Evaluation decision missing in state.")
             return "Evaluation decision not available."

    def _prepare_template_data(self, state: ComicState, config: Dict, trace_id: Optional[str]) -> Dict[str, Any]:
        """Jinja2 템플릿 컨텍스트 데이터 준비"""
        log_prefix = f"[{trace_id}]" if trace_id else ""
        logger.debug(f"{log_prefix} Preparing data for progress report template...")

        timestamp_str = state.timestamp or datetime.now(timezone.utc).isoformat()
        formatted_timestamp = timestamp_str # 기본값
        try: formatted_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S %Z")
        except ValueError: pass # 파싱 실패 시 원래 문자열 사용

        used_links = state.used_links or []
        fact_urls_info = self._prepare_url_info(state.fact_urls or [], used_links, trace_id)
        opinion_urls_info = self._prepare_url_info(state.opinion_urls or [], used_links, trace_id)

        final_summary = state.final_summary
        final_summary_preview = self._truncate_text(final_summary, 250) or "[Summary not available]"
        decision = state.decision or "[Decision not available]"

        top_n = config.get("trends_report_top_n", 3) # config에서 가져오기
        trend_scores_top_n = self._prepare_top_trends(state.trend_scores or [], top_n, trace_id)

        evaluation_metrics = state.evaluation_metrics or {}
        topic_coverage_val = evaluation_metrics.get("topic_coverage", 0.0)
        topic_match_percent = f"{topic_coverage_val * 100:.1f}%" if isinstance(topic_coverage_val, float) else "-"
        rouge_l = self._format_metric(evaluation_metrics.get("rouge_l", "-"), 3)
        bertscore = self._format_metric(evaluation_metrics.get("bert_score", "-"), 3)

        next_step = self._determine_next_step(state.decision, trace_id)

        processing_stats = state.processing_stats or {}
        # 이전 노드에서 저장한 시간 값 사용 (키 이름 일치 확인 필요)
        elapsed_summary = self._format_metric(processing_stats.get("synthesis_summarizer_node_time", 0), 1)
        elapsed_trend = self._format_metric(processing_stats.get("trend_analyzer_node_time", 0), 1)
        # 전체 노드 시간 합계 (선택적)
        total_elapsed = sum(v for k, v in processing_stats.items() if k.endswith('_time') and isinstance(v, (int, float)))

        context = {
            "timestamp": formatted_timestamp,
            "trace_id": state.trace_id,
            "fact_urls_info": fact_urls_info,
            "opinion_urls_info": opinion_urls_info,
            "final_summary_preview": final_summary_preview,
            "decision": decision,
            "trend_scores_top_n": trend_scores_top_n,
            "topic_match": topic_match_percent,
            "rouge_l": rouge_l,
            "bertscore": bertscore,
            "next_step": next_step,
            "elapsed_summary": elapsed_summary,
            "elapsed_trend": elapsed_trend,
            "total_elapsed": self._format_metric(total_elapsed, 1) # 선택적 추가
        }
        logger.debug(f"{log_prefix} Template data preparation complete.")
        return context

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """Jinja2 템플릿을 렌더링하여 진행 보고서 생성"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing ProgressReportNode...")

        config = state.config or {}
        processing_stats = state.processing_stats or {}
        report_content = f"# Report Generation Failed\n\nJinja2 library not available." # 기본 오류 메시지
        error_message: Optional[str] = None

        # Jinja2 라이브러리 가용성 확인
        if not JINJA2_AVAILABLE:
            error_message = "Jinja2 library not available. Cannot generate report."
            logger.error(f"{log_prefix} {error_message}")
        else:
            # 설정에서 템플릿 경로 및 파일명 로드
            template_dir = config.get("template_dir")
            template_name = config.get("progress_report_template_a_filename")

            if not template_dir or not template_name:
                error_message = "Template directory or filename not found in config."
                logger.error(f"{log_prefix} {error_message}")
                report_content = f"# Report Generation Failed\n\n{error_message}"
            # Jinja2 환경 설정 (실행 시점에 경로 기반으로 로드)
            elif self._setup_jinja_env(template_dir):
                logger.info(f"{log_prefix} Generating progress report using template: {template_name}")
                try:
                    template = self.template_env.get_template(template_name)
                    template_data = self._prepare_template_data(state, config, state.trace_id)
                    report_content = template.render(**template_data)
                    logger.info(f"{log_prefix} Progress report generated successfully (Length: {len(report_content)} chars).")
                except jinja2.TemplateNotFound:
                    error_message = f"Template file '{template_name}' not found in directory '{template_dir}'."
                    logger.error(f"{log_prefix} {error_message}")
                    report_content = f"# Report Generation Failed\n\n{error_message}"
                except Exception as e:
                    error_message = f"Failed to prepare data or render template: {str(e)}"
                    logger.exception(f"{log_prefix} {error_message}")
                    report_content = f"# Report Generation Error\n\nAn unexpected error occurred: {str(e)}"
            else:
                # _setup_jinja_env 에서 이미 오류 로깅됨
                error_message = f"Failed to initialize Jinja2 environment from directory: {template_dir}"
                report_content = f"# Report Generation Failed\n\n{error_message}"


        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['progress_report_node_time'] = node_processing_time
        logger.info(f"{log_prefix} ProgressReportNode finished in {node_processing_time:.2f} seconds.")

        # TODO: LangSmith 로깅

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            "progress_report": report_content, # 성공/실패 시 모두 생성된 내용을 담음
            "processing_stats": processing_stats,
            "error_message": error_message # 생성 중 발생한 오류 메시지
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}