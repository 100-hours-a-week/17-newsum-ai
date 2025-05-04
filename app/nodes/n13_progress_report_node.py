# app/nodes/13_progress_report_node.py (Refactored)

import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set, Tuple

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

JINJA2_AVAILABLE = False
template_env_cache: Optional[Dict[str, 'jinja2.Environment']] = {}
try:
    import jinja2
    JINJA2_AVAILABLE = True
    logger.info("jinja2 library found.")
except ImportError:
    jinja2 = None
    logger.error("jinja2 library not installed. Report generation disabled.")
except Exception as e:
    jinja2 = None
    logger.exception(f"Error importing jinja2: {e}")

class ProgressReportNode:
    """
    Jinja2 템플릿을 사용하여 중간 진행 보고서(Markdown, Template A)를 생성합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = [ # MODIFIED: Added comic_id
        "fact_urls", "opinion_urls", "final_summary", "decision", "trend_scores",
        "evaluation_metrics", "timestamp", "trace_id", "comic_id", "used_links",
        # Removed node13 processing stats from input, it's an output
        "config"
    ]
    outputs: List[str] = ["progress_report", "node13_processing_stats", "error_message"]

    def __init__(self):
        logger.info("ProgressReportNode initialized.")
        if not JINJA2_AVAILABLE:
             logger.error("Report generation disabled due to missing Jinja2 library.")

    # --- MODIFIED: Added extra_log_data argument ---
    def _setup_jinja_env(self, template_dir: str, trace_id: Optional[str], comic_id: Optional[str]) -> Optional['jinja2.Environment']:
        """설정된 경로로 Jinja2 환경 로드 또는 캐시된 환경 반환"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not JINJA2_AVAILABLE or not template_dir: return None

        if template_dir in template_env_cache:
            logger.debug("Using cached Jinja2 environment", extra=extra_log_data) # MODIFIED
            return template_env_cache[template_dir]

        try:
            if not os.path.isdir(template_dir):
                logger.error(f"Jinja2 template directory not found: {template_dir}", extra=extra_log_data) # MODIFIED
                return None

            loader = jinja2.FileSystemLoader(searchpath=template_dir)
            env = jinja2.Environment(
                loader=loader,
                autoescape=jinja2.select_autoescape(['html', 'xml', 'md']),
                undefined=jinja2.StrictUndefined
            )
            logger.info(f"Jinja2 environment loaded from: {template_dir}", extra=extra_log_data) # MODIFIED
            template_env_cache[template_dir] = env
            return env
        except Exception as e:
            logger.exception(f"Error initializing Jinja2 environment from {template_dir}", extra=extra_log_data) # MODIFIED
            return None

    def _truncate_text(self, text: Optional[str], max_length: int) -> str:
        if not text: return ""
        return text[:max_length - 3] + "..." if len(text) > max_length else text

    def _format_metric(self, value: Any, precision: int = 2) -> str:
         try: return f"{float(value):.{precision}f}" if isinstance(value, (int, float)) else str(value)
         except (ValueError, TypeError): return str(value)

    # --- MODIFIED: Added extra_log_data argument ---
    def _prepare_url_info(self, url_list: List[Dict[str, Any]], used_links: List[Dict[str, Any]], url_type: str, trace_id: Optional[str], comic_id: Optional[str]) -> List[Dict[str, Any]]:
        """URL 목록과 사용 상태를 병합하여 보고서용 데이터 생성"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        prepared_list = []
        if not url_list or not isinstance(url_list, list): # Added type check
            logger.debug(f"No valid {url_type} URLs provided.", extra=extra_log_data) # MODIFIED
            return prepared_list

        used_link_map: Dict[str, Dict[str, Any]] = {
            link.get('url', ''): link for link in used_links if isinstance(link, dict) and link.get('url') # Ensure link is dict
        }

        for idx, url_info in enumerate(url_list):
             # Ensure url_info is a dict
             if not isinstance(url_info, dict): continue
             url = url_info.get('url', '')
             if not url or not isinstance(url, str): continue # Added type check

             link_data = used_link_map.get(url)
             status = '?'
             # Use purpose from url_info first if available (might have source/keyword)
             purpose = url_info.get('purpose') or url_info.get('search_keyword') or 'N/A'

             if link_data and isinstance(link_data, dict):
                 link_status = link_data.get('status', 'tracked')
                 # Prefer purpose from used_links if more detailed
                 purpose = link_data.get('purpose', purpose)

                 if link_status == 'processed': status = '✓'
                 elif 'failed' in str(link_status): status = '!' # Use str() for safety
                 elif link_status == 'collected': status = '-'
                 elif link_status == 'context_used': status = 'C'
                 else: status = '-' # Default symbol for tracked/other statuses

             prepared_list.append({
                'index': idx + 1,
                'url': self._truncate_text(url, 60),
                'status_symbol': status,
                'purpose': self._truncate_text(purpose, 40) # Use potentially updated purpose
             })
        logger.debug(f"Prepared {len(prepared_list)} {url_type} URL info entries.", extra=extra_log_data) # MODIFIED
        return prepared_list

    # --- MODIFIED: Added extra_log_data argument ---
    def _prepare_top_trends(self, trend_scores: List[Dict[str, Any]], top_n: int, trace_id: Optional[str], comic_id: Optional[str]) -> List[Dict[str, Any]]:
        """상위 N개 트렌드 점수 포맷팅"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if not trend_scores or not isinstance(trend_scores, list): # Added type check
            logger.warning("No valid trend scores available for report.", extra=extra_log_data) # MODIFIED
            return [{'rank': i, 'keyword': 'N/A', 'score': '-'} for i in range(1, top_n + 1)]

        # Ensure items are dicts with expected keys before sorting/accessing
        valid_scores = [item for item in trend_scores if isinstance(item, dict) and 'keyword' in item and 'score' in item]
        # Sorting was done in Node 12, just take top N
        top_trends = valid_scores[:top_n]

        prepared_trends = [
            {'rank': idx + 1,
             'keyword': item.get('keyword', 'N/A'),
             'score': self._format_metric(item.get('score', '-'), 1)}
            for idx, item in enumerate(top_trends)
        ]
        while len(prepared_trends) < top_n:
             prepared_trends.append({'rank': len(prepared_trends) + 1, 'keyword': 'N/A', 'score': '-'})
        logger.debug(f"Prepared top {len(prepared_trends)} trend scores.", extra=extra_log_data) # MODIFIED
        return prepared_trends

    # --- MODIFIED: Added extra_log_data argument ---
    def _determine_next_step(self, decision: Optional[str], trace_id: Optional[str], comic_id: Optional[str]) -> str:
        """결정에 따른 다음 단계 설명 생성"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        if decision == "proceed": return "Evaluation passed or default. Proceeding to creative generation."
        elif decision == "research_again": return "**Action Recommended**: Evaluation failed thresholds. Re-run data collection/analysis."
        elif decision == "refine_topic": return "**Action Recommended**: Summary quality okay, topic coverage low. Refine topic/keywords."
        elif isinstance(decision, str): # Handle other potential string values
             return f"Evaluation Decision: '{decision}'. Further action depends on logic."
        else:
             logger.warning("Evaluation decision missing or invalid in state.", extra=extra_log_data) # MODIFIED
             return "Evaluation decision not available."

    # --- MODIFIED: Added extra_log_data argument ---
    def _prepare_template_data(self, state: ComicState, config: Dict, trace_id: Optional[str], comic_id: Optional[str]) -> Dict[str, Any]:
        """Jinja2 템플릿 컨텍스트 데이터 준비"""
        extra_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        logger.debug("Preparing data for progress report template...", extra=extra_log_data) # MODIFIED

        timestamp_str = getattr(state, 'timestamp', None) or datetime.now(timezone.utc).isoformat() # Safe access
        formatted_timestamp = timestamp_str
        try: formatted_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S %Z")
        except ValueError: pass # Keep ISO format if parsing fails

        used_links = getattr(state, 'used_links', []) or []
        fact_urls = getattr(state, 'fact_urls', []) or []
        opinion_urls = getattr(state, 'opinion_urls', []) or []

        # Pass IDs
        fact_urls_info = self._prepare_url_info(fact_urls, used_links, "Fact", trace_id, comic_id)
        opinion_urls_info = self._prepare_url_info(opinion_urls, used_links, "Opinion", trace_id, comic_id)

        final_summary = getattr(state, 'final_summary', None) # Safe access
        final_summary_preview = self._truncate_text(final_summary, 300) if final_summary else "[Summary not generated or failed]"

        decision = getattr(state, 'decision', None) # Safe access
        # Pass IDs
        next_step = self._determine_next_step(decision, trace_id, comic_id)

        trend_scores = getattr(state, 'trend_scores', []) or []
        top_n = config.get("trends_report_top_n", settings.DEFAULT_TRENDS_REPORT_TOP_N)
        # Pass IDs
        trend_scores_top_n = self._prepare_top_trends(trend_scores, top_n, trace_id, comic_id)

        evaluation_metrics = getattr(state, 'evaluation_metrics', {}) or {} # Safe access, ensure dict
        topic_coverage = self._format_metric(evaluation_metrics.get("topic_coverage", 0.0) * 100, 1) + "%"
        rouge_l = self._format_metric(evaluation_metrics.get("rouge_l", "-"), 3)
        bertscore = self._format_metric(evaluation_metrics.get("bert_score", "-"), 3)

        node_times = {}
        total_elapsed = 0.0
        for i in range(1, 13): # Only up to Node 12 times are available when Node 13 runs
            field_name = f"node{i}_processing_stats"
            time_val = getattr(state, field_name, None) # Safe access
            if isinstance(time_val, (int, float)):
                # Map node number to a more descriptive name if possible (simple version here)
                node_map = {1: "Init", 2: "Topic", 3: "NewsColl", 4: "OpColl", 5: "NewsScrp", 6: "OpScrp",
                           7: "Filter", 8: "NewsSum", 9: "OpSum", 10: "Synth", 11: "Eval", 12: "Trend"}
                node_display_name = f"N{i}({node_map.get(i, '?')})"
                node_times[node_display_name] = self._format_metric(time_val, 1)
                total_elapsed += time_val
        # Add self time (approximate, as full run isn't finished yet)
        current_elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))).total_seconds()

        context = {
            "timestamp": formatted_timestamp,
            "trace_id": trace_id, # Use trace_id from state/input
            "comic_id": comic_id, # Add comic_id
            "fact_urls_info": fact_urls_info,
            "opinion_urls_info": opinion_urls_info,
            "final_summary_preview": final_summary_preview,
            "decision": decision or "[Decision not available]", # Handle None case
            "trend_scores_top_n": trend_scores_top_n,
            "topic_match_percent": topic_coverage,
            "rouge_l_score": rouge_l,
            "bertscore_f1": bertscore,
            "next_step": next_step,
            "node_processing_times": node_times,
            "total_elapsed_time": self._format_metric(total_elapsed, 1) # Use total from *previous* nodes
        }
        logger.debug("Template data preparation complete.", extra=extra_log_data) # MODIFIED
        return context

    def _save_report_to_file(self, report_content: str, comic_id: str, filename_prefix: str, extra_log_data: Dict):
        """보고서 내용을 로컬 파일로 저장"""
        if not report_content or not comic_id:
            logger.warning("Report content or comic_id missing, skipping file save.", extra=extra_log_data)
            return False
        try:
            # 설정에서 보고서 저장 디렉토리 가져오기
            output_dir = settings.REPORT_OUTPUT_DIR
            if not output_dir:
                logger.warning("REPORT_OUTPUT_DIR not configured in settings, skipping file save.",
                               extra=extra_log_data)
                return False

            # 디렉토리 존재 확인 및 생성 (시작 시 생성하지만 여기서 한번 더 확인)
            os.makedirs(output_dir, exist_ok=True)

            # 파일 경로 조합
            file_name = f"{filename_prefix}_{comic_id}.md"
            file_path = os.path.join(output_dir, file_name)

            # 파일 쓰기 (UTF-8 인코딩 명시)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            logger.info(f"Report successfully saved to: {file_path}", extra=extra_log_data)
            return True
        except Exception as e:
            logger.error(f"Failed to save report to file '{file_path}': {e}", exc_info=True, extra=extra_log_data)
            return False

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """Jinja2 템플릿을 렌더링하여 진행 보고서 생성"""
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

        config = getattr(state, 'config', {}) or {}
        report_content = "# Report Generation Failed\n\nInitial error or missing dependencies."
        error_message: Optional[str] = None

        if not JINJA2_AVAILABLE:
            error_message = "Jinja2 library not available."
            logger.error(error_message, extra=extra_log_data) # MODIFIED
        else:
            template_dir = config.get("template_dir", settings.DEFAULT_TEMPLATE_DIR)
            template_name = config.get("progress_report_template_a_filename", settings.DEFAULT_TEMPLATE_A_FILENAME) # Use Template A

            if not template_dir or not template_name:
                error_message = "Template directory or Template A filename not configured."
                logger.error(error_message, extra=extra_log_data) # MODIFIED
            else:
                # Pass IDs
                jinja_env = self._setup_jinja_env(template_dir, trace_id, comic_id)
                if jinja_env:
                    logger.info(f"Generating progress report using template: {template_name}", extra=extra_log_data) # MODIFIED
                    try:
                        template = jinja_env.get_template(template_name)
                        # Pass IDs
                        template_data = self._prepare_template_data(state, config, trace_id, comic_id)
                        report_content = template.render(**template_data)
                        logger.info("Progress report generated successfully.", extra=extra_log_data) # MODIFIED
                        # --- *** 보고서 파일 저장 시도 *** ---
                        report_saved = self._save_report_to_file(
                            report_content,
                            comic_id,
                            "13 node report",  # 파일명 접두사
                            extra_log_data
                        )
                        # ---------------------------------
                    except jinja2.TemplateNotFound:
                        error_message = f"Template file '{template_name}' not found in '{template_dir}'."
                        logger.error(error_message, extra=extra_log_data) # MODIFIED
                    except Exception as e:
                        error_message = f"Failed to prepare/render template '{template_name}': {str(e)}"
                        logger.exception("Template rendering error.", extra=extra_log_data) # MODIFIED
                else:
                    error_message = f"Failed to initialize Jinja2 environment from '{template_dir}'."
                    logger.error(error_message, extra=extra_log_data) # MODIFIED

        if error_message:
             report_content = f"# Report Generation Failed\n\n{error_message}"

        end_time = datetime.now(timezone.utc)
        node13_processing_stats = (end_time - start_time).total_seconds()

        update_data: Dict[str, Any] = {
            "progress_report": report_content,
            "node13_processing_stats": node13_processing_stats,
            "error_message": error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if error_message else logger.info
        log_level(f"Report generation result: {'Failed' if error_message else 'Success'}. Errors: {error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node13_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}