# app/nodes/07_filter_node.py (Refactored)

import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
from datetime import datetime, timezone

# 프로젝트 구성 요소 임포트
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState
from app.config.settings import settings
from app.tools.analysis.language_detector import LanguageDetectionTool
from app.services.spam_detector import SpamDetectionService # Assuming this is the correct import path
from app.tools.analysis.text_clusterer import TextClusteringTool

logger = get_logger(__name__)

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
    [... existing docstring ...]
    """
    inputs: List[str] = ["opinions_raw", "trace_id", "comic_id", "config"] # MODIFIED: Added comic_id
    outputs: List[str] = ["opinions_clean", "node7_processing_stats", "error_message"]

    def __init__(
        self,
        language_tool: LanguageDetectionTool,
        spam_service: SpamDetectionService,
        clustering_tool: TextClusteringTool
    ):
        self.language_tool = language_tool
        self.spam_service = spam_service
        self.clustering_tool = clustering_tool

        self.simhash_threshold = settings.SIMHASH_THRESHOLD
        self.simhash_width = settings.SIMHASH_TOKEN_WIDTH
        if SIMHASH_AVAILABLE:
             logger.info(f"SimHash configured (Threshold: {self.simhash_threshold}, Width: {self.simhash_width}).")

        if not self.language_tool: logger.warning("LanguageDetectionTool not injected!")
        if not self.spam_service: logger.warning("SpamDetectionService not injected!")
        if not self.clustering_tool: logger.warning("TextClusteringTool not injected!")
        logger.info("FilterNode initialized.")

    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict): # MODIFIED: Added extra_log_data
        self.allowed_languages = config.get('language_filter', settings.LANGUAGE_FILTER)
        # Ensure allowed_languages is a list or set
        if not isinstance(self.allowed_languages, (list, set)):
            logger.warning(f"language_filter is not a list or set: {self.allowed_languages}. Using default.", extra=extra_log_data) # MODIFIED
            self.allowed_languages = settings.LANGUAGE_FILTER
        logger.debug(f"Runtime config loaded. Allowed languages: {self.allowed_languages}", extra=extra_log_data) # MODIFIED

    def _compute_simhash(self, text: str) -> Optional[Simhash]:
        if not SIMHASH_AVAILABLE or not text: return None
        try:
            tokens = re.findall(r'\b\w+\b', text.lower())
            if not tokens: return None
            return Simhash(tokens, f=self.simhash_width)
        except Exception as e:
            logger.error(f"Error computing Simhash: {e}", exc_info=True) # Keep detailed log
            return None

    def _is_near_duplicate(self, current_simhash: Simhash, existing_hashes: Set[int]) -> bool:
        if not SIMHASH_AVAILABLE or current_simhash is None: return False
        current_value = current_simhash.value
        for existing_value in existing_hashes:
            # Use Simhash's distance method if available, otherwise manual Hamming
            distance = current_simhash.distance(Simhash(value=existing_value, f=self.simhash_width))
            # distance = bin(current_value ^ existing_value).count('1') # Manual Hamming
            if distance <= self.simhash_threshold:
                return True
        return False

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """의견 필터링 및 클러스터링 파이프라인 실행"""
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

        opinions_raw = getattr(state, 'opinions_raw', []) # Safe access
        config = getattr(state, 'config', {}) or {}

        # --- ADDED: Input Validation ---
        if not opinions_raw:
            error_message = "No raw opinions provided for filtering."
            logger.warning(error_message, extra=extra_log_data) # Warning as it might be valid
            end_time = datetime.now(timezone.utc)
            node7_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "opinions_clean": [],
                "node7_processing_stats": node7_processing_stats,
                "error_message": error_message # Pass warning/error
            }
            # --- ADDED: End Logging (Early Exit) ---
            logger.debug(f"Returning updates (no opinions):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (No Opinions) --- (Elapsed: {node7_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        # --------------------------

        logger.info(f"Starting filtering for {len(opinions_raw)} raw opinions...", extra=extra_log_data)

        current_opinions = opinions_raw
        initial_count = len(current_opinions)
        stats_log = {"initial": initial_count}
        error_messages = [] # Collect errors from steps

        try:
            # 1. 언어 필터링
            if self.language_tool:
                filtered_by_lang = []
                lang_errors = 0
                for opinion in current_opinions:
                    text = opinion.get('text', '')
                    try:
                        # Pass trace_id/comic_id if tool supports it
                        lang_code = self.language_tool.detect(text, trace_id, comic_id) if text else 'und'
                        opinion['language'] = lang_code
                        if lang_code in self.allowed_languages:
                            filtered_by_lang.append(opinion)
                        # else: logger.debug("Opinion filtered (Language)", extra={'url': opinion.get('url'), **extra_log_data})
                    except Exception as lang_err:
                        logger.error(f"Language detection failed for opinion: {lang_err}", exc_info=True, extra={**extra_log_data, 'url': opinion.get('url')})
                        lang_errors += 1
                        opinion['language'] = 'error' # Mark as error
                        filtered_by_lang.append(opinion) # Keep item despite error? Or filter out? Keeping for now.
                if lang_errors > 0: error_messages.append(f"Language detection failed for {lang_errors} items.")
                current_opinions = filtered_by_lang
                stats_log['after_lang'] = len(current_opinions)
            else:
                logger.warning("Language detection tool unavailable. Skipping.", extra=extra_log_data)
                for opinion in current_opinions: opinion.setdefault('language', 'und')

            # 2. 스팸 필터링
            if self.spam_service:
                filtered_by_spam = []
                spam_errors = 0
                for opinion in current_opinions:
                    text = opinion.get('text', '')
                    try:
                        # Pass trace_id/comic_id if service supports it
                        if not self.spam_service.is_spam(text, trace_id, comic_id):
                            filtered_by_spam.append(opinion)
                        # else: logger.debug("Opinion filtered (Spam)", extra={'url': opinion.get('url'), **extra_log_data})
                    except Exception as spam_err:
                        logger.error(f"Spam detection failed for opinion: {spam_err}", exc_info=True, extra={**extra_log_data, 'url': opinion.get('url')})
                        spam_errors += 1
                        filtered_by_spam.append(opinion) # Keep item despite error
                if spam_errors > 0: error_messages.append(f"Spam detection failed for {spam_errors} items.")
                current_opinions = filtered_by_spam
                stats_log['after_spam'] = len(current_opinions)
            else:
                logger.warning("Spam detection service unavailable. Skipping.", extra=extra_log_data)

            # 3. 유사 중복 제거 (Simhash)
            if SIMHASH_AVAILABLE:
                filtered_by_dupe = []
                seen_simhashes_values: Set[int] = set()
                logger.info(f"Performing Simhash near-duplicate detection (Threshold: {self.simhash_threshold})...", extra=extra_log_data)
                skipped_computation = 0
                duplicate_count = 0

                for opinion in current_opinions:
                    text = opinion.get('text', '')
                    simhash_obj = self._compute_simhash(text) # Already has error handling

                    if simhash_obj:
                        try:
                            if not self._is_near_duplicate(simhash_obj, seen_simhashes_values):
                                filtered_by_dupe.append(opinion)
                                seen_simhashes_values.add(simhash_obj.value)
                            else:
                                duplicate_count += 1
                                # logger.debug("Opinion filtered (Near-duplicate)", extra={'url': opinion.get('url'), **extra_log_data})
                        except Exception as dupe_err:
                            logger.error(f"Error checking Simhash duplicate: {dupe_err}", exc_info=True, extra={**extra_log_data, 'url': opinion.get('url')})
                            filtered_by_dupe.append(opinion) # Keep on error
                    else:
                        filtered_by_dupe.append(opinion) # Keep if computation failed
                        skipped_computation += 1

                if skipped_computation > 0:
                     logger.warning(f"Skipped Simhash computation for {skipped_computation} items due to errors.", extra=extra_log_data)
                current_opinions = filtered_by_dupe
                stats_log['after_simhash'] = len(current_opinions)
                logger.debug(f"Simhash removed {duplicate_count} near-duplicates.", extra=extra_log_data)
            else:
                logger.warning("Simhash library unavailable. Skipping near-duplicate detection.", extra=extra_log_data)


            # 4. 클러스터링 및 대표 의견 선정
            opinions_clean: List[Dict[str, Any]] = []
            if self.clustering_tool and current_opinions:
                logger.info("Performing text clustering...", extra=extra_log_data)
                try:
                    # Pass trace_id/comic_id if tool supports it
                    # Assume cluster_texts adds 'cluster_id' and 'is_representative'
                    opinions_clean = self.clustering_tool.cluster_texts(current_opinions, trace_id, comic_id)
                    cluster_summary = Counter(op.get('cluster_id', -1) for op in opinions_clean)
                    rep_count = sum(1 for op in opinions_clean if op.get('is_representative'))
                    stats_log['after_clustering'] = len(opinions_clean)
                    logger.info(f"Clustering complete. Clusters: {len(cluster_summary)}, Representatives: {rep_count}.", extra=extra_log_data)
                    logger.debug(f"Cluster distribution: {dict(cluster_summary)}", extra=extra_log_data)
                except Exception as cluster_err:
                     logger.exception("Error during text clustering. Proceeding without clustering.", extra=extra_log_data) # Use exception
                     error_messages.append(f"Clustering failed: {cluster_err}")
                     for op in current_opinions: op.update({'cluster_id': -1, 'is_representative': True}) # Default values
                     opinions_clean = current_opinions
                     stats_log['after_clustering'] = len(opinions_clean)
            else:
                if not current_opinions:
                    logger.info("No opinions left to cluster.", extra=extra_log_data)
                else: # Tool unavailable
                    logger.warning("Clustering skipped (tool unavailable). Marking all as representative.", extra=extra_log_data)
                for op in current_opinions: op.update({'cluster_id': 0, 'is_representative': True}) # Default values
                opinions_clean = current_opinions
                stats_log['after_clustering'] = len(opinions_clean)

        except Exception as e:
            # Catch unexpected errors in the main filtering flow
            logger.exception("Unexpected error during opinion filtering process.", extra=extra_log_data)
            error_messages.append(f"Unexpected filtering error: {e}")
            # Use whatever opinions were processed last, or default to empty
            opinions_clean = current_opinions if 'current_opinions' in locals() else []
            # Ensure default cluster/representative flags if error occurred before clustering
            for op in opinions_clean:
                op.setdefault('cluster_id', -1)
                op.setdefault('is_representative', True)

        logger.info(f"Filtering stats: {stats_log}", extra=extra_log_data)

        end_time = datetime.now(timezone.utc)
        node7_processing_stats = (end_time - start_time).total_seconds()
        final_error_message = "; ".join(error_messages) if error_messages else None

        # --- 상태 업데이트 준비 ---
        update_data = {
            "opinions_clean": opinions_clean,
            "node7_processing_stats": node7_processing_stats,
            "error_message": final_error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if final_error_message else logger.info
        log_level(f"Filtering result: {len(opinions_clean)} cleaned opinions. Errors: {final_error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node7_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}