# app/nodes/18_translator_node.py (Refactored)

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import aiohttp

# 프로젝트 구성 요소 임포트
from app.config.settings import settings
from app.services.papago_translation_service import PapagoTranslationService
from app.utils.logger import get_logger, summarize_for_logging # MODIFIED: Added summarize_for_logging
from app.workflows.state import ComicState

logger = get_logger(__name__)

class TranslatorNode:
    """
    (선택적) 시나리오 대사를 번역합니다.
    [... existing docstring ...]
    """
    inputs: List[str] = ["scenarios", "config", "trace_id", "comic_id"] # MODIFIED: Added comic_id
    outputs: List[str] = ["translated_text", "node18_processing_stats", "error_message"]

    def __init__(self, translator_client: PapagoTranslationService):
        if not translator_client: raise ValueError("PapagoTranslationService is required.")
        self.translator = translator_client
        logger.info("TranslatorNode initialized.")

    # --- MODIFIED: Added extra_log_data argument ---
    def _load_runtime_config(self, config: Dict[str, Any], extra_log_data: Dict):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.translation_enabled = config.get("translation_enabled", settings.ENABLE_TRANSLATION)
        self.target_lang = config.get("target_language", settings.DEFAULT_TARGET_LANG)
        self.source_lang = config.get("source_language")
        if not self.source_lang:
             self.source_lang = "ko" if self.target_lang == "en" else "en"
             logger.warning(f"Source language not specified, assuming '{self.source_lang}' based on target '{self.target_lang}'.", extra=extra_log_data) # MODIFIED
        self.source_lang = self.source_lang or settings.DEFAULT_SOURCE_LANG
        self.concurrency_limit = int(config.get('translator_concurrency', settings.DEFAULT_TRANSLATOR_CONCURRENCY))
        self.http_timeout = config.get('http_timeout', settings.DEFAULT_HTTP_TIMEOUT)

        logger.debug(f"Runtime config loaded. Enabled: {self.translation_enabled}, Source: {self.source_lang}, Target: {self.target_lang}", extra=extra_log_data) # MODIFIED
        logger.debug(f"Concurrency: {self.concurrency_limit}, Timeout: {self.http_timeout}", extra=extra_log_data) # MODIFIED

    # --- MODIFIED: Added comic_id argument ---
    async def _translate_dialogue_wrapper(
        self, dialogue: str, session: aiohttp.ClientSession, trace_id: Optional[str], comic_id: Optional[str]
        ) -> Optional[str]:
        """단일 대사 번역. Returns translated text or None on failure."""
        if not dialogue: return ""
        trans_log_data = {'trace_id': trace_id, 'comic_id': comic_id} # MODIFIED
        try:
            # Pass IDs if service supports them
            translated = await self.translator.translate(
                dialogue, self.source_lang, self.target_lang, session, trace_id #, comic_id
            )
            if translated is None:
                 logger.warning(f"Translation returned None for '{dialogue[:30]}...'.", extra=trans_log_data) # MODIFIED
            return translated # Could be None
        except Exception as e:
            # Service internal retries failed
            logger.error(f"Translation failed for '{dialogue[:30]}...': {e}", exc_info=True, extra=trans_log_data) # MODIFIED
            return None # Indicate failure

    async def run(self, state: ComicState) -> Dict[str, Any]:
        """번역 프로세스 실행 (활성화된 경우)"""
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

        scenarios = getattr(state, 'scenarios', None) # Safe access
        config = getattr(state, 'config', {}) or {}
        error_message: Optional[str] = None
        translated_text_output: Optional[List[Dict[str, Any]]] = None # Default to None if disabled

        # --- MODIFIED: Pass log data ---
        self._load_runtime_config(config, extra_log_data)
        # --------------------------

        # --- ADDED: Check if enabled ---
        if not self.translation_enabled or not self.translator.is_enabled:
            status = "disabled in config" if not self.translation_enabled else "service unavailable"
            logger.info(f"Translation {status}. Skipping.", extra=extra_log_data)
            end_time = datetime.now(timezone.utc)
            node18_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "translated_text": None, # Explicitly None when skipped
                "node18_processing_stats": node18_processing_stats,
                "error_message": None # Not an error, just skipped
            }
            # --- ADDED: End Logging (Skipped Case) ---
            logger.debug(f"Returning updates (skipped):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Skipped) --- (Elapsed: {node18_processing_stats:.2f}s)", extra=extra_log_data)
            # ------------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # ---------------------------------

        # --- ADDED: Input Validation ---
        if not scenarios or not isinstance(scenarios, list) or len(scenarios) != 4 or not all(isinstance(p, dict) for p in scenarios):
            error_message = "Invalid or missing 4-panel 'scenarios' list for translation."
            logger.error(error_message, extra=extra_log_data)
            translated_text_output = [] # Return empty list on error
            end_time = datetime.now(timezone.utc)
            node18_processing_stats = (end_time - start_time).total_seconds()
            update_data = {
                "translated_text": translated_text_output,
                "node18_processing_stats": node18_processing_stats,
                "error_message": error_message
            }
             # --- ADDED: End Logging (Error Case) ---
            logger.debug(f"Returning updates (input error):\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
            logger.info(f"--- Finished {node_class_name} (Input Error) --- (Elapsed: {node18_processing_stats:.2f}s)", extra=extra_log_data)
            # ----------------------------------------
            valid_keys = set(ComicState.model_fields.keys())
            return {k: v for k, v in update_data.items() if k in valid_keys}
        # -------------------------------

        logger.info(f"Starting translation of {len(scenarios)} panels from '{self.source_lang}' to '{self.target_lang}'...", extra=extra_log_data)

        translated_results: List[Optional[Dict[str, Any]]] = [None] * len(scenarios)
        tasks = []
        task_errors: List[str] = []

        try:
            timeout = aiohttp.ClientTimeout(total=self.http_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                semaphore = asyncio.Semaphore(self.concurrency_limit)
                for i, panel in enumerate(scenarios):
                    original_dialogue = panel.get('dialogue', '')
                    scene_num = panel.get('scene', i + 1)

                    if original_dialogue and isinstance(original_dialogue, str): # Ensure it's a string
                        async def translate_task(dialogue, panel_idx, scene_n):
                            async with semaphore:
                                # Pass comic_id
                                translated = await self._translate_dialogue_wrapper(dialogue, session, trace_id, comic_id)
                                return panel_idx, scene_n, dialogue, translated # Return enough info

                        tasks.append(translate_task(original_dialogue, i, scene_num))
                    else:
                        # Store result for panels with no dialogue immediately
                        translated_results[i] = {
                            "scene": scene_num, "original_dialogue": "", "translated_dialogue": ""
                        }

                if tasks:
                    # MODIFIED: Use return_exceptions=True
                    gather_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # MODIFIED: Process results carefully
                    for result in gather_results:
                        if isinstance(result, Exception):
                            # Log exception details
                            err_msg = f"A translation task failed: {result}"
                            logger.error(err_msg, exc_info=result, extra=extra_log_data)
                            task_errors.append(f"Translation Task Failed: {result}") # Summary
                            # How to know which panel failed? We don't easily here.
                            # Mark overall error, but can't mark specific panel as failed easily.
                        elif isinstance(result, tuple) and len(result) == 4:
                            panel_idx, scene_n, orig_dialogue, trans_dialogue = result
                            # Store result, trans_dialogue might be None if wrapper failed
                            translated_results[panel_idx] = {
                                "scene": scene_n,
                                "original_dialogue": orig_dialogue,
                                "translated_dialogue": trans_dialogue
                            }
                            if trans_dialogue is None: # Explicitly record wrapper failure
                                task_errors.append(f"Panel {scene_n} translation failed (returned None).")
                        else:
                            # Unexpected result type from gather
                            logger.warning(f"Unexpected result type from translation task: {type(result)}", extra=extra_log_data)
                            task_errors.append(f"Unexpected translation result type: {type(result)}")
                else:
                     logger.info("No dialogues found needing translation.", extra=extra_log_data)

        except Exception as e:
            # Catch errors setting up session or gather
            logger.exception("Error during translation task setup or execution.", extra=extra_log_data) # Use exception
            error_message = f"Unexpected error during translation: {e}"
            task_errors.append(error_message)

        # Prepare final output list, filtering out potential Nones if setup failed badly
        final_output = [res for res in translated_results if res is not None]
        successful_translations = sum(1 for res in final_output if isinstance(res.get("translated_dialogue"), str)) # Count successful strings
        failed_translations = len(final_output) - successful_translations - sum(1 for res in final_output if res.get("original_dialogue") == "")
        logger.info(f"Translation tasks finished. Successful: {successful_translations}, Failed/Skipped: {failed_translations}.", extra=extra_log_data)

        final_error_message = "; ".join(task_errors) if task_errors else None
        # Combine setup/gather errors with task errors
        if error_message and final_error_message: error_message = f"{error_message}; {final_error_message}"
        elif final_error_message: error_message = final_error_message

        if error_message:
             logger.warning(f"Some errors occurred during translation: {error_message}", extra=extra_log_data)

        end_time = datetime.now(timezone.utc)
        node18_processing_stats = (end_time - start_time).total_seconds()

        update_data: Dict[str, Any] = {
            "translated_text": final_output, # Return list even if errors occurred
            "node18_processing_stats": node18_processing_stats,
            "error_message": error_message
        }

        # --- ADDED: End Logging ---
        log_level = logger.warning if error_message else logger.info
        log_level(f"Translation result: {successful_translations}/{len(final_output)} dialogues translated. Errors: {error_message is not None}", extra=extra_log_data)
        logger.debug(f"Returning updates:\n{summarize_for_logging(update_data, is_state_object=False)}", extra=extra_log_data)
        logger.info(f"--- Finished {node_class_name} --- (Elapsed: {node18_processing_stats:.2f}s)", extra=extra_log_data)
        # -------------------------

        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}