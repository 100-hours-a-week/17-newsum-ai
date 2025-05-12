# app/services/google_translation_service.py
import asyncio
import httpx
from typing import Optional, Dict, Any, Union, List
# import json
# from pathlib import Path
import html  # HTML 엔티티 처리를 위해 추가

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config.settings import Settings
from app.utils.logger import get_logger, summarize_for_logging

logger = get_logger("GoogleRestTranslationService")
settings = Settings()

class GoogleRestTranslationService:
    """Google Cloud Translation API v2 (REST, API Key)와 상호작용하는 서비스"""

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.api_key = settings.GOOGLE_API_KEY
        self.api_url = "https://translation.googleapis.com/language/translate/v2"
        self.is_enabled = bool(self.api_key)
        self._client = http_client
        self._created_client = False

        # 배치 관련 설정
        self.batch_size = getattr(settings, 'TRANSLATION_BATCH_SIZE', 50)
        self.batch_char_limit = getattr(settings, 'TRANSLATION_BATCH_CHAR_LIMIT', 5000)

        if not self.is_enabled:
            logger.warning("GOOGLE_API_KEY is missing in settings. Google Translation service (REST API Key) disabled.")
        else:
            if self._client is None:
                timeout_seconds = getattr(settings, 'TOOL_HTTP_TIMEOUT', 30.0)
                self._client = httpx.AsyncClient(timeout=float(timeout_seconds))
                self._created_client = True
                logger.info(
                    f"GoogleRestTranslationService initialized. Using API Key. Timeout: {timeout_seconds}s, Batch Size: {self.batch_size}, Batch Char Limit: {self.batch_char_limit}")

    def _unescape_html_entities(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        return html.unescape(text)

    @retry(
        stop=stop_after_attempt(getattr(settings, 'TOOL_RETRY_ATTEMPTS', 3)),
        wait=wait_exponential(multiplier=1,
                              min=getattr(settings, 'TOOL_RETRY_WAIT_MIN', 1),
                              max=getattr(settings, 'TOOL_RETRY_WAIT_MAX', 5)),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError)),
        reraise=True
    )
    async def _translate_texts_via_api(
            self,
            texts: Union[str, List[str]],
            target_lang: str,
            source_lang: Optional[str] = None,
            text_format: str = 'html',  # 'html' or 'text'
            trace_id: Optional[str] = None
    ) -> Optional[List[Dict[str, str]]]:
        """
        Helper method to call Google Translate API for single or multiple texts.
        Returns a list of translation result dictionaries, or None on failure.
        Each dictionary in the list corresponds to an input text.
        """
        if not self.is_enabled or not self._client:
            logger.warning(f"{trace_id} Google Translation service is disabled or client not initialized.")
            return None

        if not texts:
            logger.debug(f"{trace_id} Empty text(s) received for translation, returning None.")
            return None

        log_prefix = f"[{trace_id}]" if trace_id else ""
        num_texts_to_translate = len(texts) if isinstance(texts, list) else 1
        source_info = f"from '{source_lang}' " if source_lang else "(auto-detect) "
        logger.debug(
            f"{log_prefix} Google REST: Translating {num_texts_to_translate} text(s) "
            f"'{summarize_for_logging(str(texts), 30)}...' {source_info}to '{target_lang}', format: '{text_format}'"
        )

        request_url = f"{self.api_url}?key={self.api_key}"
        payload: Dict[str, Any] = {
            'q': texts,  # API accepts single string or list of strings
            'target': target_lang,
            'format': text_format
        }
        if source_lang:
            payload['source'] = source_lang

        try:
            response = await self._client.post(request_url, json=payload)
            response.raise_for_status()
            result = response.json()

            if result and 'data' in result and 'translations' in result['data']:
                # API returns a list of translation objects, one for each input string in 'q'
                translations_data = result['data']['translations']
                if translations_data and len(translations_data) == num_texts_to_translate:
                    logger.debug(
                        f"{log_prefix} Google REST translation successful for {len(translations_data)} text(s).")
                    return translations_data  # List of dicts like [{"translatedText": "...", "detectedSourceLanguage": "..."}]
                else:
                    logger.warning(
                        f"{log_prefix} Mismatch in translated items count or empty translations. Expected: {num_texts_to_translate}, Got: {len(translations_data) if translations_data else 0}. Response: {summarize_for_logging(result)}")
                    return None

            logger.warning(
                f"{log_prefix} Google REST API returned no valid translation data structure. Response: {summarize_for_logging(result)}")
            return None
        except httpx.HTTPStatusError as e:
            response_text = e.response.text
            logger.error(
                f"{log_prefix} Google REST API HTTP error (Status {e.response.status_code}): {summarize_for_logging(response_text)}",
                exc_info=False)
            try:
                error_details = e.response.json()
                logger.error(f"{log_prefix} Google REST API error details: {error_details}")
            except ValueError:
                logger.error(f"{log_prefix} Google REST API error response (not JSON): {response_text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"{log_prefix} Google REST API request error: {type(e).__name__} - {e}", exc_info=True)
            raise
        except Exception as e:
            logger.exception(f"{log_prefix} Google REST translation unexpected error: {e}")
            raise

    async def translate(
            self,
            text: str,
            target_lang: str,
            source_lang: Optional[str] = None,
            text_format: str = 'html',  # Allow specifying format
            unescape_result: bool = False,  # For JSON, we might want to unescape
            trace_id: Optional[str] = None
    ) -> Optional[str]:
        if not text:  # API 호출 전에 빈 텍스트 처리
            logger.debug(f"{trace_id} Empty text received for single translation, returning None.")
            return None

        translation_results = await self._translate_texts_via_api(
            texts=text,
            target_lang=target_lang,
            source_lang=source_lang,
            text_format=text_format,
            trace_id=trace_id
        )

        if translation_results and translation_results[0]:
            translated_text = translation_results[0].get('translatedText')
            if unescape_result:
                return self._unescape_html_entities(translated_text)
            return translated_text
        return None

    async def translate_batch(
            self,
            texts: List[str],
            target_lang: str,
            source_lang: Optional[str] = None,
            text_format: str = 'html',
            unescape_results: bool = False,
            trace_id: Optional[str] = None
    ) -> List[Optional[str]]:
        if not texts:
            logger.debug(f"{trace_id} Empty list of texts received for batch translation, returning empty list.")
            return []

        results: List[Optional[str]] = [None] * len(texts)

        # Filter out empty strings to avoid unnecessary API calls / errors if API doesn't handle them well in batch
        # Keep track of original indices to reconstruct the results
        indexed_texts_to_translate: List[tuple[int, str]] = []
        for i, t in enumerate(texts):
            if t and t.strip():  # Only translate non-empty, non-whitespace-only strings
                indexed_texts_to_translate.append((i, t))
            else:
                results[i] = t  # Preserve empty or whitespace-only strings as is

        if not indexed_texts_to_translate:
            logger.debug(f"{trace_id} All texts in batch are empty or whitespace, returning original empty strings.")
            return results

        # Split into smaller batches if necessary based on count or character limit
        current_batch_texts: List[str] = []
        current_batch_indices: List[int] = []
        current_char_count = 0

        # Iterate through texts that actually need translation
        for original_idx, text_to_translate in indexed_texts_to_translate:
            # Check if adding this text exceeds limits
            if (len(current_batch_texts) >= self.batch_size or
                    (current_char_count + len(text_to_translate)) > self.batch_char_limit and current_batch_texts):
                # Process the current batch
                logger.debug(
                    f"{trace_id} Processing a sub-batch of {len(current_batch_texts)} texts (char count: {current_char_count}).")
                api_results = await self._translate_texts_via_api(
                    texts=current_batch_texts, target_lang=target_lang, source_lang=source_lang,
                    text_format=text_format, trace_id=f"{trace_id}-sub_batch"
                )
                if api_results:
                    for i, api_res_item in enumerate(api_results):
                        idx_in_original_texts_list = current_batch_indices[i]
                        translated_text = api_res_item.get('translatedText')
                        results[idx_in_original_texts_list] = self._unescape_html_entities(
                            translated_text) if unescape_results else translated_text
                else:  # API call failed for this sub-batch, mark corresponding results as None (or keep original if preferred)
                    for i in current_batch_indices:
                        results[i] = texts[i]  # Revert to original on sub-batch failure, or keep None
                    logger.warning(f"{trace_id} Sub-batch translation failed. Originals retained for this sub-batch.")

                # Reset for next batch
                current_batch_texts = []
                current_batch_indices = []
                current_char_count = 0

            current_batch_texts.append(text_to_translate)
            current_batch_indices.append(original_idx)  # Store the original index in the input 'texts' list
            current_char_count += len(text_to_translate)

        # Process any remaining texts in the last batch
        if current_batch_texts:
            logger.debug(
                f"{trace_id} Processing the final sub-batch of {len(current_batch_texts)} texts (char count: {current_char_count}).")
            api_results = await self._translate_texts_via_api(
                texts=current_batch_texts, target_lang=target_lang, source_lang=source_lang,
                text_format=text_format, trace_id=f"{trace_id}-final_sub_batch"
            )
            if api_results:
                for i, api_res_item in enumerate(api_results):
                    idx_in_original_texts_list = current_batch_indices[i]
                    translated_text = api_res_item.get('translatedText')
                    results[idx_in_original_texts_list] = self._unescape_html_entities(
                        translated_text) if unescape_results else translated_text
            else:
                for i in current_batch_indices:
                    results[i] = texts[i]  # Revert to original
                logger.warning(f"{trace_id} Final sub-batch translation failed. Originals retained.")

        return results

    async def close(self):
        if self._client and self._created_client:
            try:
                await self._client.aclose()
                logger.info("GoogleRestTranslationService: httpx client closed successfully.")
                self._client = None
            except Exception as e:
                logger.error(f"GoogleRestTranslationService: Failed to close httpx client: {e}", exc_info=True)
        elif self._client:
            logger.info("GoogleRestTranslationService: Using externally managed httpx client. Not closing here.")


# --- Main Test Function ---
async def collect_and_translate_json_recursively_batch(
        data: Union[Dict[str, Any], List[Any]],
        service: GoogleRestTranslationService,
        target_lang: str,
        source_lang: Optional[str],
        trace_id: str,
        keys_to_skip: Optional[List[str]] = None,  # 특정 키 번역 제외
        min_str_len_to_translate: int = 1  # 번역할 최소 문자열 길이
) -> Union[Dict[str, Any], List[Any]]:
    """
    Helper function to recursively find strings in JSON-like structures,
    translate them in batches, and reconstruct the structure.
    """
    keys_to_skip = keys_to_skip or []
    strings_to_translate_map: Dict[int, Dict[str, Any]] = {}  # id -> {'path': (path_tuple), 'text': str}
    string_id_counter = 0

    # 1. Collect all strings to be translated with their paths
    def _collect_strings(current_data: Any, current_path: tuple):
        nonlocal string_id_counter
        if isinstance(current_data, dict):
            for key, value in current_data.items():
                if key in keys_to_skip:
                    continue
                _collect_strings(value, current_path + (key,))
        elif isinstance(current_data, list):
            for i, item in enumerate(current_data):
                _collect_strings(item, current_path + (i,))
        elif isinstance(current_data, str) and len(current_data.strip()) >= min_str_len_to_translate:
            strings_to_translate_map[string_id_counter] = {'path': current_path, 'text': current_data}
            string_id_counter += 1

    _collect_strings(data, tuple())

    if not strings_to_translate_map:
        logger.info(f"{trace_id} No strings found to translate in JSON data based on criteria.")
        return data  # Return original data if no strings to translate

    original_texts_list = [item['text'] for item in strings_to_translate_map.values()]

    logger.info(f"{trace_id} Collected {len(original_texts_list)} strings from JSON for batch translation.")

    # 2. Translate collected strings in batch (true for unescaping as it's JSON)
    # For JSON values, we typically want 'text' format, not 'html', unless values are HTML snippets.
    # Here, assuming 'text' format for JSON string values. If HTML is embedded, it will be treated as text.
    # Set unescape_results=True because the API might return HTML entities even for text format if input had them.
    # However, Google API for 'text' format should return plain text. For 'html' format, it returns HTML.
    # Since we are putting this back into JSON, unescaping is crucial if API returns entities.
    translated_texts_list = await service.translate_batch(
        original_texts_list, target_lang, source_lang, text_format='text', unescape_results=True,
        trace_id=f"{trace_id}-json_batch"
    )

    # 3. Reconstruct the data with translated strings
    # Create a deep copy to modify
    import copy
    new_data = copy.deepcopy(data)

    map_values_list = list(strings_to_translate_map.values())  # Ensure order if dict is not ordered (Python < 3.7)

    for i, translated_text in enumerate(translated_texts_list):
        if translated_text is not None:  # Only update if translation was successful
            path_info = map_values_list[i]
            current_level = new_data
            for j, path_segment in enumerate(path_info['path']):
                if j == len(path_info['path']) - 1:  # Last segment
                    current_level[path_segment] = translated_text
                else:
                    current_level = current_level[path_segment]
        else:  # Translation failed for this specific string, keep original
            original_text = map_values_list[i]['text']
            path_info = map_values_list[i]
            logger.warning(
                f"{trace_id} Translation failed for string at path {path_info['path']}. Keeping original: '{summarize_for_logging(original_text)}'")

    return new_data


# async def main_test():
#     if logger.name == "GoogleRestTranslationService_Fallback":
#         logging.getLogger().setLevel(logging.INFO)
#         logger.info("Using fallback logger for main_test.")
#
#     logger.info("--- GoogleRestTranslationService File-Based Test (with Batching & Unescape) Start ---")
#
#     TARGET_LANG_FOR_TEST = "ko"
#     SOURCE_LANG_FOR_TEST = "en"
#     # For JSON string values, we usually want 'text' format and unescaping.
#     # For HTML files, we want 'html' format and no unescaping (browser handles it).
#     # For TXT files, 'text' format and unescaping (if API might return entities).
#
#     script_dir = Path(__file__).resolve().parent
#     test_data_base_dir = script_dir / "test_translate_data"
#     input_dir = test_data_base_dir / "input"
#     output_dir = test_data_base_dir / "output"
#
#     input_dir.mkdir(parents=True, exist_ok=True)
#     output_dir.mkdir(parents=True, exist_ok=True)
#
#     logger.info(f"Test input directory: {input_dir.resolve()}")
#     logger.info(f"Test output directory: {output_dir.resolve()}")
#     logger.info(f"Target language for test: {TARGET_LANG_FOR_TEST}")
#     if SOURCE_LANG_FOR_TEST:
#         logger.info(f"Source language for test: {SOURCE_LANG_FOR_TEST}")
#
#     # Sample files (ensure they exist or are created for testing)
#     sample_files_to_create = {
#         "sample_en_batch.txt": "Hello world.\nThis is a test.\nAnother line for batching.",
#         "sample_en_batch.html": "<h1>Hello &amp; Welcome</h1><p>This is a paragraph with entities.</p><p>Another paragraph for <b>OpenAI</b> &amp; friends.</p>",
#         "sample_en_batch.json": json.dumps({
#             "title": "Batch Test Document",
#             "description": "This document contains several &quot;strings&quot; to test batch translation.",
#             "items": [
#                 {"id": "001", "name": "First item", "details": "Details for the first item &amp; more."},
#                 {"id": "002", "name": "Second item", "details": "Details for the second item."},
#                 {"id": "003", "name": "NoDetails Inc."}  # No details key
#             ],
#             "footer": "All rights reserved. &copy; 2025",
#             "skip_this_key": "This string should not be translated if key is skipped.",
#             "empty_value_test": ""
#         }, indent=2, ensure_ascii=False),  # ensure_ascii=False for source to have actual chars
#         "only_short_strings.json": json.dumps({"a": "Go", "b": "Run", "c": "Stop it now please"}, indent=2)
#     }
#     for filename, content in sample_files_to_create.items():
#         sample_file_path = input_dir / filename
#         if not sample_file_path.exists():  # Create if doesn't exist
#             with open(sample_file_path, "w", encoding="utf-8") as f:
#                 f.write(content)
#             logger.info(f"Created sample file: {sample_file_path}")
#
#     if not settings.GOOGLE_API_KEY:
#         logger.error("GOOGLE_API_KEY is not set. Service disabled. Skipping tests.")
#         return
#
#     translation_service = GoogleRestTranslationService()
#     if not translation_service.is_enabled:
#         logger.warning("Translation service is disabled (e.g. no API key). Skipping API call tests.")
#         await translation_service.close()
#         return
#
#     allowed_extensions = [".txt", ".html", ".json"]
#     trace_id_base = "file-batch-test"
#     file_counter = 0
#
#     for filepath in input_dir.glob("*"):
#         if filepath.is_file() and filepath.suffix.lower() in allowed_extensions:
#             file_counter += 1
#             trace_id = f"{trace_id_base}-{file_counter}-{filepath.stem}"
#             logger.info(f"\n--- Processing file: {filepath.name} (Trace ID: {trace_id}) ---")
#
#             try:
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     original_content_str = f.read()
#
#                 translated_content_str: Optional[str] = None
#                 output_filename_stem = f"{filepath.stem}_Translated_{TARGET_LANG_FOR_TEST}"
#                 output_filepath = output_dir / f"{output_filename_stem}{filepath.suffix}"
#
#                 if filepath.suffix.lower() == ".json":
#                     try:
#                         json_data = json.loads(original_content_str)
#                         logger.info(f"Translating JSON content for {filepath.name} using batch...")
#                         translated_data = await collect_and_translate_json_recursively_batch(
#                             data=json_data,
#                             service=translation_service,
#                             target_lang=TARGET_LANG_FOR_TEST,
#                             source_lang=SOURCE_LANG_FOR_TEST,
#                             trace_id=trace_id,
#                             keys_to_skip=["comic_id", "generation_timestamp", "skip_this_key"],  # Example skip keys
#                             min_str_len_to_translate=2  # Example: translate strings with 2 or more chars
#                         )
#                         translated_content_str = json.dumps(translated_data, indent=2, ensure_ascii=False)
#                         logger.info(f"JSON content translated for {filepath.name}")
#                     except json.JSONDecodeError as je:
#                         logger.error(f"Failed to decode JSON from {filepath.name}: {je}")
#                         continue
#                 elif filepath.suffix.lower() == ".html":
#                     logger.info(f"Translating HTML content for {filepath.name}...")
#                     # For HTML, we want 'html' format and typically no unescaping by service (browser handles it)
#                     translated_content_str = await translation_service.translate(
#                         text=original_content_str,
#                         target_lang=TARGET_LANG_FOR_TEST,
#                         source_lang=SOURCE_LANG_FOR_TEST,
#                         text_format='html',
#                         unescape_result=False,  # HTML content, unescaping done by browser
#                         trace_id=trace_id
#                     )
#                     logger.info(f"HTML content translated for {filepath.name}")
#                 elif filepath.suffix.lower() == ".txt":
#                     logger.info(f"Translating TXT content for {filepath.name}...")
#                     # For TXT, we want 'text' format and unescaping (if API might return entities)
#                     # Using translate_batch for .txt to demonstrate it, even for single large text.
#                     # Split by lines for a pseudo-batch effect if desired, or send as one block.
#                     # Here, sending as one block via single translate method.
#                     # Or, demonstrate batching:
#                     # lines = original_content_str.splitlines()
#                     # translated_lines = await translation_service.translate_batch(
#                     # lines, TARGET_LANG_FOR_TEST, SOURCE_LANG_FOR_TEST, text_format='text', unescape_results=True, trace_id=trace_id)
#                     # translated_content_str = "\n".join(l if l is not None else "" for l in translated_lines)
#
#                     # Simpler: translate the whole text block as one.
#                     translated_content_str = await translation_service.translate(
#                         text=original_content_str,
#                         target_lang=TARGET_LANG_FOR_TEST,
#                         source_lang=SOURCE_LANG_FOR_TEST,
#                         text_format='text',
#                         unescape_result=True,  # Ensure plain text
#                         trace_id=trace_id
#                     )
#                     logger.info(f"TXT content translated for {filepath.name}")
#
#                 if translated_content_str is not None:
#                     with open(output_filepath, "w", encoding="utf-8") as f_out:
#                         f_out.write(translated_content_str)
#                     logger.info(f"Saved translated file: {output_filepath.resolve()}")
#                 else:
#                     logger.warning(f"Translation returned None for {filepath.name}. Output file not saved.")
#
#             except httpx.HTTPStatusError as e:
#                 logger.error(
#                     f"HTTPStatusError for '{filepath.name}': {e.response.status_code} - {summarize_for_logging(e.response.text, 200)}")
#             except Exception as e:
#                 logger.error(f"Error processing file {filepath.name}: {type(e).__name__} - {e}", exc_info=True)
#             finally:
#                 await asyncio.sleep(getattr(settings, 'TEST_API_CALL_DELAY', 0.2))
#
#     if file_counter == 0:
#         logger.warning(f"No files found to process in {input_dir.resolve()} with extensions {allowed_extensions}")
#
#     await translation_service.close()
#     logger.info("--- GoogleRestTranslationService File-Based Test (with Batching & Unescape) End ---")
#
#
# if __name__ == "__main__":
#     if logger.name == "GoogleRestTranslationService_Fallback":
#         import logging
#
#         _handler = logging.StreamHandler()
#         _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] %(message)s')
#         _handler.setFormatter(_formatter)
#         _root_logger = logging.getLogger()
#         if not _root_logger.hasHandlers():
#             _root_logger.addHandler(_handler)
#         _root_logger.setLevel(logging.INFO)
#         logger.setLevel(logging.DEBUG)
#         logger.info("Fallback logger setup complete for __main__ execution.")
#
#     asyncio.run(main_test())