# app/nodes/18_translator_node.py (Improved Version)

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import aiohttp # 세션 관리용

# 프로젝트 구성 요소 임포트
from app.config.settings import settings # 기본값 참조용
from app.services.papago_translation_service import PapagoTranslationService # 번역 서비스
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# 로거 설정
logger = get_logger(__name__)

class TranslatorNode:
    """
    (선택적) 시나리오 대사를 번역합니다.
    - PapagoTranslationService 사용 (재시도 로직은 서비스 내부에 구현 가정).
    - `state.config`의 `translation_enabled` 플래그 및 서비스 자체 활성화 여부에 따라 실행.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["scenarios", "config", "trace_id", "processing_stats"]
    outputs: List[str] = ["translated_text", "processing_stats", "error_message"]

    # PapagoTranslationService 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        translator_client: PapagoTranslationService,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        if not translator_client: raise ValueError("PapagoTranslationService is required.")
        self.translator = translator_client
        # self.langsmith = langsmith_service
        logger.info("TranslatorNode initialized.")

    # --- 내부 설정 로드 (실행 시점) ---
    def _load_runtime_config(self, config: Dict[str, Any]):
        """실행 시 필요한 설정을 config 또는 settings에서 로드"""
        self.translation_enabled = config.get("translation_enabled", settings.ENABLE_TRANSLATION)
        self.target_lang = config.get("target_language", settings.DEFAULT_TARGET_LANG)
        # 소스 언어: config 우선, 없으면 target 기반 추론, 그것도 안되면 settings 기본값
        self.source_lang = config.get("source_language")
        if not self.source_lang:
             self.source_lang = "ko" if self.target_lang == "en" else "en" # 간단 휴리스틱
             logger.warning(f"Source language not specified, assuming '{self.source_lang}' based on target '{self.target_lang}'.")
        self.source_lang = self.source_lang or settings.DEFAULT_SOURCE_LANG # 최종 fallback

        self.concurrency_limit = int(config.get('translator_concurrency', settings.DEFAULT_TRANSLATOR_CONCURRENCY))
        # HTTP 타임아웃 (번역 서비스 호출 시 사용)
        self.http_timeout = config.get('http_timeout', settings.DEFAULT_HTTP_TIMEOUT)

        logger.debug(f"Runtime config loaded. Enabled: {self.translation_enabled}, Source: {self.source_lang}, Target: {self.target_lang}")
        logger.debug(f"Concurrency: {self.concurrency_limit}, Timeout: {self.http_timeout}")


    async def _translate_dialogue_wrapper(
        self, dialogue: str, session: aiohttp.ClientSession, trace_id: Optional[str]
        ) -> Optional[str]:
        """단일 대사 번역 (서비스 호출 및 오류 처리)"""
        if not dialogue: return "" # 빈 대사는 번역 없이 빈 문자열 반환
        log_prefix = f"[{trace_id}]" if trace_id else ""
        try:
            # 서비스 클라이언트의 translate 메서드 호출 (내부적으로 재시도 처리 가정)
            # translate 메서드는 실패 시 None 또는 Exception 발생 가정
            translated = await self.translator.translate(dialogue, self.source_lang, self.target_lang, session, trace_id)
            if translated is None:
                 logger.warning(f"{log_prefix} Translation returned None for '{dialogue[:30]}...'.")
            return translated # 성공 시 번역된 텍스트, 실패 시 None
        except Exception as e:
            # 서비스 클라이언트 재시도 실패 후 발생한 최종 예외 처리
            logger.error(f"{log_prefix} Translation failed for '{dialogue[:30]}...': {e}", exc_info=True)
            return None # 최종 실패 시 None 반환

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """번역 프로세스 실행 (활성화된 경우)"""
        start_time = datetime.now(timezone.utc)
        trace_id = state.trace_id
        log_prefix = f"[{trace_id}]"
        logger.info(f"{log_prefix} Executing TranslatorNode...")

        scenarios = state.scenarios or []
        config = state.config or {}
        processing_stats = state.processing_stats or {}
        error_message: Optional[str] = None

        # --- 실행 시점 설정 로드 ---
        self._load_runtime_config(config)
        # --------------------------

        # 번역 비활성화 조건 확인 (설정 또는 서비스 자체)
        if not self.translation_enabled or not self.translator.is_enabled:
            status = "disabled in config" if not self.translation_enabled else "service unavailable"
            logger.info(f"{log_prefix} Translation {status}. Skipping.")
            processing_stats['translator_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            # translated_text는 Optional 이므로 None 반환
            return {"translated_text": None, "processing_stats": processing_stats}

        if not scenarios or len(scenarios) != 4:
            logger.warning(f"{log_prefix} Invalid or missing scenarios ({len(scenarios)} found). Cannot translate.")
            processing_stats['translator_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"translated_text": [], "processing_stats": processing_stats} # 빈 리스트 반환

        logger.info(f"{log_prefix} Starting translation of {len(scenarios)} panels from '{self.source_lang}' to '{self.target_lang}'...")

        # translated_text 필드 형식: List[Dict[str, Optional[str]]]
        # {"scene": int, "original_dialogue": str, "translated_dialogue": Optional[str]}
        translated_results: List[Optional[Dict[str, Any]]] = [None] * len(scenarios)
        tasks = []
        task_errors: List[str] = []
        panel_indices_to_translate = [] # 실제 번역 작업 대상 인덱스

        # aiohttp 세션 생성 (타임아웃 적용)
        timeout = aiohttp.ClientTimeout(total=self.http_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            semaphore = asyncio.Semaphore(self.concurrency_limit)
            for i, panel in enumerate(scenarios):
                original_dialogue = panel.get('dialogue', '')
                scene_num = panel.get('scene', i + 1)

                if original_dialogue: # 대사가 있는 경우에만 번역
                    panel_indices_to_translate.append(i) # 인덱스 저장
                    async def translate_task(dialogue, panel_idx):
                        async with semaphore:
                            translated = await self._translate_dialogue_wrapper(dialogue, session, trace_id)
                            return panel_idx, translated # 원래 인덱스와 결과 반환

                    tasks.append(translate_task(original_dialogue, i))
                else:
                    # 대사 없는 패널은 미리 결과 저장
                    translated_results[i] = {
                        "scene": scene_num,
                        "original_dialogue": "",
                        "translated_dialogue": "" # 번역 결과도 빈 문자열
                    }

            # 번역 작업 실행 (대사가 있는 패널만)
            if tasks:
                 gather_results = await asyncio.gather(*tasks, return_exceptions=True)

                 # 결과 처리 (반환된 인덱스 사용)
                 for result in gather_results:
                      if isinstance(result, Exception):
                           # 특정 태스크 실패 (어떤 패널인지 알기 어려움 - 개선 필요)
                           err_msg = f"A translation task failed: {result}"
                           logger.error(f"{log_prefix} {err_msg}")
                           task_errors.append(err_msg)
                           # 실패한 패널을 특정하기 어려우므로, 일단 오류 메시지만 기록
                      elif isinstance(result, tuple) and len(result) == 2:
                           panel_idx, translated_dialogue = result
                           original_panel = scenarios[panel_idx]
                           scene_num = original_panel.get('scene', panel_idx + 1)
                           original_dialogue = original_panel.get('dialogue', '')
                           # 결과 저장 (번역 실패 시 translated_dialogue는 None일 수 있음)
                           translated_results[panel_idx] = {
                               "scene": scene_num,
                               "original_dialogue": original_dialogue,
                               "translated_dialogue": translated_dialogue
                           }
                           if translated_dialogue is None: # 명시적 실패 기록
                               task_errors.append(f"Panel {scene_num} translation failed (returned None).")
            else:
                 logger.info(f"{log_prefix} No dialogues found needing translation.")

        # 최종 결과 리스트 (None 항목 제거)
        final_output = [res for res in translated_results if res is not None]
        successful_translations = sum(1 for res in final_output if res.get("translated_dialogue") is not None)
        failed_translations = len(final_output) - successful_translations - sum(1 for res in final_output if res.get("original_dialogue") == "") # 빈 대사 제외
        logger.info(f"{log_prefix} Translation tasks finished. Successful: {successful_translations}, Failed: {failed_translations}.")

        final_error_message = "; ".join(task_errors) if task_errors else None
        if final_error_message:
             logger.warning(f"{log_prefix} Some errors occurred during translation: {final_error_message}")

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        processing_stats['translator_node_time'] = (end_time - start_time).total_seconds() # 키 형식 통일
        logger.info(f"{log_prefix} TranslatorNode finished in {processing_stats['translator_node_time']:.2f} seconds.")

        # ComicState 업데이트
        update_data: Dict[str, Any] = {
            "translated_text": final_output if self.translation_enabled else None,
            "processing_stats": processing_stats,
            "error_message": final_error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}