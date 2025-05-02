# app/nodes/18_translator_node.py

import asyncio
# --- datetime, timezone 임포트 추가 ---
from datetime import datetime, timezone
# ------------------------------------
from typing import List, Dict, Any, Optional
import aiohttp # 세션 관리를 위해 필요

# --- 프로젝트 구성 요소 임포트 ---
# from app.config.settings import settings # 직접 사용하지 않음
from app.services.papago_translation_service import PapagoTranslationService # 실제 번역 서비스
# from app.services.langsmith_service_v2 import LangSmithService # 필요시
from app.utils.logger import get_logger
from app.workflows.state import ComicState

# --- 로거 설정 ---
logger = get_logger("TranslatorNode")

class TranslatorNode:
    """
    (Refactored) (선택적) 시나리오 대사를 번역합니다.
    - PapagoTranslationService 사용.
    - config의 translation_enabled 플래그에 따라 활성화.
    """

    # 상태 입력/출력 정의 (ComicState 필드 기준)
    inputs: List[str] = ["scenarios", "config", "trace_id", "processing_stats"]
    # translated_text는 Optional[List[Dict[str, str]]] 이므로 실패 시 None 반환 가능
    outputs: List[str] = ["translated_text", "processing_stats", "error_message"]

    # PapagoTranslationService 인스턴스를 외부에서 주입받음
    def __init__(
        self,
        translator_client: PapagoTranslationService,
        # langsmith_service: Optional[LangSmithService] = None # 선택적
    ):
        self.translator = translator_client
        # self.langsmith_service = langsmith_service
        logger.info("TranslatorNode initialized with PapagoTranslationService.")

    async def _translate_dialogue(
        self, dialogue: str, source_lang: str, target_lang: str, session: aiohttp.ClientSession, trace_id: Optional[str]
        ) -> Optional[str]:
        """단일 대사 번역 (서비스 호출)"""
        if not dialogue or not self.translator.is_enabled:
            return dialogue if not dialogue else None # 빈 대사는 그대로, 비활성화 시 None

        try:
            # 서비스 클라이언트의 translate 메서드 호출 (내부적으로 재시도 처리)
            translated = await self.translator.translate(dialogue, source_lang, target_lang, session, trace_id)
            return translated # 성공 시 번역된 텍스트, 실패 시 None 반환
        except Exception as e:
            # 서비스 클라이언트 재시도 실패 후 발생한 예외 처리
            logger.error(f"[{trace_id}] Translation failed for dialogue '{dialogue[:30]}...': {e}")
            return None # 최종 실패 시 None 반환

    # --- 메인 실행 메서드 ---
    async def run(self, state: ComicState) -> Dict[str, Any]:
        """번역 프로세스 실행 (활성화된 경우)"""
        start_time = datetime.now(timezone.utc)
        log_prefix = f"[{state.trace_id}]"
        logger.info(f"{log_prefix} Executing TranslatorNode...")

        # 상태 및 설정 로드
        scenarios = state.scenarios or []
        config = state.config or {}
        processing_stats = state.processing_stats or {}
        error_message: Optional[str] = None

        # --- 번역 활성화 여부 및 입력 확인 ---
        if not self.translator.is_enabled or not config.get("translation_enabled", False):
            logger.info(f"{log_prefix} Translation disabled. Skipping.")
            processing_stats['translator_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            # ComicState.translated_text 는 Optional 이므로 None 반환 가능
            return {"translated_text": None, "processing_stats": processing_stats}

        if not scenarios:
            logger.warning(f"{log_prefix} No scenarios to translate. Skipping.")
            processing_stats['translator_node_time'] = (datetime.now(timezone.utc) - start_time).total_seconds()
            return {"translated_text": [], "processing_stats": processing_stats} # 빈 리스트 반환

        # 설정값 로드
        target_lang = config.get("target_language", "en")
        # 원본 언어 결정 (config 우선, 없으면 휴리스틱)
        source_lang = config.get("source_language")
        if not source_lang:
             source_lang = "ko" if target_lang == "en" else "en" # 간단한 휴리스틱
             logger.warning(f"{log_prefix} Source language not specified, assuming '{source_lang}' based on target '{target_lang}'.")
        concurrency_limit = config.get('translator_concurrency', 3)

        logger.info(f"{log_prefix} Starting translation of {len(scenarios)} panels from '{source_lang}' to '{target_lang}'...")

        translated_results: List[Optional[Dict[str, Any]]] = [None] * len(scenarios) # 결과 순서 유지용
        tasks = []
        semaphore = asyncio.Semaphore(concurrency_limit)
        task_map = {} # 인덱스 매핑용

        # aiohttp 세션 생성하여 서비스에 전달
        async with aiohttp.ClientSession() as session:
            for i, panel in enumerate(scenarios):
                original_dialogue = panel.get('dialogue', '')
                scene_num = panel.get('scene', i + 1)

                if original_dialogue:
                    # 번역 작업 생성
                    async def translate_task(dialogue, scn_idx):
                        async with semaphore:
                            translated = await self._translate_dialogue(dialogue, source_lang, target_lang, session, state.trace_id)
                            return scn_idx, translated # 인덱스와 결과 반환

                    task = translate_task(original_dialogue, i)
                    tasks.append(task)
                    task_map[id(task)] = i # 태스크 ID와 인덱스 매핑
                else:
                    # 대사 없는 패널 처리
                    translated_results[i] = {
                        "scene": scene_num,
                        "original_dialogue": "",
                        "translated_dialogue": ""
                    }

            # 번역 작업 실행 (대사가 있는 패널만)
            if tasks:
                 results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)

                 # 결과 처리 및 순서 매핑
                 for idx, result in enumerate(results_from_gather):
                      # gather 결과 순서가 tasks 순서와 같다고 가정하기 어려울 수 있음
                      # 좀 더 안전하게 하려면 task 객체를 직접 사용하여 매핑 필요
                      # 여기서는 gather 결과 순서가 입력 task 순서와 같다고 가정하고 진행
                      original_panel_index = -1
                      # 임시 방편: tasks 리스트에서 순서대로 인덱스 찾기 (비효율적)
                      # 더 나은 방법: create_task 사용 및 완료 콜백 또는 task 객체 직접 참조
                      try:
                           # tasks[idx]를 사용하여 해당 태스크가 어떤 인덱스였는지 찾아야 함
                           # 이 방식은 gather의 결과 순서가 입력 순서와 같다는 가정 하에 동작
                           original_panel_index = [j for j, p in enumerate(scenarios) if p.get('dialogue')][idx]

                           original_panel = scenarios[original_panel_index]
                           scene_num = original_panel.get('scene', original_panel_index + 1)
                           original_dialogue = original_panel.get('dialogue', '')

                           if isinstance(result, str) or result is None: # 성공 또는 API가 None 반환
                                translated_results[original_panel_index] = {
                                    "scene": scene_num,
                                    "original_dialogue": original_dialogue,
                                    "translated_dialogue": result # 성공 시 번역된 텍스트, 실패 시 None
                                }
                           elif isinstance(result, Exception):
                                logger.error(f"{log_prefix} Translation task for panel index {original_panel_index} failed: {result}")
                                translated_results[original_panel_index] = {
                                    "scene": scene_num,
                                    "original_dialogue": original_dialogue,
                                    "translated_dialogue": None # 실패 시 None
                                }
                                error_message = error_message + f"; Panel {scene_num} translation failed" if error_message else f"Panel {scene_num} translation failed"
                      except IndexError:
                           logger.error(f"{log_prefix} Error mapping gather result back to original panel index {idx}.")
                           error_message = error_message + "; Result mapping error" if error_message else "Result mapping error"

            else:
                 logger.info(f"{log_prefix} No dialogues found needing translation.")


        # 최종 결과 리스트 (None 제거 확인 - 실제로는 None이 있을 수 있음)
        # ComicState의 translated_text는 Optional[List[...]] 이므로 None 포함 가능
        final_output = [res for res in translated_results if res is not None]
        successful_translations = sum(1 for res in final_output if res.get("translated_dialogue"))
        failed_translations = len(final_output) - successful_translations

        logger.info(f"{log_prefix} Translation tasks finished. Successful: {successful_translations}, Failed/Skipped: {failed_translations}.")

        # --- 처리 시간 및 상태 반환 ---
        end_time = datetime.now(timezone.utc)
        node_processing_time = (end_time - start_time).total_seconds()
        processing_stats['translator_node_time'] = node_processing_time
        logger.info(f"{log_prefix} TranslatorNode finished in {node_processing_time:.2f} seconds.")

        # TODO: LangSmith 로깅

        # --- ComicState 업데이트를 위한 결과 반환 ---
        update_data: Dict[str, Any] = {
            # final_output은 scene, original_dialogue, translated_dialogue(Optional[str]) 포함
            "translated_text": final_output if config.get("translation_enabled", False) else None,
            "processing_stats": processing_stats,
            "error_message": error_message
        }
        valid_keys = set(ComicState.model_fields.keys())
        return {k: v for k, v in update_data.items() if k in valid_keys}