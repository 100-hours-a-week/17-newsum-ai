# app/agents/imager_agent.py

import logging
from app.workflows.state import ComicState
from app.services.image_server_client import generate_single_image
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class ImagerAgent:
    """
    시나리오의 각 컷을 기반으로 이미지를 생성하는 에이전트
    """
    async def run(self, state: ComicState) -> Dict[str, Optional[Any]]:
        logger.info("--- Imager Agent 실행 시작 ---")
        updates: Dict[str, Optional[Any]] = {}

        if not state.scenarios:
            logger.error("시나리오(prompt) 정보가 없습니다.")
            updates["error_message"] = "No scenarios available for image generation."
            return updates

        generated_image_urls = []

        try:
            for idx, scene in enumerate(state.scenarios):
                prompt = scene.get("description", "")
                if not prompt:
                    logger.warning(f"{idx+1}번째 컷: prompt가 비어 있습니다. 건너뜁니다.")
                    continue

                logger.info(f"🖼️ {idx+1}번째 컷 생성 요청 - 프롬프트: {prompt}")

                image_url = await generate_single_image(prompt=prompt, idx=idx)
                generated_image_urls.append(image_url)

            if not generated_image_urls:
                logger.error("이미지가 하나도 생성되지 않았습니다.")
                updates["error_message"] = "No images were generated."
            else:
                updates["image_urls"] = generated_image_urls
                updates["error_message"] = None
                logger.info(f"✅ 총 {len(generated_image_urls)}장의 이미지 생성 완료.")

        except Exception as e:
            logger.exception(f"Imager Agent 실행 실패: {e}")
            updates["error_message"] = f"Failed to generate images: {str(e)}"

        logger.info("--- Imager Agent 실행 종료 ---")
        return updates
    