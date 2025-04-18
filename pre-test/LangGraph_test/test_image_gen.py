# test_image_gen.py
import asyncio
import uuid
from pathlib import Path

# 필요한 모듈 및 함수 임포트
from src.core.imagers import generate_image
from src.core.schemas import ImagePromptResult
from src.core.utils import logger # 로거 사용
from src import settings # 설정 값 사용

async def main():
    """이미지 생성 함수를 직접 호출하여 테스트"""
    logger.info("--- Starting Simple Image Generation Test ---")

    # 1. 테스트용 이미지 프롬프트 생성
    #    실제 워크플로우에서는 이 부분이 draft_prompt_node에서 생성됩니다.
    #    여기서는 직접 값을 지정합니다.
    test_prompt = ImagePromptResult(
        news_id=f"test_{uuid.uuid4().hex[:6]}", # 테스트용 임의 ID
        positive_prompt="photo of a cute cat wearing a small party hat, digital art, high detail, sharp focus",
        # negative_prompt는 schemas에서 기본값 사용됨 (필요시 명시)
        # negative_prompt="ugly, blurry, text, words, low quality"
    )
    logger.info(f"Using test prompt: '{test_prompt.positive_prompt}'")

    # 2. 이미지 생성 함수 호출
    #    generate_image 함수는 내부에 API 호출 및 폴백 로직 포함
    try:
        render_result = await generate_image(test_prompt)

        # 3. 결과 출력
        logger.info("--- Image Generation Result ---")
        if render_result:
            logger.info(f"News ID: {render_result.news_id}")
            logger.info(f"Image saved to (relative path): {render_result.image_path}")
            logger.info(f"Is fallback image used? {render_result.is_fallback}")

            # 실제 파일 경로 확인 (선택적)
            full_path = settings.OUTPUT_DIR / render_result.image_path
            if full_path.exists():
                logger.info(f"Full image path: {full_path}")
                if render_result.is_fallback:
                     logger.warning("Fallback image was used.")
                else:
                     logger.info("Image generation successful!")
            else:
                logger.error(f"Image file not found at expected path: {full_path}")

        else:
            logger.error("Image generation function returned None.")

    except Exception as e:
        logger.error(f"An error occurred during image generation test: {e}", exc_info=True)

    logger.info("--- Test Finished ---")

if __name__ == "__main__":
    # Windows 환경에서 asyncio 관련 설정 (선택적, run.py와 동일)
    # import os
    # if os.name == "nt":
    #     try:
    #         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    #         logger.info("Windows asyncio policy set.")
    #     except Exception as e:
    #         logger.warning(f"Could not set Windows asyncio policy: {e}")

    asyncio.run(main())