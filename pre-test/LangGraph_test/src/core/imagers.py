# src/core/imagers.py (수정: 외부 SD API 사용)
import aiohttp
import base64
import io
from PIL import Image
import asyncio
from .schemas import ImagePromptResult, ImageRenderResult
from .utils import logger, save_image
from src import settings
from pathlib import Path
from datetime import datetime

async def generate_image_via_api(prompt_result: ImagePromptResult) -> ImageRenderResult:
    """외부 Stable Diffusion API를 호출하여 이미지 생성"""
    logger.info(f"Requesting image generation via API for {prompt_result.news_id}...")
    api_url = f"{settings.SD_API_BASE_URL}/predictions"

    payload = {
        "prompt": prompt_result.positive_prompt,
        "negative_prompt": prompt_result.negative_prompt,
        "steps": settings.SD_DEFAULT_STEPS,
        "cfg_scale": settings.SD_DEFAULT_CFG_SCALE,
        "width": settings.SD_DEFAULT_WIDTH,
        "height": settings.SD_DEFAULT_HEIGHT,
        # "seed": random.randint(0, 2**32 - 1) # 필요시 랜덤 시드
    }

    output_filename = f"{prompt_result.news_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    output_path = settings.IMAGE_SAVE_DIR / output_filename
    relative_image_path = str(output_path.relative_to(settings.OUTPUT_DIR))
    fallback_relative_path = str(Path(settings.FALLBACK_IMAGE_PATH).relative_to(settings.OUTPUT_DIR))

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload, timeout=180) as response: # 타임아웃 늘리기 (3분)
                if response.status == 200:
                    result_data = await response.json()
                    if "image_base64" in result_data:
                        img_b64 = result_data["image_base64"]
                        img_bytes = base64.b64decode(img_b64)
                        pil_image = Image.open(io.BytesIO(img_bytes))

                        # 이미지 저장 (비동기 실행 필요 시 to_thread 사용)
                        # save_image는 동기 함수이므로 바로 호출
                        save_image(pil_image, output_path)

                        logger.info(f"Image generated via API and saved for {prompt_result.news_id} at {output_path}")
                        return ImageRenderResult(
                            news_id=prompt_result.news_id,
                            image_path=relative_image_path,
                            is_fallback=False
                        )
                    else:
                        logger.error(f"SD API response missing 'image_base64' for {prompt_result.news_id}")
                        return ImageRenderResult(news_id=prompt_result.news_id, image_path=fallback_relative_path, is_fallback=True)
                else:
                    error_text = await response.text()
                    logger.error(f"SD API request failed for {prompt_result.news_id}. Status: {response.status}, Body: {error_text[:200]}")
                    return ImageRenderResult(news_id=prompt_result.news_id, image_path=fallback_relative_path, is_fallback=True)
    except asyncio.TimeoutError:
        logger.error(f"SD API request timed out for {prompt_result.news_id}. Using fallback.")
        return ImageRenderResult(news_id=prompt_result.news_id, image_path=fallback_relative_path, is_fallback=True)
    except Exception as e:
        logger.error(f"SD API request unexpected error for {prompt_result.news_id}: {e}. Using fallback.")
        return ImageRenderResult(news_id=prompt_result.news_id, image_path=fallback_relative_path, is_fallback=True)

# generate_image 함수 이름을 유지하되 내부에서 API 호출 함수 사용
async def generate_image(prompt_result: ImagePromptResult) -> ImageRenderResult:
     return await generate_image_via_api(prompt_result)