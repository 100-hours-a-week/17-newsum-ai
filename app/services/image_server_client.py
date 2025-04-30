
# app/services/image_server_client.py

import httpx
import os
import logging
from app.config.settings import settings

logger = logging.getLogger(__name__)

async def generate_single_image(prompt: str, idx: int = 0) -> str:
    """
    단일 prompt로 이미지를 생성하는 함수
    - prompt: 생성할 프롬프트
    - idx: 이미지 번호 (seed와 파일명에 사용)
    """
    url = f"{settings.IMAGE_SERVER_URL}/generate/text-to-image"

    payload = {
        "prompt": prompt,
        "negative_prompt": "",
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
        "seed": 42 + idx,  # 컷마다 다른 seed
    }

    try:
        logger.info(f"📤 {idx+1}번째 이미지 요청: {payload['prompt'][:40]}...")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=payload)

        response.raise_for_status()  # 4xx, 5xx 오류 시 예외 발생

        save_path = f"./generated_image_{idx+1}.png"
        with open(save_path, "wb") as f:
            f.write(response.content)

        logger.info(f"✅ {idx+1}번째 이미지 저장 완료: {save_path}")
        return save_path

    except httpx.TimeoutException:
        logger.error(f"⏰ {idx+1}번째 이미지 요청 타임아웃 발생 (TimeoutException)")
        raise RuntimeError("Image generation request timed out.")

    except httpx.RequestError as e:
        logger.error(f"🌐 {idx+1}번째 이미지 요청 실패: {str(e)}")
        raise RuntimeError(f"Network error during image generation: {str(e)}")

    except httpx.HTTPStatusError as e:
        logger.error(f"❌ {idx+1}번째 이미지 생성 실패: HTTP {e.response.status_code}")
        raise RuntimeError(f"HTTP error {e.response.status_code}: {e.response.text}")

    except Exception as e:
        logger.exception(f"⚠️ {idx+1}번째 이미지 생성 중 알 수 없는 에러 발생")
        raise RuntimeError(f"Unexpected error during image generation: {str(e)}")

