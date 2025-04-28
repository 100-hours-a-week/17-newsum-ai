# app/services/image_server_client.py

import httpx
from app.config.settings import settings

async def call_image_generation(prompt: str, control_image_url: str = None) -> dict:
    """
    이미지 생성 서버에 요청을 보내는 함수.
    ControlNet용 참조 이미지 URL을 같이 넘길 수도 있음.
    """
    url = f"{settings.image_server_url}/v1/image/generate"
    payload = {
        "prompt": prompt,
        "control_image_url": control_image_url
    }
    
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
