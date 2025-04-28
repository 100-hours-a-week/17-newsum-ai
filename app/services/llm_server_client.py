# app/services/llm_server_client.py

import httpx
from app.config.settings import settings

async def call_llm_generate(prompt: str) -> dict:
    """
    LLM 서버에 텍스트 생성을 요청하는 함수.
    """
    url = f"{settings.llm_server_url}/v1/llm/generate"
    payload = {"prompt": prompt}
    
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
