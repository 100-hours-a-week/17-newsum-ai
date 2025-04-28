# tests/test_services/test_llm_server_client.py

import pytest
from app.services.llm_server_client import call_llm_generate

@pytest.mark.asyncio
async def test_call_llm_generate(monkeypatch):
    async def mock_post(prompt):
        return {"summary": "Mock Summary"}
    monkeypatch.setattr("app.services.llm_server_client.call_llm_generate", mock_post)

    response = await call_llm_generate("테스트 프롬프트")
    assert "summary" in response
