# tests/test_tools/test_summarizer_tool.py

import pytest
from app.tools.llm.summarizer_tool import summarize_text

@pytest.mark.asyncio
async def test_summarizer_tool():
    summary = await summarize_text("테스트 입력 텍스트")
    assert isinstance(summary, str)
