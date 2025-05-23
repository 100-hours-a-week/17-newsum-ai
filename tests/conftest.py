# tests/conftest.py
import pytest
import asyncio
from unittest.mock import AsyncMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_llm_service():
    """Mock LLM Service for testing"""
    service = AsyncMock()
    service.generate_response.return_value = "Mock LLM response"
    return service


@pytest.fixture
async def mock_image_service():
    """Mock Image Service for testing"""
    service = AsyncMock()
    service.generate_image.return_value = {
        "image_path": "/mock/path/image.png",
        "status": "success"
    }
    return service
