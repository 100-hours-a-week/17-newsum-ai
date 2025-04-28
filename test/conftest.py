# tests/conftest.py

import pytest

@pytest.fixture(scope="session")
def example_text():
    return "테스트용 예시 텍스트"
