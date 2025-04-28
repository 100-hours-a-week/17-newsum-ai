# tests/test_workflows/test_main_workflow.py

import pytest
from app.workflows.main_workflow import build_main_workflow
from app.workflows.state import ComicState

@pytest.mark.asyncio
async def test_main_workflow_build():
    workflow = build_main_workflow()
    assert workflow is not None

@pytest.mark.asyncio
async def test_main_workflow_execution():
    workflow = build_main_workflow()
    initial_state = ComicState()
    result_state = await workflow.astart(initial_state)
    assert isinstance(result_state, ComicState)
