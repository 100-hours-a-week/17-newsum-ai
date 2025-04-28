# tests/test_agents/test_collector_agent.py

import pytest
from app.agents.collector_agent import CollectorAgent
from app.workflows.state import ComicState

@pytest.mark.asyncio
async def test_collector_agent_run():
    agent = CollectorAgent()
    state = ComicState()
    new_state = await agent.run(state)
    assert isinstance(new_state, ComicState)
