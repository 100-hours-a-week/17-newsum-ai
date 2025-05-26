# ai/app/nodes_v2/show_state_node.py
from typing import Dict, Any
from app.workflows.state_v2 import WorkflowState
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ShowStateNode:
    def __init__(self):
        pass

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        logger.debug("🧾 Final state dump:\n%s", state.model_dump_json(indent=2))
        return state.model_dump()
