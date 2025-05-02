# app/agents/test_entry.py
import logging
from typing import Dict, Any
from app.workflows.state import ComicState

logger = logging.getLogger(__name__)

class TestEntryAgent:
    """
    Entry point for testing — sets up placeholder humor_texts in English.
    """
    async def run(self, state: ComicState) -> Dict[str, Any]:
        logger.info("[TestEntryAgent] Setting up test humor texts")

        return {
            "humor_texts": [
                "At a climate conference, leaders argue while ignoring a burning Earth.",
                "A scientist looks at a thermometer outside and starts sweating in panic.",
                "A firefighter tries to enter the conference hall but is blocked by politicians.",
                "Citizens sigh as they look at a building billowing smoke.",
                "(Thumbnail) A burning Earth model is placed in the center of the conference room."
            ],
            "error_message": None,
            "lora_style": state.lora_style  # 👈 이 줄이 핵심!
        }
