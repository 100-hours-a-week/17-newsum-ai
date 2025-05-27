# ai/app/nodes_v2/n01_initialize_node.py
"""
N01InitializeNode for *section-based* WorkflowState.
- Populates `meta`, `query`, `search`, `report` … subsections instead of flat keys.
- Leaves every legacy field value intact by storing them **inside** the proper subsection.
- Returns a partial-dict that LangGraph merges into the main state.
"""

from __future__ import annotations

import uuid
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.workflows.state_v2 import WorkflowState  # ✨ sectioned model
from app.utils.logger import get_logger, summarize_for_logging

logger = get_logger(__name__)

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 512


class N01InitializeNode:
    """Initialises the section‑based WorkflowState and validates the incoming query."""

    async def run(self, state: WorkflowState) -> Dict[str, Any]:
        node_name = self.__class__.__name__

        # ------------------------------------------------------------------
        # 0. Gather incoming values (may be pre‑set by background_tasks)
        # ------------------------------------------------------------------
        original_query: str = state.query.original_query or ""
        initial_config: Dict[str, Any] = state.config.config or {}

        # IDs might already be present in meta; if not, generate.
        comic_id: Optional[str] = state.meta.comic_id or str(uuid.uuid4())
        trace_id: Optional[str] = state.meta.trace_id or comic_id

        timestamp = datetime.now(timezone.utc).isoformat()
        writer_id = initial_config.get("writer_id", "default_writer")

        extra_log = {
            "trace_id": trace_id,
            "comic_id": comic_id,
            "writer_id": writer_id,
            "node_name": node_name,
            "retry_count": 0,
        }

        logger.info(
            f"[{node_name}] Entered. Query: '{original_query}' | Config: {summarize_for_logging(initial_config, 200)}",
            extra=extra_log,
        )

        # ------------------------------------------------------------------
        # 1. Validate input query
        # ------------------------------------------------------------------
        errors = []
        if not original_query:
            errors.append("Initial query is empty or missing.")
        elif len(original_query) < MIN_QUERY_LENGTH:
            errors.append(f"Query too short (<{MIN_QUERY_LENGTH}).")
        elif len(original_query) > MAX_QUERY_LENGTH:
            errors.append(f"Query too long (>{MAX_QUERY_LENGTH}).")

        if errors:
            msg = " ".join(errors)
            logger.error(f"[{node_name}] Validation failed: {msg}", extra=extra_log)
            return {
                "meta": {
                    "trace_id": trace_id,
                    "comic_id": comic_id,
                    "timestamp": timestamp,
                    "current_stage": "ERROR",
                    "error_message": f"N01 Validation Error: {msg}",
                    "error_log": [
                        {"stage": node_name, "error": msg, "timestamp": timestamp}
                    ],
                },
                "query": {
                    "original_query": original_query,
                },
                "config": {"config": initial_config},
            }

        # ------------------------------------------------------------------
        # 2. Build update‑dict for successful init
        # ------------------------------------------------------------------
        update_dict: Dict[str, Any] = {
            "meta": {
                "trace_id": trace_id,
                "comic_id": comic_id,
                "timestamp": timestamp,
                "current_stage": node_name,
                "retry_count": 0,
                "error_log": [],
                "error_message": None,
            },
            "query": {
                "original_query": original_query,
                "query_context": {},
                "initial_context_results": [],
            },
            "search": {
                "search_strategy": None,
                "raw_search_results": None,
            },
            "report": {
                "report_content": None,
                "saved_report_path": None,
                "hitl_status": None,
                "hitl_revision_history": [],
            },
            "idea": {"comic_ideas": [], "selected_comic_idea_for_scenario": None},
            "scenario": {"comic_scenarios": [], "thumbnail_image_prompt": None},
            "image": {"generated_comic_images": []},
            "upload": {
                "uploaded_image_urls": [],
                "uploaded_report_s3_uri": None,
                "uploaded_translated_report_s3_uri": None,
            },
            # keep full config inside its section
            "config": {"config": initial_config},
        }

        logger.info(
            f"[{node_name}] Initialization complete.",
            extra=extra_log,
        )
        return update_dict
