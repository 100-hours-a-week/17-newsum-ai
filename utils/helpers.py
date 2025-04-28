"""
Helper utilities for LangGraph project.
"""
import json
import logging
from typing import Dict, Any, List, Optional, Union


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with the specified level.
    
    Args:
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("langgraph")
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def save_state(state: Dict[str, Any], filepath: str) -> None:
    """Save a state dictionary to a JSON file.
    
    Args:
        state: State dictionary to save
        filepath: Path to save the state to
    """
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)


def load_state(filepath: str) -> Dict[str, Any]:
    """Load a state dictionary from a JSON file.
    
    Args:
        filepath: Path to load the state from
        
    Returns:
        Loaded state dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def merge_states(state1: Dict[str, Any], state2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two state dictionaries.
    
    Args:
        state1: First state dictionary
        state2: Second state dictionary (values override first if there are conflicts)
        
    Returns:
        Merged state dictionary
    """
    result = state1.copy()
    result.update(state2)
    return result


def extract_keys(state: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Extract specific keys from a state dictionary.
    
    Args:
        state: State dictionary
        keys: List of keys to extract
        
    Returns:
        Dictionary with only the specified keys
    """
    return {k: state[k] for k in keys if k in state}
