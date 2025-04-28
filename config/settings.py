"""
Configuration settings for the LangGraph project.
"""

# LLM Configuration
LLM_MODEL = "gpt-4"
TEMPERATURE = 0.7
MAX_TOKENS = 1000

# Application settings
DEBUG = True
VERBOSE = True

# API settings
API_TIMEOUT = 60  # seconds
RETRY_ATTEMPTS = 3
