import os

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

MODEL = "anthropic/claude-haiku-4.5"
MAX_TOOL_OUTPUT_LENGTH = 10000
MAX_MESSAGES = 20

# Prices per 1M tokens
PRICING = {
    "anthropic/claude-haiku-4.5": {"input": 0.25, "output": 1.25},
    "anthropic/claude-3.5-haiku": {"input": 0.25, "output": 1.25},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "anthropic/claude-3-opus": {"input": 15.0, "output": 75.0},
    "openai/gpt-4o": {"input": 2.5, "output": 10.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "google/gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},
}

DEFAULT_PRICING = {"input": 1.0, "output": 1.0}
