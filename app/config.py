import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-haiku-4.5",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-4-5-20251001",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
}

DEFAULT_PROVIDER = os.getenv("CLAUDY_PROVIDER", "openrouter")

MAX_TOOL_OUTPUT_LENGTH = 10000
MAX_MESSAGES = 20

# ---------------------------------------------------------------------------
# Prices per 1M tokens
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    # OpenRouter-routed
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "anthropic/claude-3.5-haiku": {"input": 0.80, "output": 4.0},
    "anthropic/claude-4.5-haiku": {"input": 1.0, "output": 5.0},
    "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    "anthropic/claude-4.6-opus": {"input": 5.0, "output": 25.0},
    "openai/gpt-4o": {"input": 2.5, "output": 10.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "google/gemini-2.0-flash": {"input": 0.10, "output": 0.4},
    # Native OpenAI
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "o1": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 3.0, "output": 12.0},
    # Native Anthropic
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-4-5-haiku-20251001": {"input": 1.0, "output": 5.0},
    "claude-4-6-opus-20251210": {"input": 5.0, "output": 25.0},
    # Native Gemini
    "gemini-2.0-flash": {"input": 0.10, "output": 0.4},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.3},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.3},
}

DEFAULT_PRICING: dict[str, float] = {"input": 1.0, "output": 1.0}

# Conservative max_tokens per provider.
# OpenRouter free tier charges against a credit balance — a high max_tokens
# reservation can exhaust it even on cheap models. Native providers are more
# forgiving but we keep sane defaults so users can override via --max-tokens.
PROVIDER_MAX_TOKENS: dict[str, int] = {
    "openrouter": 512,  # very conservative to avoid credit reservation lock
    "openai": 1024,
    "anthropic": 1024,
    "gemini": 1024,
}

# ---------------------------------------------------------------------------
# Runtime-resolved provider config
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str


def resolve_provider(
    provider_name: str, model_override: str | None = None
) -> ProviderConfig:
    """
    Resolve a provider name to a ProviderConfig, raising RuntimeError if the
    required API key env var is not set.
    """
    provider_name = provider_name.lower()
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise RuntimeError(
            f"Unknown provider '{provider_name}'. Available: {available}"
        )

    spec = PROVIDERS[provider_name]
    env_var = spec["api_key_env"]
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"Provider '{provider_name}' requires {env_var} to be set.")

    model = model_override or spec["default_model"]

    return ProviderConfig(
        name=provider_name,
        base_url=spec["base_url"],
        api_key=api_key,
        model=model,
    )
