import argparse

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.agent import agent_loop
from app.config import DEFAULT_PROVIDER, PROVIDERS, resolve_provider
from app.meta import build_system_prompt
from app.stats import SessionStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="claudy — a command-line AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )

    # Prompt input (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", metavar="PROMPT", help="Prompt string")
    group.add_argument("--file", metavar="PATH", help="Path to prompt file")

    # Provider & model routing
    parser.add_argument(
        "--provider",
        metavar="NAME",
        default=DEFAULT_PROVIDER,
        choices=list(PROVIDERS.keys()),
        help=(
            f"LLM provider to use (default: {DEFAULT_PROVIDER}). "
            f"Choices: {', '.join(PROVIDERS.keys())}."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        metavar="MODEL",
        default=None,
        help=(
            "Model identifier to use. Overrides the provider's default model. "
            "Examples: gpt-4o, claude-3-5-sonnet-20241022, gemini-1.5-pro, "
            "anthropic/claude-3.5-sonnet (OpenRouter)."
        ),
    )

    # Stats flags
    parser.add_argument(
        "--response-stats",
        action="store_true",
        help="Display per-response statistics",
    )
    parser.add_argument(
        "--session-stats",
        action="store_true",
        help="Display session statistics at end",
    )

    return parser.parse_args()


def _epilog() -> str:
    lines = ["Provider defaults and required env vars:", ""]
    for name, spec in PROVIDERS.items():
        lines.append(
            f"  {name:<12}  model={spec['default_model']:<40}  key={spec['api_key_env']}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # Resolve provider config (validates API key presence)
    provider_cfg = resolve_provider(args.provider, model_override=args.model)

    client = OpenAI(api_key=provider_cfg.api_key, base_url=provider_cfg.base_url)

    prompt = args.p if args.p else open(args.file).read()
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=build_system_prompt(model=provider_cfg.model),
        ),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]
    stats = SessionStats()

    try:
        agent_loop(
            client,
            messages,
            stats,
            args,
            model=provider_cfg.model,
            provider=provider_cfg.name,
        )
    finally:
        if args.session_stats:
            stats.render_session_stats()


if __name__ == "__main__":
    main()
