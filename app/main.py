import argparse

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from app.agent import agent_loop
from app.meta import build_system_prompt
from app.config import API_KEY, BASE_URL
from app.stats import SessionStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", help="Prompt string")
    group.add_argument("--file", help="Path to prompt file")
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


def main() -> None:
    args = parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt = args.p if args.p else open(args.file).read()
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=build_system_prompt()),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]
    stats = SessionStats()

    try:
        agent_loop(client, messages, stats, args)
    finally:
        if args.session_stats:
            stats.render_session_stats()


if __name__ == "__main__":
    main()
