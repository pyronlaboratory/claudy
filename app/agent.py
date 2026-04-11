import time
from typing import cast

from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from app.config import MAX_MESSAGES, MAX_TOOL_OUTPUT_LENGTH, MODEL
from app.stats import SessionStats, count_tokens, get_messages_tokens
from app.tools import TOOLS, execute_tool

console = Console()


class AgentError(Exception):
    """Clean, user-facing agent error."""


def summarize_content(client: OpenAI, content: str, stats: SessionStats) -> str:
    prompt = (
        "Please provide a concise summary of the following content, "
        "focusing on key technical details relevant for a developer:\n\n"
        f"{content[:MAX_TOOL_OUTPUT_LENGTH]}"
    )
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        latency = time.time() - start_time
        usage = response.usage
        if usage:
            stats.add_call(MODEL, usage.prompt_tokens, usage.completion_tokens, latency)
        else:
            stats.add_call(
                MODEL,
                count_tokens(prompt),
                count_tokens(response.choices[0].message.content or ""),
                latency,
            )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error summarizing content: {e}"


def _cap_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    if len(messages) <= MAX_MESSAGES:
        return messages
    return [messages[0]] + messages[-(MAX_MESSAGES - 1) :]


def _raise_status_error(e: APIStatusError) -> None:
    """Convert an APIStatusError into a clean AgentError."""
    status = e.status_code
    try:
        detail = e.response.json().get("error", {}).get("message", str(e))
    except Exception:
        detail = str(e)

    if status == 402:
        raise AgentError(f"Out of credits — {detail}")
    elif status == 401:
        raise AgentError("Authentication failed — check your OPENROUTER_API_KEY.")
    elif status == 429:
        raise AgentError("Rate limited — too many requests, slow down and retry.")
    elif status == 503:
        raise AgentError("Service unavailable — the API is down, try again shortly.")
    else:
        raise AgentError(f"API error {status}: {detail}")


def _stream_response(
    client: OpenAI, messages: list[ChatCompletionMessageParam]
) -> tuple[str, list[dict], int, int, float]:
    """
    Stream one LLM turn.

    Returns:
        (full_content, tool_calls, input_tokens, output_tokens, latency)

    Raises:
        AgentError on API-level failures.
    """
    input_tokens = get_messages_tokens(messages)
    start_time = time.time()
    response = None

    try:
        response = client.chat.completions.create(  # type: ignore[call-overload]
            model=MODEL,
            messages=messages,
            tools=cast(list[ChatCompletionToolParam], TOOLS),
            max_tokens=2048,
            stream=True,
            stream_options={"include_usage": True},
        )
    except APIStatusError as e:
        _raise_status_error(e)
    except APITimeoutError:
        raise AgentError("Request timed out — the API took too long to respond.")
    except APIConnectionError:
        raise AgentError("Connection failed — check your network and try again.")

    full_content = ""
    tool_calls_dict: dict[int, dict] = {}
    output_tokens = 0

    try:
        assert response is not None
        with Live(auto_refresh=False, vertical_overflow="visible") as live:
            for chunk in response:
                if not chunk.choices:
                    if hasattr(chunk, "usage") and chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens
                    continue

                delta = chunk.choices[0].delta

                if delta.content:
                    full_content += delta.content
                    output_tokens = count_tokens(full_content)

                    status = Text()
                    status.append(full_content)
                    status.append(f"\n[dim]● {output_tokens} tokens[/dim]")
                    live.update(status, refresh=True)

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {
                                "id": tc_delta.id,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.id:
                            tool_calls_dict[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_dict[idx]["function"]["name"] += (
                                    tc_delta.function.name
                                )
                            if tc_delta.function.arguments:
                                tool_calls_dict[idx]["function"]["arguments"] += (
                                    tc_delta.function.arguments
                                )

                    active_tools = [
                        t["function"]["name"]
                        for t in tool_calls_dict.values()
                        if t["function"]["name"]
                    ]
                    tool_line = Text()
                    if full_content:
                        tool_line.append(full_content + "\n\n")
                    tool_line.append("  running ", style="dim")
                    tool_line.append(", ".join(active_tools), style="yellow")
                    tool_line.append(" …", style="dim")
                    live.update(tool_line, refresh=True)

            if full_content:
                live.update(Markdown(full_content), refresh=True)
            else:
                live.update(Text(""), refresh=True)

    except APIStatusError as e:
        _raise_status_error(e)

    latency = time.time() - start_time
    tool_calls = [tool_calls_dict[i] for i in sorted(tool_calls_dict)]
    return (
        full_content,
        tool_calls,
        input_tokens,
        output_tokens or count_tokens(full_content),
        latency,
    )


def agent_loop(
    client: OpenAI,
    messages: list[ChatCompletionMessageParam],
    stats: SessionStats,
    args,
) -> None:
    while True:
        messages = _cap_messages(messages)

        try:
            full_content, tool_calls, input_tokens, output_tokens, latency = (
                _stream_response(client, messages)
            )
        except AgentError as e:
            console.print(f"\n[red]✗[/red] [bold]{e}[/bold]")
            return

        stats.add_call(MODEL, input_tokens, output_tokens, latency)
        if args.response_stats:
            stats.render_summary(stats.calls[-1])

        assistant_message = ChatCompletionAssistantMessageParam(
            role="assistant", content=full_content or None
        )
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls  # type: ignore[typeddict-unknown-key]
        messages.append(assistant_message)

        if not tool_calls:
            break

        for tc in tool_calls:
            func_name = tc["function"]["name"]

            console.print(
                Text.assemble(
                    ("  ⟳ ", "dim"),
                    (func_name, "yellow"),
                    ("  ", "dim"),
                ),
                end="",
            )

            result = execute_tool(tc)

            if func_name == "Fetch" and len(result) > 1000:
                console.print("[dim]summarizing…[/dim]")
                result = summarize_content(client, result, stats)
            else:
                console.print(Text.assemble(("done", "dim green")))

            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool", tool_call_id=tc["id"], content=result
                )
            )

        console.print()
