import json
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import tiktoken
from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from app.config import DEFAULT_PRICING, PRICING

console = Console()


@dataclass
class CallStats:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0
    cost: float = 0.0


class SessionStats:
    def __init__(self):
        self.calls: list[CallStats] = []
        self.start_time = time.time()

    def add_call(
        self, model: str, input_tokens: int, output_tokens: int, latency: float
    ):
        pricing = PRICING.get(model, DEFAULT_PRICING)
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )
        self.calls.append(CallStats(model, input_tokens, output_tokens, latency, cost))

    def render_summary(self, last_call: CallStats):
        """Print a compact single-line response stat summary."""
        total_tokens = last_call.input_tokens + last_call.output_tokens
        cost_color = "green" if last_call.cost < 0.01 else "yellow"

        line = Text()
        line.append("  ↳ ", style="dim")
        line.append(f"in {last_call.input_tokens:,}", style="cyan")
        line.append(" · ", style="dim")
        line.append(f"out {last_call.output_tokens:,}", style="cyan")
        line.append(" · ", style="dim")
        line.append(f"total {total_tokens:,} tok", style="bright_white")
        line.append("  ", style="dim")
        line.append(f"${last_call.cost:.6f}", style=cost_color)
        line.append("  ", style="dim")
        line.append(f"{last_call.latency:.2f}s", style="dim")

        console.print(line)

    def render_session_stats(self):
        if not self.calls:
            return

        elapsed = time.time() - self.start_time
        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)
        total_cost = sum(c.cost for c in self.calls)
        total_latency = sum(c.latency for c in self.calls)
        num_calls = len(self.calls)

        console.print()
        console.print(Rule(" SESSION SUMMARY ", style="bright_black", align="left"))
        console.print()

        # Per-model breakdown table
        by_model: dict[str, dict] = defaultdict(
            lambda: {"calls": 0, "input": 0, "output": 0, "cost": 0.0, "latency": 0.0}
        )
        for c in self.calls:
            by_model[c.model]["calls"] += 1
            by_model[c.model]["input"] += c.input_tokens
            by_model[c.model]["output"] += c.output_tokens
            by_model[c.model]["cost"] += c.cost
            by_model[c.model]["latency"] += c.latency

        table = Table(
            box=box.SIMPLE_HEAD,
            show_footer=True,
            footer_style="bold",
            header_style="dim",
            border_style="bright_black",
            pad_edge=True,
            padding=(0, 2),
        )

        table.add_column("Model", footer="Total", style="white", no_wrap=True)
        table.add_column(
            "Calls",
            justify="right",
            footer=str(num_calls),
            style="cyan",
        )
        table.add_column(
            "Input tok",
            justify="right",
            footer=f"{total_input:,}",
            style="cyan",
        )
        table.add_column(
            "Output tok",
            justify="right",
            footer=f"{total_output:,}",
            style="cyan",
        )
        table.add_column(
            "Avg tok/call",
            justify="right",
            footer=f"{(total_input + total_output) // max(num_calls, 1):,}",
            style="bright_black",
        )
        table.add_column(
            "Cost (USD)",
            justify="right",
            footer=f"${total_cost:.6f}",
            style="green" if total_cost < 0.01 else "yellow",
        )
        table.add_column(
            "Latency",
            justify="right",
            footer=f"{total_latency:.2f}s",
            style="dim",
        )

        sorted_models = sorted(
            by_model.items(), key=lambda x: x[1]["cost"], reverse=True
        )
        for model, stats in sorted_models:
            avg_tok = (stats["input"] + stats["output"]) // max(stats["calls"], 1)
            cost_color = "green" if stats["cost"] < 0.01 else "yellow"
            table.add_row(
                model,
                str(stats["calls"]),
                f"{stats['input']:,}",
                f"{stats['output']:,}",
                f"{avg_tok:,}",
                f"[{cost_color}]${stats['cost']:.6f}[/{cost_color}]",
                f"{stats['latency']:.2f}s",
            )

        console.print(table)

        # Footer metadata line
        avg_latency = total_latency / max(num_calls, 1)
        meta = Text()
        meta.append(f"  {num_calls} call{'s' if num_calls != 1 else ''}", style="dim")
        meta.append("  ·  ", style="bright_black")
        meta.append(f"wall time {elapsed:.1f}s", style="dim")
        meta.append("  ·  ", style="bright_black")
        meta.append(f"avg latency {avg_latency:.2f}s/call", style="dim")
        console.print(meta)
        console.print()


def count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def get_messages_tokens(messages: Sequence[ChatCompletionMessageParam]) -> int:
    total = 0
    for m in messages:
        raw = dict(m)
        content = raw.get("content")
        if content:
            total += count_tokens(
                content if isinstance(content, str) else json.dumps(content)
            )
        tool_calls = raw.get("tool_calls")
        if tool_calls:
            total += count_tokens(json.dumps(tool_calls))
    return total
