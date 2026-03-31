import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field

import tiktoken
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
        self.calls = []
        self.start_time = time.time()

    def add_call(self, model, input_tokens, output_tokens, latency):
        pricing = PRICING.get(model, DEFAULT_PRICING)
        cost = (input_tokens * pricing["input"] / 1_000_000) + (output_tokens * pricing["output"] / 1_000_000)
        self.calls.append(CallStats(model, input_tokens, output_tokens, latency, cost))

    def get_total_stats(self):
        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)
        total_cost = sum(c.cost for c in self.calls)
        total_latency = sum(c.latency for c in self.calls)
        return total_input, total_output, total_cost, total_latency

    def render_summary(self, last_call: CallStats):
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("Input Tokens", f"[cyan]{last_call.input_tokens}[/cyan]")
        table.add_row("Output Tokens", f"[cyan]{last_call.output_tokens}[/cyan]")
        table.add_row("Total Tokens", f"[cyan]{last_call.input_tokens + last_call.output_tokens}[/cyan]")
        
        cost_color = "green" if last_call.cost < 0.01 else "yellow"
        table.add_row("Cost", f"[{cost_color}]${last_call.cost:.6f}[/{cost_color}]")
        table.add_row("Latency", f"{last_call.latency:.2f}s")

        console.print(Panel(table, title="[bold]Response Stats[/bold]", expand=False, border_style="dim"))

    def render_session_stats(self):
        if not self.calls:
            return

        table = Table(title="[bold not italic]Session Statistics[/]", title_justify="left", show_footer=True)
        table.add_column("Model", footer="Total")
        table.add_column("Input", justify="right", footer=str(sum(c.input_tokens for c in self.calls)))
        table.add_column("Output", justify="right", footer=str(sum(c.output_tokens for c in self.calls)))
        table.add_column("Cost", justify="right", footer=f"${sum(c.cost for c in self.calls):.6f}")
        table.add_column("Latency", justify="right", footer=f"{sum(c.latency for c in self.calls):.2f}s")

        # Group by model
        by_model = defaultdict(lambda: {"input": 0, "output": 0, "cost": 0.0, "latency": 0.0})
        for c in self.calls:
            by_model[c.model]["input"] += c.input_tokens
            by_model[c.model]["output"] += c.output_tokens
            by_model[c.model]["cost"] += c.cost
            by_model[c.model]["latency"] += c.latency

        # Sort by cost descending
        sorted_models = sorted(by_model.items(), key=lambda x: x[1]["cost"], reverse=True)

        for model, stats in sorted_models:
            table.add_row(
                model,
                str(stats["input"]),
                str(stats["output"]),
                f"${stats["cost"]:.6f}",
                f"{stats["latency"]:.2f}s"
            )

        console.print(table)

def count_tokens(text, model=MODEL):
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o") # Fallback for estimation
        return len(encoding.encode(text))
    except:
        return len(text) // 4 # Very rough fallback

def get_messages_tokens(messages):
    total = 0
    for m in messages:
        if m.get("content"):
            total += count_tokens(m["content"])
        if m.get("tool_calls"):
            total += count_tokens(json.dumps(m["tool_calls"]))
    return total

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Surgical targeted file edits by replacing an old string with a new string",
            "parameters": {
                "type": "object",
                "required": ["file_path", "old_str", "new_str"],
                "properties": {
                    "file_path": {"type": "string", "description": "The path to the file"},
                    "old_str": {"type": "string", "description": "The exact string to be replaced"},
                    "new_str": {"type": "string", "description": "The string to replace with"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Search for files using glob patterns",
            "parameters": {
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {"type": "string", "description": "The glob pattern (e.g. '**/*.py')"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Tree",
            "description": "Recursive directory listing for project orientation",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "The directory path to list"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "Fetch",
            "description": "Retrieve content from a URL",
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"}
                }
            }
        }
    }
]


def read_file(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(file_path, content):
    try:
        with open(file_path, "w") as f:
            f.write(content)
        return "File written successfully"
    except Exception as e:
        return f"Error writing file: {e}"


def edit_file(file_path, old_str, new_str):
    try:
        content = read_file(file_path)
        if isinstance(content, str) and content.startswith("Error reading file:"):
            return content
        
        if old_str not in content:
            return f"Error: '{old_str}' not found in {file_path}"
        
        new_content = content.replace(old_str, new_str)
        return write_file(file_path, new_content)
    except Exception as e:
        return f"Error editing file: {e}"


def glob_files(pattern):
    try:
        files = glob.glob(pattern, recursive=True)
        return "\n".join(files) if files else "No files found"
    except Exception as e:
        return f"Error running glob: {e}"


def tree_dir(path):
    try:
        output = []
        for root, dirs, files in os.walk(path):
            level = root.replace(path, "").count(os.sep)
            indent = " " * 4 * level
            output.append(f"{indent}{os.path.basename(root)}/")
            sub_indent = " " * 4 * (level + 1)
            for f in files:
                output.append(f"{sub_indent}{f}")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing directory: {e}"


def fetch_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        return f"Error fetching URL: {e}"


def run_bash(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )
        output = (result.stdout + result.stderr).strip()
        return output or "Command executed successfully"
    except Exception as e:
        return f"Error executing bash command: {e}"


def execute_tool(tool_call):
    if hasattr(tool_call, "function"):
        func_name = tool_call.function.name
        args_str = tool_call.function.arguments
    else:
        func_name = tool_call["function"]["name"]
        args_str = tool_call["function"]["arguments"]
    
    args = json.loads(args_str)

    if func_name == "Read":
        result = read_file(args["file_path"])
    elif func_name == "Write":
        result = write_file(args["file_path"], args["content"])
    elif func_name == "Bash":
        result = run_bash(args["command"])
    elif func_name == "Edit":
        result = edit_file(args["file_path"], args["old_str"], args["new_str"])
    elif func_name == "Glob":
        result = glob_files(args["pattern"])
    elif func_name == "Tree":
        result = tree_dir(args["path"])
    elif func_name == "Fetch":
        result = fetch_url(args["url"])
    else:
        raise RuntimeError(f"Unknown tool {func_name}")

    if len(result) > MAX_TOOL_OUTPUT_LENGTH:
        return result[:MAX_TOOL_OUTPUT_LENGTH] + f"\n... (truncated {len(result) - MAX_TOOL_OUTPUT_LENGTH} characters)"
    return result


def summarize_content(client, content, stats):
    prompt = f"Please provide a concise summary of the following content, focusing on key technical details relevant for a developer:\n\n{content[:MAX_TOOL_OUTPUT_LENGTH]}"
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        latency = time.time() - start_time
        
        usage = response.usage
        if usage:
            stats.add_call(MODEL, usage.prompt_tokens, usage.completion_tokens, latency)
        else:
            in_t = count_tokens(prompt)
            out_t = count_tokens(response.choices[0].message.content)
            stats.add_call(MODEL, in_t, out_t, latency)
            
        return response.choices[0].message.content
    except Exception as e:
        return f"Error summarizing content: {e}"


def cap_messages(messages):
    if len(messages) <= MAX_MESSAGES:
        return messages
    return [messages[0]] + messages[-(MAX_MESSAGES-1):]


def agent_loop(client, messages, stats: SessionStats, args):
    while True:
        messages = cap_messages(messages)
        
        input_tokens = get_messages_tokens(messages)
        
        start_time = time.time()
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=2048,
            stream=True,
            stream_options={"include_usage": True}
        )

        full_content = ""
        tool_calls_dict = {}
        output_tokens = 0

        with Live(auto_refresh=False) as live:
            for chunk in response:
                if not chunk.choices:
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = chunk.usage
                        input_tokens = usage.prompt_tokens
                        output_tokens = usage.completion_tokens
                    continue
                
                delta = chunk.choices[0].delta
                
                if delta.content:
                    full_content += delta.content
                    output_tokens = count_tokens(full_content)
                    live.update(
                        Text.assemble(
                            full_content,
                            f"\n\n[dim italic]Tokens: {output_tokens}[/dim italic]"
                        ),
                        refresh=True
                    )
                
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_dict:
                            tool_calls_dict[idx] = {
                                "id": tc_delta.id,
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            }
                        
                        if tc_delta.id:
                            tool_calls_dict[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_dict[idx]["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_dict[idx]["function"]["arguments"] += tc_delta.function.arguments
                    
                    live.update(
                        Text.assemble(
                            full_content,
                            "\n\n[yellow]Calling tools...[/yellow]"
                        ),
                        refresh=True
                    )
            
            # End of stream: render cleanly
            if full_content:
                live.update(Markdown(full_content), refresh=True)
            else:
                live.update(Text(""), refresh=True)

        latency = time.time() - start_time
        stats.add_call(MODEL, input_tokens, output_tokens or count_tokens(full_content), latency)
        
        if args.response_stats:
            stats.render_summary(stats.calls[-1])

        tool_calls = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())]
        
        assistant_message = {
            "role": "assistant",
            "content": full_content or None,
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        
        messages.append(assistant_message)

        if tool_calls:
            for tc in tool_calls:
                result = execute_tool(tc)
                
                func_name = tc["function"]["name"]
                if func_name == "Fetch" and len(result) > 1000:
                    console.print(f"\n[Summarizing fetched content...]", style="dim italic")
                    result = summarize_content(client, result, stats)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result
                })
        else:
            break


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", help="Prompt string")
    group.add_argument("--file", help="Path to prompt file")
    parser.add_argument("--response-stats", action="store_true", help="Display per-response statistics")
    parser.add_argument("--session-stats", action="store_true", help="Display session statistics at end")
    return parser.parse_args()


def main():
    args = parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    if args.p:
        prompt = args.p
    else:
        with open(args.file, "r") as f:
            prompt = f.read()

    messages = [{"role": "user", "content": prompt}]
    stats = SessionStats()

    try:
        agent_loop(client, messages, stats, args)
    finally:
        if args.session_stats:
            stats.render_session_stats()


if __name__ == "__main__":
    main()
