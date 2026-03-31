import argparse
import json
import os
import subprocess

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

MODEL = "anthropic/claude-haiku-4.5"

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
    }
]


def read_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


def write_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)
    return "File written successfully"


def run_bash(command):
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
    )

    output = (result.stdout + result.stderr).strip()
    return output or "Command executed successfully"


def execute_tool(tool_call):
    # Support both object-like tool calls from non-streaming and dicts from streaming
    if hasattr(tool_call, "function"):
        func_name = tool_call.function.name
        args_str = tool_call.function.arguments
    else:
        func_name = tool_call["function"]["name"]
        args_str = tool_call["function"]["arguments"]
    
    args = json.loads(args_str)

    if func_name == "Read":
        return read_file(args["file_path"])

    if func_name == "Write":
        return write_file(args["file_path"], args["content"])
    
    if func_name == "Bash":
        return run_bash(args["command"])

    raise RuntimeError(f"Unknown tool {func_name}")


def agent_loop(client, messages):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=1024,
            stream=True
        )

        full_content = ""
        tool_calls_dict = {}

        for chunk in response:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            
            if delta.content:
                print(delta.content, end="", flush=True)
                full_content += delta.content
            
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

        if full_content:
            print() # Newline after content stream

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

    agent_loop(client, messages)


if __name__ == "__main__":
    main()
