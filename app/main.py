import argparse
import json
import os
import subprocess
import sys

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
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if func_name == "Read":
        return read_file(args["file_path"])

    if func_name == "Write":
        return write_file(args["file_path"], args["content"])
    
    if func_name == "Bash":
        return run_bash(args["command"])

    raise RuntimeError(f"Unknown tool {func_name}")


def call_llm(client, messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        max_tokens=1024
    )

    if not response.choices:
        raise RuntimeError("no choices in response")

    return response.choices[0].message


def agent_loop(client, messages):
    while True:
        print("Logs from your program will appear here!", file=sys.stderr)

        message = call_llm(client, messages)
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                result = execute_tool(tool_call)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            print(message.content)
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    messages = [{"role": "user", "content": args.p}]

    agent_loop(client, messages)


if __name__ == "__main__":
    main()
