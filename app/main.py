import argparse
import glob
import json
import os
import subprocess
import sys
import urllib.request

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

MODEL = "anthropic/claude-haiku-4.5"
MAX_TOOL_OUTPUT_LENGTH = 10000
MAX_MESSAGES = 20

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
        if content.startswith("Error reading file:"):
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


def summarize_content(client, content):
    prompt = f"Please provide a concise summary of the following content, focusing on key technical details relevant for a developer:\n\n{content[:MAX_TOOL_OUTPUT_LENGTH]}"
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error summarizing content: {e}"


def cap_messages(messages):
    if len(messages) <= MAX_MESSAGES:
        return messages
    
    # Keep the first message (usually system/initial user prompt) and the last N-1 messages
    return [messages[0]] + messages[-(MAX_MESSAGES-1):]


def agent_loop(client, messages):
    while True:
        messages = cap_messages(messages)
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=2048,
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
            print()

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
                
                # If it was a Fetch call, summarize the result if it's large
                func_name = tc.function.name if hasattr(tc, "function") else tc["function"]["name"]
                if func_name == "Fetch" and len(result) > 1000:
                    print(f"\n[Summarizing fetched content...]", file=sys.stderr)
                    result = summarize_content(client, result)

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
