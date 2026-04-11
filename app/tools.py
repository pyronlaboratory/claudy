import glob
import json
import os
import subprocess
import urllib.request

from app.config import MAX_TOOL_OUTPUT_LENGTH

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
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
                        "description": "The path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
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
                        "description": "The path of the file to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
            },
        },
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
                        "description": "The command to execute",
                    }
                },
            },
        },
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
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The exact string to be replaced",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The string to replace with",
                    },
                },
            },
        },
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
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern (e.g. '**/*.py')",
                    }
                },
            },
        },
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
                    "path": {
                        "type": "string",
                        "description": "The directory path to list",
                    }
                },
            },
        },
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
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
_HANDLERS = {
    "Read": lambda args: read_file(args["file_path"]),
    "Write": lambda args: write_file(args["file_path"], args["content"]),
    "Bash": lambda args: run_bash(args["command"]),
    "Edit": lambda args: edit_file(args["file_path"], args["old_str"], args["new_str"]),
    "Glob": lambda args: glob_files(args["pattern"]),
    "Tree": lambda args: tree_dir(args["path"]),
    "Fetch": lambda args: fetch_url(args["url"]),
}


def execute_tool(tool_call) -> str:
    """Accept either an OpenAI object or a plain dict."""
    if hasattr(tool_call, "function"):
        func_name = tool_call.function.name
        args_str = tool_call.function.arguments
    else:
        func_name = tool_call["function"]["name"]
        args_str = tool_call["function"]["arguments"]

    handler = _HANDLERS.get(func_name)
    if handler is None:
        raise RuntimeError(f"Unknown tool: {func_name}")

    result = handler(json.loads(args_str))

    if len(result) > MAX_TOOL_OUTPUT_LENGTH:
        truncated = len(result) - MAX_TOOL_OUTPUT_LENGTH
        return (
            result[:MAX_TOOL_OUTPUT_LENGTH]
            + f"\n... (truncated {truncated} characters)"
        )
    return result
