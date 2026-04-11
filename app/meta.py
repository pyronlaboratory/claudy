import tomllib
from pathlib import Path

from app.config import MODEL


def _load_project_meta() -> dict[str, str]:
    for directory in [Path(__file__).parent, Path(__file__).parent.parent]:
        candidate = directory / "pyproject.toml"
        if candidate.exists():
            with open(candidate, "rb") as f:
                data = tomllib.load(f)
            project = data.get("project", {})
            return {
                "name": project.get("name", "claudy"),
                "version": project.get("version", "0.0.0"),
            }
    return {"name": "claudy", "version": "0.0.0"}


PROJECT = _load_project_meta()


def build_system_prompt(author: str = "Ronnie(github@pyronlaboratory)") -> str:
    name = PROJECT["name"].capitalize()
    version = PROJECT["version"]
    return (
        f"You are {name} v{version}, a command-line AI assistant built by {author}."
        f"You are powered by large language model APIs (currently using {MODEL}) via OpenRouter, "
        f"and you have access to tools for reading/writing files, running shell commands, "
        f"searching the filesystem, and fetching URLs. "
        f"When asked about your identity, describe yourself in these terms — "
        f"do not refer to yourself as any specific underlying model or assistant. "
        f"Be concise and developer-focused in your responses."
    )
