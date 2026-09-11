from __future__ import annotations

import subprocess
from pathlib import Path

import httpx

from .models import AgentSpec, ToolSpec
from .security import SecurityEngine


class ToolRuntime:
    def __init__(self, workspace: Path, security: SecurityEngine, timeout: int = 30) -> None:
        self.workspace = workspace.resolve()
        self.security = security
        self.timeout = timeout
        self.specs = [
            ToolSpec(
                id="filesystem",
                description="Read/write workspace files",
                risk="medium",
                capabilities=["files"],
            ),
            ToolSpec(
                id="terminal",
                description="Run restricted workspace commands",
                risk="high",
                capabilities=["shell"],
            ),
            ToolSpec(
                id="git", description="Inspect Git repository", risk="medium", capabilities=["git"]
            ),
            ToolSpec(id="http", description="HTTP GET", risk="medium", capabilities=["network"]),
        ]

    def execute(self, agent: AgentSpec, tool_id: str, args: dict[str, object]) -> str:
        self.security.check_tool(agent, tool_id, args)
        if tool_id == "filesystem":
            path = (self.workspace / str(args["path"])).resolve()
            action = str(args.get("action", "read"))
            if action == "read":
                return path.read_text(encoding="utf-8")
            if action == "write":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(args["content"]), encoding="utf-8")
                return "written"
            raise ValueError("filesystem action must be read or write")
        if tool_id == "terminal":
            proc = subprocess.run(
                str(args["command"]),
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            return proc.stdout + proc.stderr
        if tool_id == "git":
            cmd = str(args.get("command", "status --short"))
            allowed = ("status", "diff", "log", "branch", "show")
            if not cmd.startswith(allowed):
                raise PermissionError("Git command not permitted")
            proc = subprocess.run(
                ["git", *cmd.split()],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            return proc.stdout + proc.stderr
        if tool_id == "http":
            response = httpx.get(str(args["url"]), timeout=20, follow_redirects=True)
            response.raise_for_status()
            return response.text[:100_000]
        raise KeyError(tool_id)
