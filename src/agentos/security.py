from pathlib import Path

from .models import AgentSpec


class PermissionDenied(Exception):
    pass


class SecurityEngine:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def check_tool(self, agent: AgentSpec, tool_id: str, args: dict[str, object]) -> None:
        if tool_id not in agent.tools:
            raise PermissionDenied(f"{agent.id} is not allowed to use {tool_id}")
        policy = agent.permissions.get(tool_id, {})
        if isinstance(policy, dict) and policy.get("deny"):
            raise PermissionDenied(f"policy denies {tool_id}")
        if tool_id == "terminal":
            command = str(args.get("command", ""))
            blocked = ("rm -rf /", "mkfs", "shutdown", "reboot", ":(){:|:&};:")
            if any(x in command for x in blocked):
                raise PermissionDenied("dangerous command blocked")
        if tool_id == "filesystem":
            path = (self.workspace / str(args.get("path", ""))).resolve()
            if path != self.workspace and self.workspace not in path.parents:
                raise PermissionDenied("filesystem path escapes workspace")
