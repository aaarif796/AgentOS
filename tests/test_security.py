from pathlib import Path

import pytest

from agentos.models import AgentSpec
from agentos.security import PermissionDenied, SecurityEngine


def test_workspace_escape_denied(tmp_path: Path) -> None:
    s = SecurityEngine(tmp_path)
    a = AgentSpec(id="x", mission="x", tools=["filesystem"])
    with pytest.raises(PermissionDenied):
        s.check_tool(a, "filesystem", {"path": "../secret.txt"})
