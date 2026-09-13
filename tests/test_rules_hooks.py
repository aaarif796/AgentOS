from pathlib import Path

from agentos.builtin_hooks import AuditHook
from agentos.events import Event, EventBus
from agentos.hooks import HookManager
from agentos.rules import default_rules


def test_rules_and_audit(tmp_path: Path) -> None:
    bus = EventBus()
    hooks = HookManager(bus)
    audit = AuditHook(tmp_path / "audit.jsonl")
    hooks.register(audit)
    bus.publish(Event("TASK_CREATED", {"task_id": "x"}))
    assert (tmp_path / "audit.jsonl").exists()
    assert "security-first" in default_rules().rules
