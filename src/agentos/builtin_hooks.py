from pathlib import Path

import structlog

from .events import Event

log = structlog.get_logger()


class AuditHook:
    name = "audit"
    events = ("*",)

    def __init__(self, audit_file: Path) -> None:
        self.audit_file = audit_file
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    def handle(self, event: Event) -> None:
        with self.audit_file.open("a", encoding="utf-8") as f:
            import json

            f.write(
                json.dumps({"type": event.type, "data": event.data, "at": event.at.isoformat()})
                + "\n"
            )


class ErrorLogHook:
    name = "error-log"
    events = ("MODEL_FAILED", "TOOL_FAILED", "TASK_FAILED")

    def handle(self, event: Event) -> None:
        log.error("agentos_error_event", event=event.type, **event.data)
