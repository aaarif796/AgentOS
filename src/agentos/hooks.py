from __future__ import annotations

from typing import Protocol

from .events import Event, EventBus


class Hook(Protocol):
    name: str
    events: tuple[str, ...]

    def handle(self, event: Event) -> None: ...


class HookManager:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.hooks: dict[str, Hook] = {}

    def register(self, hook: Hook) -> None:
        self.hooks[hook.name] = hook
        for event in hook.events:
            self.bus.subscribe(event, hook.handle)

    def names(self) -> list[str]:
        return sorted(self.hooks)
