from __future__ import annotations

import re
from uuid import uuid4

from agentos.memory.models import MemoryEntry, MemoryKind
from agentos.memory.store import MemoryStore


class ProceduralMemory:
    """How-to memory: recipes, validated workflows and skill instructions."""

    def __init__(self, store: MemoryStore, top_k: int = 5) -> None:
        self.store = store
        self.top_k = top_k

    async def add_recipe(
        self,
        name: str,
        instructions: str,
        triggers: list[str],
        verified: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> MemoryEntry:
        meta: dict[str, object] = {"verified": verified, "triggers": triggers}
        if metadata:
            meta.update(metadata)
        return await self.store_insert(name, instructions, meta)

    async def store_insert(
        self, name: str, instructions: str, meta: dict[str, object]
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=uuid4().hex,
            kind=MemoryKind.PROCEDURAL,
            scope=name,
            content=instructions,
            summary=instructions[:200],
            metadata=meta,
        )
        await self.store.insert(entry)
        return entry

    async def recall(self, goal: str, k: int | None = None) -> list[MemoryEntry]:
        recipes = await self.store.list_kind(MemoryKind.PROCEDURAL, limit=500)
        words = {w for w in re.findall(r"[a-z0-9]+", goal.lower()) if len(w) > 2}
        scored: list[tuple[float, MemoryEntry]] = []
        for r in recipes:
            triggers = [str(t).lower() for t in r.metadata.get("triggers", [])]
            score: float = sum(1 for t in triggers if t in goal.lower())
            score += sum(1 for w in words if w in r.content.lower() or w in r.summary.lower()) * 0.5
            if r.metadata.get("verified"):
                score += 1.0
            scored.append((score, r))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        limit = k if k is not None else self.top_k
        return [r for score, r in scored[:limit] if score > 0]

    async def format_context(self, entries: list[MemoryEntry]) -> str:
        if not entries:
            return ""
        lines = []
        for e in entries:
            marker = "VERIFIED" if e.metadata.get("verified") else "candidate"
            lines.append(f"## {e.scope} [{marker}]\n{e.content}\n")
        return "\n".join(lines)
