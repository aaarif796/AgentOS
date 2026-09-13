from __future__ import annotations

from uuid import uuid4

from agentos.memory.models import MemoryEntry, MemoryKind
from agentos.memory.store import MemoryStore


class ShortTermMemory:
    """Session / working memory: a bounded, per-context transcript of exchanges."""

    def __init__(self, store: MemoryStore, max_entries: int = 40) -> None:
        self.store = store
        self.max_entries = max_entries

    async def remember(self, scope: str, role: str, content: str) -> MemoryEntry:
        entry = MemoryEntry(
            id=uuid4().hex,
            kind=MemoryKind.SHORT_TERM,
            scope=scope,
            content=content,
            summary=f"{role} message",
            metadata={"role": role},
        )
        await self.store.insert(entry)
        # ring buffer: drop the oldest entries beyond the cap
        existing = await self.store.list_scope(MemoryKind.SHORT_TERM, scope, limit=10_000)
        overflow = len(existing) - self.max_entries
        for old in existing[: max(overflow, 0)]:
            await self.store.delete(old.id)
        return entry

    async def recall(self, scope: str, limit: int = 20) -> list[MemoryEntry]:
        """Latest `limit` entries for a scope, in chronological (ASC) order."""
        entries = await self.store.list_scope(MemoryKind.SHORT_TERM, scope, limit=10_000)
        return entries[-limit:] if limit < len(entries) else entries

    async def format(self, scope: str, limit: int = 10) -> str:
        entries = await self.recall(scope, limit=limit)
        return "\n".join(f"[{e.metadata.get('role', '?')}] {e.content}" for e in entries)
