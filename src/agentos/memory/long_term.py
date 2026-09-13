from __future__ import annotations

from uuid import uuid4

from agentos.memory.embedder import Embedder, TFIDFEmbedder
from agentos.memory.models import MemoryEntry, MemoryKind
from agentos.memory.store import MemoryStore


class LongTermMemory:
    """Episodic / semantic memory: facts, lessons and research stored with embeddings."""

    def __init__(self, store: MemoryStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or TFIDFEmbedder()

    async def add(
        self,
        content: str,
        scope: str = "global",
        summary: str = "",
        metadata: dict[str, object] | None = None,
    ) -> MemoryEntry:
        vectors = self.embedder.embed([content])
        entry = MemoryEntry(
            id=uuid4().hex,
            kind=MemoryKind.LONG_TERM,
            scope=scope,
            content=content,
            summary=summary or content[:160],
            metadata=metadata or {},
            embedding=vectors[0],
        )
        await self.store.insert(entry)
        return entry

    async def add_lesson(
        self, task_id: str, summary: str, detail: str, metadata: dict[str, object] | None = None
    ) -> MemoryEntry:
        meta: dict[str, object] = {"task_id": task_id, "kind": "lesson"}
        if metadata:
            meta.update(metadata)
        return await self.add(
            content=f"Lesson: {summary}\n{detail}", scope=task_id, summary=summary, metadata=meta
        )

    async def search(self, query: str, k: int = 5, scope: str | None = None) -> list[MemoryEntry]:
        entries = await self.store.list_kind(MemoryKind.LONG_TERM, limit=2000)
        if scope:
            entries = [e for e in entries if e.scope == scope]
        if not entries:
            return []
        qv = self.embedder.embed([query])[0]
        scored = [
            (
                self.embedder.cosine(
                    qv, e.embedding if e.embedding else self.embedder.embed([e.content])[0]
                ),
                e,
            )
            for e in entries
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [e for score, e in scored[:k] if score > 0.01]

    async def format_context(self, entries: list[MemoryEntry]) -> str:
        if not entries:
            return ""
        lines = []
        for e in entries:
            tag = e.metadata.get("kind", "fact")
            lines.append(f"- [{tag}] {e.summary or e.content[:200]}")
        return "\n".join(lines)
