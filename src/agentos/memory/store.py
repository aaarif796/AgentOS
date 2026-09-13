from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite

from agentos.memory.models import MemoryEntry, MemoryKind

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  scope TEXT NOT NULL,
  content TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  metadata TEXT NOT NULL DEFAULT '{}',
  embedding TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
"""


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()

    async def insert(self, entry: MemoryEntry) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO memories(id,kind,scope,content,summary,metadata,embedding,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    entry.id,
                    entry.kind.value,
                    entry.scope,
                    entry.content,
                    entry.summary,
                    json.dumps(entry.metadata),
                    json.dumps(entry.embedding),
                    entry.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def delete(self, entry_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM memories WHERE id=?", (entry_id,))
            await db.commit()

    async def list_kind(
        self, kind: MemoryKind | None = None, limit: int = 500
    ) -> list[MemoryEntry]:
        async with aiosqlite.connect(self.db_path) as db:
            if kind is None:
                cur = await db.execute(
                    "SELECT id,kind,scope,content,summary,metadata,embedding,created_at "
                    "FROM memories ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            else:
                cur = await db.execute(
                    "SELECT id,kind,scope,content,summary,metadata,embedding,created_at "
                    "FROM memories WHERE kind=? ORDER BY created_at DESC LIMIT ?",
                    (kind.value, limit),
                )
            rows = await cur.fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def list_scope(self, kind: MemoryKind, scope: str, limit: int = 100) -> list[MemoryEntry]:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT id,kind,scope,content,summary,metadata,embedding,created_at "
                "FROM memories WHERE kind=? AND scope=? ORDER BY created_at ASC LIMIT ?",
                (kind.value, scope, limit),
            )
            rows = await cur.fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def all(self, kind: MemoryKind | None = None) -> list[MemoryEntry]:
        return await self.list_kind(kind)

    @staticmethod
    def _row_to_entry(row: Any) -> MemoryEntry:
        (
            _id,
            kind,
            scope,
            content,
            summary,
            metadata,
            embedding,
            created_at,
        ) = row
        return MemoryEntry(
            id=_id,
            kind=MemoryKind(kind),
            scope=scope,
            content=content,
            summary=summary,
            metadata=json.loads(metadata),
            embedding=json.loads(embedding),
            created_at=created_at,
        )
