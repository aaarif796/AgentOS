from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from .models import Task


class TaskStore:
    def __init__(self, database_url: str) -> None:
        self.url = database_url
        self.path = (
            Path(database_url.rsplit("/", 1)[-1])
            if database_url.startswith("sqlite")
            else Path("agentos.db")
        )

    async def init(self) -> None:
        if self.url.startswith("sqlite"):
            async with aiosqlite.connect(self.path) as db:
                await db.execute(
                    "CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, data TEXT NOT NULL)"
                )
                await db.commit()

    async def save(self, task: Task) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO tasks(id,data) VALUES (?,?)",
                (task.id, json.dumps(task.model_dump(mode="json"))),
            )
            await db.commit()

    async def get(self, task_id: str) -> Task | None:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute("SELECT data FROM tasks WHERE id=?", (task_id,))
            row = await cur.fetchone()
        return Task.model_validate(json.loads(row[0])) if row else None
