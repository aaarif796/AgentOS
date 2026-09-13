from __future__ import annotations

import re
from pathlib import Path

from agentos.memory.embedder import Embedder, TFIDFEmbedder
from agentos.memory.long_term import LongTermMemory
from agentos.memory.models import MemoryEntry
from agentos.memory.procedural import ProceduralMemory
from agentos.memory.short_term import ShortTermMemory
from agentos.memory.store import MemoryStore


class MemoryHub:
    """Facade over the three memory layers exposed to agents and the orchestrator."""

    def __init__(
        self,
        db_path: Path,
        max_short_term: int = 40,
        top_k: int = 5,
        embedder: Embedder | None = None,
    ) -> None:
        self.store = MemoryStore(db_path)
        self.embedder = embedder or TFIDFEmbedder()
        self.short_term = ShortTermMemory(self.store, max_entries=max_short_term)
        self.long_term = LongTermMemory(self.store, self.embedder)
        self.procedural = ProceduralMemory(self.store, top_k=top_k)
        self.hit_count = 0

    async def init(self) -> None:
        await self.store.init()

    async def build_context(self, goal: str, agent_id: str, scope: str) -> str:
        """Assemble relevant long-term knowledge + procedural recipes + working context."""
        sections: list[str] = []
        relevant = await self.long_term.search(goal, k=5)
        if relevant:
            sections.append(
                "## Relevant prior knowledge\n" + await self.long_term.format_context(relevant)
            )
            self.hit_count += 1
        recipes = await self.procedural.recall(goal)
        if recipes:
            sections.append("## Proven recipes\n" + await self.procedural.format_context(recipes))
        working = await self.short_term.format(scope, limit=5)
        if working:
            sections.append("## Working context\n" + working)
        return "\n\n".join(sections)

    async def remember_exchange(self, scope: str, role: str, content: str) -> MemoryEntry:
        return await self.short_term.remember(scope, role, content)

    async def learn(self, scope: str, summary: str, detail: str) -> MemoryEntry:
        return await self.long_term.add_lesson(scope, summary, detail)

    async def add_fact(
        self, content: str, scope: str = "global", metadata: dict[str, object] | None = None
    ) -> MemoryEntry:
        return await self.long_term.add(content, scope=scope, metadata=metadata)

    async def add_recipe(
        self,
        name: str,
        instructions: str,
        triggers: list[str],
        verified: bool = False,
    ) -> MemoryEntry:
        return await self.procedural.add_recipe(name, instructions, triggers, verified=verified)

    async def capture(
        self,
        scope: str,
        goal: str,
        output: str,
        verified: bool,
        feedback: str = "",
    ) -> None:
        """Post-run memory write: transcript → short-term, lesson → long-term,
        verified recipe → procedural."""
        await self.remember_exchange(scope, "user", goal)
        await self.remember_exchange(scope, "assistant", output[:2000])
        if not verified and feedback:
            await self.learn(scope, f"Attempt failed for: {goal}", feedback[:1000])
        elif verified:
            triggers = re.findall(r"[a-z0-9]{4,}", goal.lower())[:6]
            await self.add_recipe(
                f"verified-pattern-{scope[:8]}",
                instructions=f"Goal: {goal}\nVerified output approach (retry-safe):\n{output[:1500]}",
                triggers=triggers,
                verified=True,
            )
