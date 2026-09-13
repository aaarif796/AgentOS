"""Unit tests for the three-layer MemoryHub (short-term, long-term, procedural)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentos.memory.embedder import TFIDFEmbedder
from agentos.memory.hub import MemoryHub
from agentos.memory.models import MemoryKind
from agentos.memory.store import MemoryStore


@pytest.fixture
async def hub(tmp_path: Path) -> MemoryHub:
    mh = MemoryHub(tmp_path / "memory.db", max_short_term=3, top_k=3)
    await mh.init()
    return mh


def test_embedder_is_deterministic_and_normalized() -> None:
    emb = TFIDFEmbedder()
    a, b = emb.embed(["python programming language"] * 2)
    c = emb.embed(["cooking pasta recipes at home"])[0]
    assert len(a) == TFIDFEmbedder.DIM == 384
    assert a == b
    assert emb.cosine(a, a) == pytest.approx(1.0, abs=1e-6)
    assert emb.cosine(a, c) < 0.3


async def test_short_term_ring_buffer_keeps_latest(hub: MemoryHub) -> None:
    for i in range(5):
        await hub.remember_exchange("session-1", "user", f"message {i}")
    entries = await hub.short_term.recall("session-1")
    assert len(entries) == 3
    contents = [e.content for e in entries]
    assert contents == ["message 2", "message 3", "message 4"]


async def test_long_term_search_ranks_related_facts(hub: MemoryHub) -> None:
    await hub.add_fact("Python testing with pytest fixtures and parametrize", scope="kb")
    await hub.add_fact("Advanced pytest plugins for python test suites", scope="kb")
    await hub.add_fact("Rose gardening tips for soil and compost care", scope="kb")
    hits = await hub.long_term.search("python pytest testing", k=3)
    assert hits, "expected at least one related memory"
    texts = [h.content for h in hits]
    assert "gardening" not in texts[0].lower()
    assert all("pytest" in t.lower() or "python" in t.lower() for t in texts)


async def test_procedural_recall_prefers_verified_recipes(hub: MemoryHub) -> None:
    await hub.add_recipe(
        "parse-logs",
        instructions="Split the log file by level, then summarize errors.",
        triggers=["python"],
        verified=True,
    )
    await hub.add_recipe(
        "knitting",
        instructions="Cast on stitches with wool and needles.",
        triggers=["knitting"],
        verified=False,
    )
    hits = await hub.procedural.recall("write a python script to parse server logs")
    assert hits
    assert hits[0].scope == "parse-logs"
    assert hits[0].metadata["verified"] is True
    assert all(r.scope != "knitting" for r in hits)


async def test_build_context_includes_all_layers(hub: MemoryHub) -> None:
    goal = "how do I write pytest tests in python"
    await hub.add_fact("pytest fixtures isolate test state in python", scope="kb")
    await hub.add_recipe(
        "pytest-recipe", "Arrange, act, assert with fixtures.", triggers=["pytest"], verified=True
    )
    await hub.remember_exchange("session-9", "user", "earlier we discussed pytest")
    context = await hub.build_context(goal, "generalist", "session-9")
    assert "Relevant prior knowledge" in context
    assert "Proven recipes" in context
    assert "Working context" in context


async def test_capture_writes_recipe_on_success_and_lesson_on_failure(hub: MemoryHub) -> None:
    await hub.capture("task-abc", "write a haiku", "A short haiku about code.", verified=True)
    recipes = await hub.procedural.recall("write a haiku")
    assert any(
        r.scope.startswith("verified-pattern-") and r.metadata.get("verified") for r in recipes
    )

    await hub.capture(
        "task-abc", "write a sonnet", "partial draft", verified=False, feedback="missing sections"
    )
    lessons = await hub.store.list_kind(MemoryKind.LONG_TERM)
    assert any("Lesson:" in e.content and "missing sections" in e.content for e in lessons)


async def test_store_roundtrip_preserves_metadata(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "raw.db")
    await store.init()
    hub = MemoryHub(tmp_path / "raw.db")
    await hub.init()
    entry = await hub.add_fact(
        "fact body", scope="s1", metadata={"kind": "research", "url": "https://x"}
    )
    rows = await store.list_kind(MemoryKind.LONG_TERM)
    match = next(r for r in rows if r.id == entry.id)
    assert match.metadata["url"] == "https://x"
    assert match.content == "fact body"
