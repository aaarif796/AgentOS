"""End-to-end offline tests for SlidesStudio: theme, pptx build, supervised loop."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from agentos.gateway import ModelGateway
from agentos.memory.hub import MemoryHub
from agentos.settings import Settings
from agentos.slides.build import build_deck
from agentos.slides.mermaid import MermaidRenderer
from agentos.slides.pipeline import PresentationPipeline
from agentos.slides.schema import (
    DeckSpec,
    SectionSpec,
    SlideLayout,
    SlideSpec,
    ThemeSpec,
)
from agentos.slides.theme import THEMES, pick_theme

from .conftest import FAIL_JSON, PASS_JSON, ScriptedAdapter


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace=tmp_path,
        allow_network=False,
        log_level="ERROR",
        default_model="fake/main",
        database_url="sqlite+aiosqlite:///./agentos-slides-test.db",
    )


# -- theme engine --------------------------------------------------------------


async def test_pick_theme_is_deterministic_and_covers_topics() -> None:
    assert pick_theme("business pitch for investors") is THEMES["paper"]
    assert pick_theme("ocean health science") is THEMES["ocean"]
    assert pick_theme("creative design story") is THEMES["sunset"]
    assert pick_theme("nature environment") is THEMES["forest"]
    assert pick_theme("agentos architecture deep dive") is THEMES["midnight"]
    assert pick_theme("agentos architecture deep dive") is THEMES["midnight"]


async def test_deck_spec_theme_and_layout_validation() -> None:
    theme = ThemeSpec(name="midnight", background="#0e1117")
    deck = DeckSpec(
        title="AgentOS",
        theme=theme,
        sections=[
            SectionSpec(
                id="section-1",
                title="Memory",
                slides=[
                    SlideSpec(
                        id="s1", layout=SlideLayout.CONTENT, title="Layers", bullets=["a", "b"]
                    ),
                    SlideSpec(
                        id="s2", layout=SlideLayout.DIAGRAM, title="Flow", bullets=["x", "y"]
                    ),
                ],
            )
        ],
    )
    assert deck.sections[0].slides[0].layout == SlideLayout.CONTENT
    assert deck.theme.dark()


# -- pptx builder --------------------------------------------------------------


async def test_build_deck_produces_valid_pptx(tmp_path: Path) -> None:
    deck = DeckSpec(
        title="AgentOS Deep Dive",
        subtitle="verified decks",
        audience="engineers",
        theme=pick_theme("agentos"),
        sections=[
            SectionSpec(
                id="section-1",
                title="Memory",
                slides=[
                    SlideSpec(
                        id="s1",
                        layout=SlideLayout.CONTENT,
                        title="Three Layers",
                        bullets=["short-term", "long-term", "procedural"],
                        notes="speaker note",
                    ),
                    SlideSpec(
                        id="s2",
                        layout=SlideLayout.CODE,
                        title="Quickstart",
                        code="from agentos import build_runtime",
                    ),
                ],
            )
        ],
    )
    out = build_deck(deck, tmp_path, MermaidRenderer(tmp_path / ".diagrams"))
    assert out.exists() and out.stat().st_size > 0
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "[Content_Types].xml" in names
        assert "ppt/presentation.xml" in names
    # 1 cover + 1 agenda + 1 divider + 2 content + 1 closing = 6 slides
    with zipfile.ZipFile(out) as zf:
        slide_files = [n for n in zf.namelist() if n.startswith("ppt/slides/slide")]
    assert len(slide_files) == 6


# -- supervised pipeline -------------------------------------------------------


def _plan_payload() -> str:
    return json.dumps(
        {
            "title": "AgentOS Deep Dive",
            "subtitle": "verified decks",
            "audience": "engineers",
            "sections": [
                {
                    "title": "Memory",
                    "slides": [
                        {
                            "layout": "content",
                            "title": "Three Layers",
                            "bullets": ["short-term", "long-term", "procedural"],
                            "notes": "speaker note",
                        }
                    ],
                }
            ],
        }
    )


def _pipeline(
    tmp_path: Path, adapter: ScriptedAdapter, verifier: object = None
) -> tuple[PresentationPipeline, MemoryHub]:
    hub = MemoryHub(tmp_path / "memory.db")
    pipeline = PresentationPipeline(
        ModelGateway(["fake/main"], adapter=adapter),
        hub,
        _settings(tmp_path),
        verifier=verifier,  # type: ignore[arg-type]
    )
    return pipeline, hub


def _verify_verifier(tmp_path: Path, payload_sequence: list[str]) -> object:
    from agentos.reflexion import Verifier

    calls = {"n": 0}

    def respond(_user: str) -> str:
        payload = payload_sequence[min(calls["n"], len(payload_sequence) - 1)]
        calls["n"] += 1
        return payload

    gateway = ModelGateway(["fake/main"], adapter=ScriptedAdapter(verifier=respond))
    return Verifier(gateway, _settings(tmp_path))


async def test_slides_pipeline_generates_pptx_and_checkpoint(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        plans={"presentation-supervisor agent": _plan_payload()},
        generic=["section content"],  # fallback for any unscripted write step
    )
    verifier = _verify_verifier(tmp_path, [PASS_JSON])
    pipeline, hub = _pipeline(tmp_path, adapter, verifier)
    await hub.init()
    pptx = await pipeline.supervise("build a deck about agentos memory")
    assert pptx.exists() and pptx.stat().st_size > 0
    deck_file = pptx.parent / "deck.json"
    saved = json.loads(deck_file.read_text(encoding="utf-8"))
    assert saved["title"] == "AgentOS Deep Dive"
    assert saved["sections"][0]["status"] == "approved"


async def test_slides_pipeline_retries_failed_section_then_approves(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        plans={"presentation-supervisor agent": _plan_payload()},
        generic=["section content"],
    )
    verifier = _verify_verifier(tmp_path, [FAIL_JSON, PASS_JSON])
    pipeline, hub = _pipeline(tmp_path, adapter, verifier)
    await hub.init()
    pptx = await pipeline.supervise("build a deck about agentos memory")
    saved = json.loads((pptx.parent / "deck.json").read_text(encoding="utf-8"))
    section = saved["sections"][0]
    assert section["status"] == "approved", "retry after failed verification must recover"
    assert section["attempts"] == 2
    assert saved["title"] == "AgentOS Deep Dive"
