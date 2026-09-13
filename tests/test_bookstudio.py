"""End-to-end offline tests for BookStudio: exporters + supervised chapter loop."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from agentos.bookstudio.exporters import ExportFormat, export_book, render_markdown
from agentos.bookstudio.pipeline import BookPipeline
from agentos.bookstudio.schema import BookSpec, ChapterSpec, ChapterStatus
from agentos.gateway import ModelGateway
from agentos.memory.hub import MemoryHub
from agentos.scraper import Scraper
from agentos.settings import Settings
from agentos.slides.schema import SlideLayout, SlideSpec

from .conftest import FAIL_JSON, PASS_JSON, ScriptedAdapter


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace=tmp_path,
        allow_network=False,
        log_level="ERROR",
        default_model="fake/main",
        database_url="sqlite+aiosqlite:///./agentos-book-test.db",
    )


def _sample_book() -> BookSpec:
    book = BookSpec(
        title="AgentOS in Action",
        subtitle="Building verified agents",
        audience="python developers",
        front_matter="Welcome to AgentOS.",
        back_matter="The end.",
        chapters=[
            ChapterSpec(
                id="chapter-1",
                title="The Memory Hub",
                outline="short, long and procedural memory",
                word_target=1200,
                draft=(
                    "## The Memory Hub\n"
                    "AgentOS stores knowledge in three layers.\n\n"
                    "- short-term\n- long-term\n- procedural\n\n"
                    "```python\nprint('hello memory')\n```\n"
                ),
                sources=["https://example.com/memory-hub"],
            )
        ],
    )
    book.chapters[0].status = ChapterStatus.APPROVED
    return book


# -- exporters ---------------------------------------------------------------


async def test_render_markdown_includes_toc_chapters_and_references() -> None:
    md = render_markdown(_sample_book())
    assert "# AgentOS in Action" in md
    assert "## Table of Contents" in md
    assert "1. The Memory Hub" in md
    assert "## Chapter 1: The Memory Hub" in md
    assert "## References" in md
    assert "https://example.com/memory-hub" in md


async def test_export_book_writes_all_four_valid_formats(tmp_path: Path) -> None:
    artifacts = export_book(_sample_book(), tmp_path / "release")
    assert set(artifacts) == {"md", "docx", "pdf", "odf"}
    for fmt, path_str in artifacts.items():
        assert Path(path_str).exists() and Path(path_str).stat().st_size > 0, fmt
    md = Path(artifacts["md"]).read_text(encoding="utf-8")
    assert "# AgentOS in Action" in md
    with open(artifacts["pdf"], "rb") as f:
        assert f.read(4) == b"%PDF"
    with zipfile.ZipFile(artifacts["docx"]) as zf:
        assert any(n.endswith(".xml") for n in zf.namelist())
    with zipfile.ZipFile(artifacts["odf"]) as zf:
        assert any(n.endswith(".xml") for n in zf.namelist())


async def test_export_book_respects_requested_formats(tmp_path: Path) -> None:
    artifacts = export_book(_sample_book(), tmp_path / "rel2", formats=[ExportFormat.MD])
    assert set(artifacts) == {"md"}


# -- supervised pipeline -------------------------------------------------------


def _plan_payload() -> str:
    return json.dumps(
        {
            "title": "AgentOS in Action",
            "subtitle": "verified agents",
            "audience": "python developers",
            "style": "professional",
            "front_matter": "preface",
            "back_matter": "conclusion",
            "chapters": [
                {"title": "The Memory Hub", "outline": "memory layers", "word_target": 900}
            ],
        }
    )


def _pipeline(
    tmp_path: Path, adapter: ScriptedAdapter, verifier: object = None
) -> tuple[BookPipeline, MemoryHub]:
    settings = _settings(tmp_path)
    hub = MemoryHub(tmp_path / "memory.db")
    pipeline = BookPipeline(
        ModelGateway(["fake/main"], adapter=adapter),
        hub,
        Scraper(settings),
        settings,
        verifier=verifier,  # type: ignore[arg-type]
    )
    return pipeline, hub


async def _init(hub: MemoryHub) -> None:
    await hub.init()


def _verify_verifier(tmp_path: Path, payload_sequence: list[str]) -> object:
    from agentos.reflexion import Verifier

    calls = {"n": 0}

    def respond(_user: str) -> str:
        payload = payload_sequence[min(calls["n"], len(payload_sequence) - 1)]
        calls["n"] += 1
        return payload

    gateway = ModelGateway(["fake/main"], adapter=ScriptedAdapter(verifier=respond))
    return Verifier(gateway, _settings(tmp_path))


async def test_book_pipeline_generates_all_formats_and_recipes(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        plans={"book-supervisor agent": _plan_payload()},
        drafts=["## The Memory Hub\nFull verified chapter body with real depth."],
    )
    verifier = _verify_verifier(tmp_path, [PASS_JSON])
    pipeline, hub = _pipeline(tmp_path, adapter, verifier)
    await _init(hub)
    manifest = await pipeline.supervise("write a book about agentos memory")
    assert manifest.status == "completed"
    assert set(manifest.artifacts) == {"md", "docx", "pdf", "odf"}
    for path_str in manifest.artifacts.values():
        assert Path(path_str).stat().st_size > 0
    project_manifest = json.loads(
        (Path(manifest.artifacts["md"]).parent.parent / "manifest.json").read_text(encoding="utf-8")
    )
    assert project_manifest["status"] == "completed"
    recipes = await hub.procedural.recall("the memory hub")
    assert any(r.metadata.get("verified") for r in recipes), "verified recipe must be stored"


async def test_book_pipeline_retries_failed_chapter_then_improves(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        plans={"book-supervisor agent": _plan_payload()},
        drafts=[
            "## The Memory Hub\nweak first draft",
            "## The Memory Hub\nimproved second draft addressing feedback",
        ],
    )
    verifier = _verify_verifier(tmp_path, [FAIL_JSON, PASS_JSON])
    pipeline, _hub = _pipeline(tmp_path, adapter, verifier)
    await _init(_hub)
    manifest = await pipeline.supervise("write a book about agentos memory")
    assert manifest.status == "completed", "retry after failed verification must recover"
    assert not adapter.pending(), "both scripted drafts must be consumed"


async def test_book_pipeline_persists_book_json_checkpoint(tmp_path: Path) -> None:
    adapter = ScriptedAdapter(
        plans={"book-supervisor agent": _plan_payload()},
        drafts=["## The Memory Hub\nchapter body"],
    )
    pipeline, _hub = _pipeline(tmp_path, adapter)
    await _init(_hub)
    manifest = await pipeline.supervise("write a book about agentos memory")
    book_file = Path(manifest.artifacts["md"]).parent.parent / "book.json"
    saved = json.loads(book_file.read_text(encoding="utf-8"))
    assert saved["title"] == "AgentOS in Action"
    assert saved["chapters"][0]["status"] == "approved"


async def test_diagram_slide_gets_native_mermaid_fallback(tmp_path: Path) -> None:
    from agentos.slides.mermaid import MermaidRenderer
    from agentos.slides.pipeline import _diagram_from_slide

    spec = SlideSpec(
        id="s1",
        layout=SlideLayout.DIAGRAM,
        title="Research Flow",
        bullets=["scrape", "verify"],
    )
    src = _diagram_from_slide(spec)
    assert src.startswith("flowchart LR")
    assert "n0[Research Flow]" in src
    assert "-->" in src
    # The native offline parser must turn that source into drawable shapes.
    renderer = MermaidRenderer(tmp_path / ".diagrams")
    nodes, edges = renderer.parse(src)
    assert len(nodes) >= 3
    assert len(edges) >= 2
