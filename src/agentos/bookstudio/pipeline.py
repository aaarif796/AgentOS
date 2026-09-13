from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import structlog

from agentos.bookstudio.exporters import export_book
from agentos.bookstudio.schema import (
    BookManifest,
    BookSpec,
    ChapterSpec,
    ChapterStatus,
)
from agentos.gateway import ModelGateway
from agentos.memory.hub import MemoryHub
from agentos.models import ModelRequest, Task, VerificationReport
from agentos.reflexion import Verifier, _extract_json
from agentos.scraper import Scraper
from agentos.settings import Settings

log = structlog.get_logger()

MAX_CHAPTERS = 20
RESEARCH_QUERY_TEMPLATE = "https://html.duckduckgo.com/html/?q={query}"


class BookPipeline:
    """Supervised multi-agent book writing loop.

    plan (supervisor agent) -> per chapter: research -> write -> verify ->
    supervisor review -> export (docx/pdf/odf/md) -> final verification.
    """

    def __init__(
        self,
        gateway: ModelGateway,
        memory: MemoryHub,
        scraper: Scraper,
        settings: Settings,
        verifier: Verifier | None = None,
    ) -> None:
        self.gateway = gateway
        self.memory = memory
        self.scraper = scraper
        self.settings = settings
        self.verifier = verifier

    async def supervise(
        self,
        goal: str,
        resume_id: str | None = None,
        approve_all: bool = True,
    ) -> BookManifest:
        project_dir = self._project_dir(resume_id, goal)
        book, manifest = await self._load_or_create(project_dir, goal)
        out_dir = project_dir / "release"

        if not book.chapters:
            await self._plan(book, goal)
            await self._save(project_dir, book, manifest)

        for chapter in book.chapters:
            if chapter.status == ChapterStatus.APPROVED:
                continue
            await self._process_chapter(book, chapter, approve_all)
            await self._save(project_dir, book, manifest)

        artifacts = export_book(book, out_dir)
        manifest.artifacts.update(artifacts)

        if all(c.status == ChapterStatus.APPROVED for c in book.chapters):
            manifest.status = "completed"
        elif any(c.status == ChapterStatus.REJECTED for c in book.chapters):
            manifest.status = "needs_review"
        else:
            manifest.status = "awaiting_approval"

        manifest.sources = self._collect_sources(book)
        await self._save(project_dir, book, manifest)
        return manifest

    # -- pipeline steps -----------------------------------------------------
    async def _plan(self, book: BookSpec, goal: str) -> None:
        book.title = book.title or self._slug_to_title(goal)
        response, _ = self.gateway.complete_with_fallback(
            ModelRequest(
                model=self.settings.default_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are the book-supervisor agent. Reply with JSON only.",
                    },
                    {"role": "user", "content": _PLAN_PROMPT.format(goal=goal)},
                ],
                temperature=0.3,
                max_tokens=4000,
            )
        )
        payload = json.loads(_extract_json(response.text))
        book.title = str(payload.get("title", book.title))
        book.subtitle = str(payload.get("subtitle", ""))
        book.audience = str(payload.get("audience", "general readers"))
        book.style = str(payload.get("style", "professional"))
        book.front_matter = str(payload.get("front_matter", ""))
        book.back_matter = str(payload.get("back_matter", ""))
        chapters_spec = payload.get("chapters", [])
        if not isinstance(chapters_spec, list):
            chapters_spec = []
        for item in chapters_spec[:MAX_CHAPTERS]:
            if not isinstance(item, dict):
                continue
            book.chapters.append(
                ChapterSpec(
                    id=(
                        re.sub(r"[^a-z0-9]+", "-", str(item.get("title", "chapter")).lower()).strip(
                            "-"
                        )
                        or f"chapter-{len(book.chapters) + 1}"
                    ),
                    title=str(item.get("title", "")),
                    outline=str(item.get("outline", "")),
                    word_target=int(item.get("word_target", 1200)),
                )
            )

    async def _process_chapter(
        self, book: BookSpec, chapter: ChapterSpec, approve_all: bool
    ) -> None:
        research_notes = await self._research(chapter)
        for attempt in range(1, self.settings.book_max_chapter_attempts + 1):
            chapter.status = ChapterStatus.DRAFTED
            chapter.attempts = attempt
            chapter.draft = await self._write(book, chapter, research_notes, attempt)
            report = await self._verify(book, chapter)
            if report is None:
                chapter.status = ChapterStatus.APPROVED
                break
            if report.passed:
                chapter.status = ChapterStatus.APPROVED
                await self.memory.add_fact(
                    f"Chapter '{chapter.title}' of '{book.title}' verified with score {report.score:.2f}",
                    scope=book.id,
                    metadata={"chapter": chapter.id, "score": report.score},
                )
                await self.memory.add_recipe(
                    f"chapter-verified-{chapter.id[:10]}",
                    instructions=(
                        f"Verified approach for chapter topic '{chapter.title}': {chapter.outline}\n\n"
                        f"Research sources: {', '.join(chapter.sources)}"
                    ),
                    triggers=[chapter.title.lower(), book.title.lower()],
                    verified=True,
                )
                break
            chapter.feedback.append(self._feedback_line(report))
            if not approve_all and self.settings.book_require_approval:
                chapter.status = ChapterStatus.REJECTED
                break
        else:
            chapter.status = ChapterStatus.REJECTED

    async def _research(self, chapter: ChapterSpec) -> str:
        query_url = RESEARCH_QUERY_TEMPLATE.format(query=quote(chapter.title))
        try:
            result = await self.scraper.fetch(query_url, source=chapter.title)
        except Exception as exc:
            log.warning("book_research_failed", chapter=chapter.id, error=str(exc))
            return ""
        if result.text:
            chapter.sources.append(result.url)
            await self.memory.add_fact(
                result.text[:4000],
                scope=chapter.id,
                metadata={"kind": "research", "url": result.url, "title": result.title},
            )
        return result.text[:8000]

    async def _write(
        self, book: BookSpec, chapter: ChapterSpec, research_notes: str, attempt: int
    ) -> str:
        feedback_block = ""
        for fb in chapter.feedback[-1:]:
            feedback_block = f"\n\nPrevious verifier feedback to address:\n{fb}\n"
        prompt = _CHAPTER_PROMPT.format(
            book_title=book.title,
            book_subtitle=book.subtitle,
            audience=book.audience,
            style=book.style,
            chapter_title=chapter.title,
            chapter_outline=chapter.outline,
            word_target=chapter.word_target,
            research=research_notes[:6000],
            feedback=feedback_block,
        )
        response, _ = self.gateway.complete_with_fallback(
            ModelRequest(
                model=self.settings.default_model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are the book-writer agent writing '{book.title}'.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=4000,
            )
        )
        return response.text

    async def _verify(self, book: BookSpec, chapter: ChapterSpec) -> VerificationReport | None:
        if self.verifier is None:
            return None
        task = Task(
            goal=(
                f"Write chapter '{chapter.title}' of book '{book.title}' covering: {chapter.outline}. "
                f"Required sources: {', '.join(chapter.sources) or 'no external sources'}."
            )
        )
        return await self.verifier.verify(task, chapter.draft, model=self.settings.default_model)

    @staticmethod
    def _feedback_line(report: VerificationReport) -> str:
        score = report.score
        detail = "; ".join(i.detail for i in report.issues if i.severity == "blocker")
        return f"score={score:.2f}; blockers={detail or 'none listed'}"

    # -- persistence / helpers ------------------------------------------------
    @staticmethod
    def _slug_to_title(goal: str) -> str:
        words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9 ]*", goal) if len(w) > 2]
        title = " ".join(words[:8])
        return title if len(title) >= 4 else "Untitled Book"

    def _project_dir(self, resume_id: str | None, goal: str) -> Path:
        if resume_id:
            project_dir = Path(self.settings.workspace) / self.settings.book_output_dir / resume_id
            if not project_dir.exists():
                raise FileNotFoundError(f"no book project found to resume: {resume_id}")
            return project_dir
        slug = self._slug_to_title(goal).lower().replace(" ", "-")
        book_id = f"{slug}-{time.time_ns() % 100_000}"
        return Path(self.settings.workspace) / self.settings.book_output_dir / book_id

    async def _load_or_create(self, project_dir: Path, goal: str) -> tuple[BookSpec, BookManifest]:
        project_dir.mkdir(parents=True, exist_ok=True)
        book_file = project_dir / "book.json"
        manifest_file = project_dir / "manifest.json"
        if book_file.exists():
            book = BookSpec.model_validate_json(book_file.read_text(encoding="utf-8"))
        else:
            book = BookSpec(title=self._slug_to_title(goal))
            book.id = project_dir.name
        if manifest_file.exists():
            manifest = BookManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))
        else:
            manifest = BookManifest(id=project_dir.name, title=book.title)
        book.id = book.id or project_dir.name
        return book, manifest

    async def _save(self, project_dir: Path, book: BookSpec, manifest: BookManifest) -> None:
        book_file = project_dir / "book.json"
        manifest_file = project_dir / "manifest.json"
        book_file.write_text(book.model_dump_json(indent=2), encoding="utf-8")
        manifest_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _collect_sources(book: BookSpec) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for ch in book.chapters:
            for src in ch.sources:
                if src not in seen:
                    seen.add(src)
                    out.append(src)
        return out


async def load_book_manifest(book_id: str, workspace: Path, output_dir: Path) -> BookManifest:
    manifest_file = workspace / output_dir / book_id / "manifest.json"
    return BookManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))


_PLAN_PROMPT = """\
You are the book-supervisor agent. Architect a technical book for the topic:

{goal}

Return STRICT JSON exactly like this:
{{"title": "...", "subtitle": "...", "audience": "...", "style": "...",
  "front_matter": "short preface",
  "back_matter": "conclusion text",
  "chapters": [{{"title": "...", "outline": "what the chapter must cover", "word_target": 1500}}]}}

Create 5-8 well-sequenced chapters of real technical depth.
"""

_CHAPTER_PROMPT = """\
Book: {book_title} {book_subtitle}
Audience: {audience}
Style: {style}

Write the chapter titled: {chapter_title}
Outline to cover:
{chapter_outline}

Target length: about {word_target} words.

Research notes (use only as grounded factual material, do not invent facts):
{research}{feedback}

Return plain Markdown (headings ##, paragraphs, lists, and one or two code blocks where useful).
"""
