"""Unit tests for the universal Verify-Before-Respond gate (Verifier + Reflector)."""

from __future__ import annotations

from pathlib import Path

from agentos.gateway import ModelError, ModelGateway
from agentos.models import ModelRequest, ModelResponse, Task
from agentos.reflexion import Reflector, Verifier, extract_tool_calls
from agentos.security import SecurityEngine
from agentos.settings import Settings
from agentos.tools import ToolRuntime

from .conftest import FAIL_JSON, ScriptedAdapter


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace=tmp_path,
        allow_network=False,
        log_level="ERROR",
        default_model="fake/main",
        database_url="sqlite+aiosqlite:///./agentos-reflexion-test.db",
    )


def _gateway(adapter: ScriptedAdapter) -> ModelGateway:
    return ModelGateway(["fake/main"], adapter=adapter)


async def test_verifier_passes_good_output(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    verifier = Verifier(_gateway(ScriptedAdapter()), settings)
    report = await verifier.verify(Task(goal="write a haiku"), "A crisp haiku about code.")
    assert report.passed is True
    assert report.score >= 0.5
    assert "output not empty" in report.checks_run


async def test_verifier_blocks_empty_output(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    verifier = Verifier(_gateway(ScriptedAdapter()), settings)
    report = await verifier.verify(Task(goal="do something"), "   ")
    assert report.passed is False
    assert any(i.severity == "blocker" and i.area == "completeness" for i in report.issues)


async def test_verifier_llm_review_flags_blocker(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    adapter = ScriptedAdapter(verifier=lambda _user: FAIL_JSON)
    verifier = Verifier(_gateway(adapter), settings)
    report = await verifier.verify(Task(goal="write a haiku"), "irrelevant text")
    assert report.passed is False
    assert report.score < 0.5
    assert any(i.severity == "blocker" for i in report.issues)


async def test_verifier_degrades_gracefully_when_llm_fails(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class DownAdapter(ScriptedAdapter):
        def complete(self, model: str, request: ModelRequest) -> ModelResponse:
            raise ModelError("verifier model down", retryable=False)

    verifier = Verifier(_gateway(DownAdapter()), settings)
    report = await verifier.verify(Task(goal="write a haiku"), "A crisp haiku about code.")
    assert report.passed is True, "deterministic checks should still pass on good output"
    assert any("degraded" in i.detail for i in report.issues), "degraded status must be visible"


async def test_verifier_checks_claimed_artifacts_on_disk(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    tools = ToolRuntime(tmp_path, SecurityEngine(tmp_path))
    verifier = Verifier(_gateway(ScriptedAdapter()), settings, tools=tools)
    report = await verifier.verify(
        Task(goal="create a report file"),
        "Created file: report.md done.",
        expected_files=["report.md"],
    )
    assert report.passed is False
    assert any("missing on disk" in i.detail for i in report.issues)

    (tmp_path / "report.md").write_text("# report", encoding="utf-8")
    report_ok = await verifier.verify(
        Task(goal="create a report file"),
        "Created file: report.md done.",
        expected_files=["report.md"],
    )
    assert report_ok.passed is True


def test_reflector_builds_improved_prompt_with_blockers() -> None:
    from agentos.models import VerificationIssue, VerificationReport

    report = VerificationReport(
        passed=False,
        score=0.2,
        issues=[
            VerificationIssue(severity="blocker", area="completeness", detail="goal not covered"),
            VerificationIssue(severity="warning", area="style", detail="too terse"),
        ],
        suggestions=["cover the full goal"],
        checks_run=["output not empty"],
    )
    feedback = Reflector().build(report, "cover the entire goal", "partial", attempt=2)
    assert "did NOT pass verification" in feedback
    assert "cover the entire goal" in feedback
    assert "(completeness) goal not covered" in feedback
    assert "cover the full goal" in feedback
    assert "Do NOT repeat" in feedback


def test_extract_tool_calls_parses_markers() -> None:
    text = (
        'TOOL_CALL {"tool": "filesystem", "args": {"path": "a.md", "action": "write", "content": "x"}}\n'
        "some observation\n"
        'TOOL_CALL {"tool": "git", "args": {"command": "status"}}'
    )
    calls = extract_tool_calls(text)
    assert [c[0] for c in calls] == ["filesystem", "git"]
    assert calls[0][1]["path"] == "a.md"
