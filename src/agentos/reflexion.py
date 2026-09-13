from __future__ import annotations

import json
import re
from dataclasses import dataclass

import structlog

from agentos.gateway import ModelError, ModelGateway
from agentos.models import (
    ModelRequest,
    Task,
    VerificationIssue,
    VerificationReport,
)
from agentos.settings import Settings
from agentos.tools import ToolRuntime

log = structlog.get_logger()

_FILE_CLAIM = re.compile(
    r"\b(?:created|wrote|writes?|written|generated|saved)[^\n]{0,60}:?\s*[`\"]?([\w./\\-]+\.(?:py|md|mdx|txt|json|yaml|yml|toml|js|ts|docx|pdf|odf|pptx))"
)


@dataclass(slots=True)
class Reflector:
    """Turns a failed VerificationReport into a better prompt for the next attempt."""

    max_feedback_chars: int = 2500

    def build(
        self, report: VerificationReport, goal: str, previous_output: str, attempt: int
    ) -> str:
        blockers = [i for i in report.issues if i.severity == "blocker"]
        warnings = [i for i in report.issues if i.severity == "warning"]
        lines = [
            f"Your previous attempt ({attempt}) did NOT pass verification "
            f"(score {report.score:.2f}). Improve the answer and try again.",
            f"Original goal: {goal}",
        ]
        if blockers:
            lines.append("Blocking problems to fix:")
            lines.extend(f"- ({i.area}) {i.detail}" for i in blockers)
        if warnings:
            lines.append("Warnings to address:")
            lines.extend(f"- ({i.area}) {i.detail}" for i in warnings)
        if report.suggestions:
            lines.append("Suggestions:")
            lines.extend(f"- {s}" for s in report.suggestions)
        if report.checks_run:
            lines.append("Checks performed:")
            lines.extend(f"- {c}" for c in report.checks_run)
        lines.append("Do NOT repeat the same mistakes. Produce a corrected, higher-quality answer.")
        return "\n".join(lines)[: self.max_feedback_chars]


class Verifier:
    """Two-layer verification gate: deterministic rules + LLM structured review."""

    def __init__(
        self,
        gateway: ModelGateway,
        settings: Settings,
        tools: ToolRuntime | None = None,
    ) -> None:
        self.gateway = gateway
        self.settings = settings
        self.tools = tools

    async def verify(
        self,
        task: Task,
        output: str,
        model: str = "",
        expected_files: list[str] | None = None,
    ) -> VerificationReport:
        checks: list[str] = []
        issues: list[VerificationIssue] = []

        # 1 -- deterministic checks (always run)
        if not output.strip():
            issues.append(
                VerificationIssue(
                    severity="blocker", area="completeness", detail="Agent returned empty output."
                )
            )
            checks.append("output not empty")
        else:
            checks.append("output not empty")

        if expected_files:
            missing = self._missing_files(expected_files)
            if missing:
                issues.append(
                    VerificationIssue(
                        severity="blocker",
                        area="artifacts",
                        detail=f"Claimed artifacts missing on disk: {', '.join(missing)}",
                    )
                )
                checks.append("claimed artifact files exist")
            else:
                checks.append("claimed artifact files exist")

        # 2 -- LLM structured review (layered; degrades gracefully)
        review_completed = False
        score = 0.5
        suggestions: list[str] = []
        if self.settings.verify_llm_review and output.strip():
            try:
                review = self._llm_review(task, output)
                raw_score = review.get("score", 0.5)
                if isinstance(raw_score, (int, float)):
                    score = float(raw_score)
                raw_issues = review.get("issues", [])
                if isinstance(raw_issues, list):
                    for item in raw_issues:
                        if not isinstance(item, dict):
                            continue
                        issues.append(
                            VerificationIssue(
                                severity=str(item.get("severity", "warning")),
                                area=str(item.get("area", "general")),
                                detail=str(item.get("detail", "")),
                            )
                        )
                raw_suggestions = review.get("suggestions", [])
                if isinstance(raw_suggestions, list):
                    suggestions = [str(s) for s in raw_suggestions]
                checks.append("llm structured review")
                review_completed = True
            except (ModelError, ValueError, json.JSONDecodeError) as exc:
                checks.append(f"llm review error: {exc}")

        if not review_completed:
            if self.settings.verify_on_error == "fail_closed":
                issues.append(
                    VerificationIssue(
                        severity="warning",
                        area="verification",
                        detail="LLM review unavailable; verification degraded to deterministic-only.",
                    )
                )
            score = max(0.0, 1.0 - 0.34 * len([i for i in issues if i.severity == "blocker"]))
            passed = all(i.severity != "blocker" for i in issues)
        else:
            score = max(0.0, min(1.0, score))
            passed = all(i.severity != "blocker" for i in issues) and score >= 0.5

        if not issues:
            score = max(score, 0.95)

        return VerificationReport(
            task_id=task.id,
            phase="response",
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
            checks_run=checks,
            model=model,
        )

    # -- helpers -----------------------------------------------------------
    def _missing_files(self, expected_files: list[str]) -> list[str]:
        if self.tools is None:
            return []
        missing: list[str] = []
        for rel in expected_files:
            path = self.settings.workspace / rel
            if not path.exists():
                missing.append(rel)
        return missing

    def _llm_review(self, task: Task, output: str) -> dict[str, object]:
        prompt = _VERIFIER_PROMPT.format(goal=task.goal, output=output)
        request = ModelRequest(
            model=self.settings.default_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict quality verifier. Reply with JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1600,
        )
        response, _errors = self.gateway.complete_with_fallback(request)
        parsed = json.loads(_extract_json(response.text))
        if not isinstance(parsed, dict):
            raise ValueError("verifier response was not a JSON object")
        return parsed


_VERIFIER_PROMPT = """\
Verify the following task output against the user goal before it is delivered.

User goal:
{goal}

Agent output:
{output}

Return STRICT JSON exactly like this:
{{"passed": true|false, "score": 0.0-1.0, "issues": [{{"severity": "blocker|warning|nit", "area": "facts|coding|consistency|completeness|style", "detail": "what is wrong"}}], "suggestions": ["actionable improvement"], "checks": ["what you checked"]}}

Rules: any hallucinated claim, missing required artifact, factual contradiction, or incomplete handling of the goal is a blocker. Be honest: if you cannot verify a claim, flag it.
"""


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in verifier response")
    return text[start : end + 1]


def extract_tool_calls(text: str) -> list[tuple[str, dict[str, object]]]:
    """Parse agent-emitted TOOL_CALL markers: `TOOL_CALL {"tool": ..., "args": {...}}`.

    Uses a balanced-brace scanner (string-aware) so nested JSON objects inside
    `args` are captured whole instead of stopping at the first `}`.
    """
    calls: list[tuple[str, dict[str, object]]] = []
    marker = "TOOL_CALL"
    idx = 0
    while True:
        idx = text.find(marker, idx)
        if idx < 0:
            break
        start = text.find("{", idx + len(marker))
        if start < 0:
            break
        depth = 0
        end = -1
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            break
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            tool = str(payload.get("tool", ""))
            args = payload.get("args", {})
            if tool and isinstance(args, dict):
                calls.append((tool, args))
        idx = end + 1
    return calls
