"""Shared offline fixtures: scripted model adapter + settings helpers.

No test here ever touches the network: every model call is served by
ScriptedAdapter, and scraping is disabled / redirected to an invalid scheme.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agentos.gateway import ModelError
from agentos.models import ModelRequest, ModelResponse
from agentos.settings import Settings

PASS_JSON = (
    '{"passed": true, "score": 0.92, "issues": [], "suggestions": [], "checks": ["llm-review"]}'
)
FAIL_JSON = (
    '{"passed": false, "score": 0.15, "issues": [{"severity": "blocker", '
    '"area": "completeness", "detail": "output does not satisfy the goal"}], '
    '"suggestions": ["address the goal directly"], "checks": ["llm-review"]}'
)


class ScriptedAdapter:
    """Fake ProviderPool. Routes by system-prompt marker:

    - "strict quality verifier"  -> verifier callable / PASS JSON
    - plan markers (supervisors) -> plan JSON per marker
    - otherwise                  -> next scripted draft, then generic replies
    """

    provider = "fake"

    def __init__(
        self,
        verifier: Callable[[str], str] | None = None,
        plans: dict[str, str] | None = None,
        drafts: list[str] | None = None,
        generic: list[str] | None = None,
    ) -> None:
        self.verifier = verifier
        self.plans = plans or {}
        self.drafts = list(drafts or [])
        self.generic = list(generic or [])
        self.attempts: list[tuple[str, str]] = []

    def complete(self, model: str, request: ModelRequest) -> ModelResponse:
        system = request.messages[0]["content"]
        user = request.messages[-1]["content"]
        self.attempts.append((model, system[:60]))
        if "strict quality verifier" in system:
            text = self.verifier(user) if self.verifier else PASS_JSON
            return ModelResponse(text=text, model=model, provider=self.provider)
        for marker, plan in self.plans.items():
            if marker in system:
                return ModelResponse(text=plan, model=model, provider=self.provider)
        if self.drafts:
            return ModelResponse(text=self.drafts.pop(0), model=model, provider=self.provider)
        if self.generic:
            text = self.generic.pop(0) if len(self.generic) > 1 else self.generic[0]
            return ModelResponse(text=text, model=model, provider=self.provider)
        raise ModelError("no scripted response left", retryable=False)

    def pending(self) -> bool:
        """True when unconsumed scripted drafts/generic replies remain."""
        return bool(self.drafts or self.generic)

    def health_check(self, provider: str = "") -> bool:
        return True


class FailingAdapter:
    """Adapter that fails for selected models and records every attempt."""

    provider = "fake"

    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.tried: list[str] = []

    def complete(self, model: str, request: ModelRequest) -> ModelResponse:
        self.tried.append(model)
        if model in self.fail:
            raise ModelError(f"{model} down", retryable=False)
        return ModelResponse(text="ok", model=model, provider=self.provider)

    def health_check(self, provider: str = "") -> bool:
        return True


class FakeResp:
    """Minimal httpx.Response stand-in for monkeypatched transport."""

    def __init__(
        self,
        status_code: int = 200,
        payload: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self) -> dict[str, object]:
        return self._payload


@pytest.fixture
def base_settings(tmp_path: Path) -> Settings:
    return Settings(
        workspace=tmp_path,
        log_level="ERROR",
        allow_network=False,
        database_url="sqlite+aiosqlite:///./agentos-test.db",
    )
