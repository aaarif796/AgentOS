"""Unit tests for the free/local-first model gateway: tiers, policies, failover."""

from __future__ import annotations

import time

import pytest

from agentos.gateway import ModelError, ModelGateway, RateLimiter
from agentos.models import ModelPolicy, ModelRequest, ModelTier

from .conftest import FailingAdapter, ScriptedAdapter


def _request(model: str = "ollama/llama3.1:8b") -> ModelRequest:
    return ModelRequest(model=model, messages=[{"role": "user", "content": "hello"}])


def test_tier_of_maps_providers() -> None:
    gw = ModelGateway(["ollama/llama3.1:8b", "groq/m", "openai/gpt-4o-mini"])
    assert gw.tier_of("ollama/llama3.1:8b") is ModelTier.LOCAL
    assert gw.tier_of("groq/m") is ModelTier.FREE
    assert gw.tier_of("openrouter/deepseek/deepseek-chat-v3-2:free") is ModelTier.FREE
    assert gw.tier_of("openai/gpt-4o-mini") is ModelTier.PAID
    assert gw.tier_of("anthropic/claude-3-5-haiku-latest") is ModelTier.PAID


def test_local_first_policy_orders_tiers() -> None:
    models = ["openai/gpt-4o-mini", "groq/llama-3.3-70b-versatile", "ollama/llama3.1:8b"]
    gw = ModelGateway(models, policy="local-first")
    ordered = gw._ordered_candidates("groq/llama-3.3-70b-versatile")
    assert ordered.index("ollama/llama3.1:8b") < ordered.index("groq/llama-3.3-70b-versatile")
    assert ordered.index("groq/llama-3.3-70b-versatile") < ordered.index("openai/gpt-4o-mini")


def test_paid_first_and_free_first_policies() -> None:
    models = ["ollama/l", "groq/f", "openai/p"]
    paid_first = ModelGateway(models, policy="paid-first")._ordered_candidates("ollama/l")
    free_first = ModelGateway(models, policy="free-first")._ordered_candidates("ollama/l")
    assert paid_first[0] == "openai/p"
    assert free_first[0] == "groq/f"


def test_policy_enum_values_are_stable() -> None:
    assert {p.value for p in ModelPolicy} == {
        "local-first",
        "free-first",
        "hybrid",
        "paid-first",
        "free-only",
    }


async def test_fallback_switches_model_when_one_fails() -> None:
    adapter = FailingAdapter(fail={"ollama/llama3.1:8b"})
    gw = ModelGateway(["ollama/llama3.1:8b", "groq/fallback"], adapter=adapter)
    response, errors = gw.complete_with_fallback(_request())
    assert response.model == "groq/fallback"
    assert any("ollama/llama3.1:8b" in e for e in errors)


async def test_paid_escalation_only_after_free_exhausted() -> None:
    adapter = FailingAdapter(fail={"groq/f", "ollama/l"})
    gw = ModelGateway(["ollama/l", "groq/f", "openai/paid"], policy="local-first", adapter=adapter)
    response, _ = gw.complete_with_fallback(_request("ollama/l"))
    assert response.model == "openai/paid"
    assert set(adapter.tried) == {"ollama/l", "groq/f", "openai/paid"}


async def test_circuit_breaker_cooldowns_broken_model() -> None:
    adapter = FailingAdapter(fail={"groq/broken"})
    gw = ModelGateway(["groq/broken"], adapter=adapter, max_failures=2, cooldown_seconds=60)
    with pytest.raises(ModelError):
        gw.complete_with_fallback(_request("groq/broken"))
    with pytest.raises(ModelError):
        gw.complete_with_fallback(_request("groq/broken"))
    tried_after_two = len(adapter.tried)
    with pytest.raises(ModelError) as exc:
        gw.complete_with_fallback(_request("groq/broken"))
    assert "cooldown" in str(exc.value)
    assert len(adapter.tried) == tried_after_two, "cooled model must not be retried"


async def test_identical_requests_are_cached() -> None:
    adapter = ScriptedAdapter(generic=["cached answer"])
    gw = ModelGateway(["fake/main"], adapter=adapter)
    r1, _ = gw.complete_with_fallback(_request("fake/main"))
    r2, _ = gw.complete_with_fallback(_request("fake/main"))
    assert r1.text == r2.text == "cached answer"
    assert len(adapter.attempts) == 1, "second identical call must be served from cache"


async def test_rate_limiter_waits_min_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = RateLimiter({"groq": 30.0})
    slept: list[float] = []
    monkeypatch.setattr("agentos.gateway.time.sleep", lambda s: slept.append(s))
    monkeypatch.setattr("agentos.gateway.time.monotonic", lambda: 100.0)
    limiter.wait("groq")
    limiter.wait("groq")
    assert slept, "second immediate call must be gated by the min interval"


async def test_health_report_reports_tiers_and_cooldowns() -> None:
    adapter = ScriptedAdapter(generic=["ok"])
    gw = ModelGateway(["ollama/l", "groq/f"], adapter=adapter)
    gw._cooldown_until["groq/f"] = time.monotonic() + 10_000.0
    report = {row["model"]: row for row in gw.health_report()}
    assert report["ollama/l"]["tier"] == "local"
    assert report["ollama/l"]["healthy"] is True
    assert report["groq/f"]["healthy"] is False
