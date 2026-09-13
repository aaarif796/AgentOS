from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .models import ModelPolicy, ModelRequest, ModelResponse, ModelTier

OPENAI = "openai"
ANTHROPIC = "anthropic"
GEMINI = "gemini"
OLLAMA = "ollama"


class ModelError(Exception):
    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class CompatEndpoint:
    name: str
    base_url: str
    api_key_env: str | None


# OpenAI-compatible free providers. Any base_url + Bearer key works, so adding
# another provider from the awesome-free-llm-apis catalog is a one-line entry.
_COMPAT_ENDPOINTS: dict[str, CompatEndpoint] = {
    "groq": CompatEndpoint("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": CompatEndpoint(
        "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
    ),
    "nvidia": CompatEndpoint("nvidia", "https://integrate.api.nvidia.com/v1", "NVIDIA_NIM_API_KEY"),
    "deepseek": CompatEndpoint("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "mistral": CompatEndpoint("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "together": CompatEndpoint("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "cerebras": CompatEndpoint("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
}


class ProviderAdapter(Protocol):
    def complete(self, model: str, request: ModelRequest) -> ModelResponse: ...

    def health_check(self, provider: str = "") -> bool: ...


class OpenAICompatProvider:
    """Generic adapter for the many free / OpenAI-compatible chat endpoints."""

    def __init__(self, name: str, base_url: str, api_key_env: str | None = None) -> None:
        self.provider = name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env

    def complete(self, model: str, request: ModelRequest) -> ModelResponse:
        key = os.getenv(self.api_key_env) if self.api_key_env else None
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://agentos.local"
            headers["X-Title"] = "AgentOS"
        body: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        r = httpx.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=120)
        return self._parse(r, model, request)

    def health_check(self) -> bool:
        try:
            headers: dict[str, str] = {}
            key = os.getenv(self.api_key_env) if self.api_key_env else None
            if key:
                headers["Authorization"] = f"Bearer {key}"
            r = httpx.get(f"{self.base_url}/models", headers=headers, timeout=10)
            return r.status_code < 500
        except Exception:
            return False

    def _parse(self, r: httpx.Response, model: str, request: ModelRequest) -> ModelResponse:
        if r.status_code in (401, 403, 404):
            raise ModelError(f"{self.provider} HTTP {r.status_code}", False)
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After") or r.headers.get("retry-after")
            extra = f"; retry_after={retry_after}" if retry_after else ""
            raise ModelError(f"{self.provider} rate limited (429){extra}", True)
        if r.status_code >= 400:
            raise ModelError(
                f"{self.provider} HTTP {r.status_code}",
                r.status_code in (408, 409, 429, 500, 502, 503, 504),
            )
        data = r.json()
        usage = data.get("usage", {})
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            text = ""
        return ModelResponse(
            text=text,
            model=request.model,
            provider=self.provider,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


class OllamaProvider(OpenAICompatProvider):
    """Local model runtime at http://127.0.0.1:11434 - zero keys, zero limits."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        super().__init__(OLLAMA, f"{base_url.rstrip('/')}/v1", None)
        self.root_url = base_url.rstrip("/")

    def health_check(self) -> bool:
        try:
            r = httpx.get(f"{self.root_url}/api/tags", timeout=10)
            return r.status_code == 200
        except Exception:
            return False


class ProviderPool:
    """Routes `provider/model` ids to the correct adapter."""

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434") -> None:
        self.ollama = OllamaProvider(ollama_url)
        self._compat: dict[str, OpenAICompatProvider] = {
            name: OpenAICompatProvider(name, ep.base_url, ep.api_key_env)
            for name, ep in _COMPAT_ENDPOINTS.items()
        }

    def complete(self, model: str, request: ModelRequest) -> ModelResponse:
        provider, _, rest = model.partition("/")
        if provider == OLLAMA:
            return self.ollama.complete(rest, request)
        if provider in self._compat:
            return self._compat[provider].complete(rest, request)
        if provider == OPENAI:
            return self._openai(rest, request)
        if provider == ANTHROPIC:
            return self._anthropic(rest, request)
        if provider == GEMINI:
            return self._gemini(rest, request)
        raise ModelError(f"unsupported provider: {provider}", retryable=False)

    def health_check(self, provider: str) -> bool:
        if provider == OLLAMA:
            return self.ollama.health_check()
        if provider in self._compat:
            return self._compat[provider].health_check()
        return True

    def _openai(self, model: str, req: ModelRequest) -> ModelResponse:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ModelError("OPENAI_API_KEY is not configured", False)
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": req.messages,
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
            },
            timeout=120,
        )
        if r.status_code in (401, 403, 404):
            raise ModelError(f"OpenAI HTTP {r.status_code}", False)
        if r.status_code >= 400:
            raise ModelError(
                f"OpenAI HTTP {r.status_code}", r.status_code in (408, 409, 429, 500, 502, 503, 504)
            )
        data = r.json()
        usage = data.get("usage", {})
        return ModelResponse(
            text=data["choices"][0]["message"]["content"],
            model=req.model,
            provider=OPENAI,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    def _anthropic(self, model: str, req: ModelRequest) -> ModelResponse:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ModelError("ANTHROPIC_API_KEY is not configured", False)
        system = "\n".join(m["content"] for m in req.messages if m["role"] == "system")
        messages = [m for m in req.messages if m["role"] != "system"]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if system:
            body["system"] = system
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json=body,
            timeout=120,
        )
        if r.status_code in (401, 403, 404):
            raise ModelError(f"Anthropic HTTP {r.status_code}", False)
        if r.status_code >= 400:
            raise ModelError(
                f"Anthropic HTTP {r.status_code}",
                r.status_code in (408, 409, 429, 500, 502, 503, 504),
            )
        data = r.json()
        text = "".join(part.get("text", "") for part in data.get("content", []))
        usage = data.get("usage", {})
        return ModelResponse(
            text=text,
            model=req.model,
            provider=ANTHROPIC,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )

    def _gemini(self, model: str, req: ModelRequest) -> ModelResponse:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ModelError("GEMINI_API_KEY/GOOGLE_API_KEY is not configured", False)
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in req.messages
            if m["role"] != "system"
        ]
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": key},
            json={"contents": contents},
            timeout=120,
        )
        if r.status_code in (400, 401, 403, 404):
            raise ModelError(f"Gemini HTTP {r.status_code}", False)
        if r.status_code >= 400:
            raise ModelError(
                f"Gemini HTTP {r.status_code}", r.status_code in (408, 409, 429, 500, 502, 503, 504)
            )
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return ModelResponse(text=text, model=req.model, provider=GEMINI)


class RateLimiter:
    """Min-interval gate per provider. Enforced only when intervals are supplied."""

    def __init__(self, intervals_s: dict[str, float] | None = None) -> None:
        self._intervals = intervals_s or {}
        self._last: dict[str, float] = {}

    def wait(self, provider: str) -> None:
        interval = self._intervals.get(provider, 0.0)
        if interval <= 0:
            return
        now = time.monotonic()
        last = self._last.get(provider, 0.0)
        remaining = last + interval - now
        if remaining > 0:
            time.sleep(remaining)
        self._last[provider] = time.monotonic()


class ModelGateway:
    def __init__(
        self,
        models: list[str],
        policy: str = "local-first",
        adapter: ProviderAdapter | None = None,
        cooldown_seconds: int = 60,
        max_failures: int = 3,
        rate_intervals: dict[str, float] | None = None,
    ) -> None:
        self.models = models
        self.policy = ModelPolicy(policy)
        self.adapter = adapter or ProviderPool()
        self.cooldown_seconds = cooldown_seconds
        self.max_failures = max_failures
        self.ratelimiter = RateLimiter(rate_intervals)
        self._failures: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}
        self._cache: dict[str, ModelResponse] = {}

    def complete_with_fallback(
        self, request: ModelRequest, candidates: list[str] | None = None
    ) -> tuple[ModelResponse, list[str]]:
        errors: list[str] = []
        for model in (
            candidates if candidates is not None else self._ordered_candidates(request.model)
        ):
            if self._in_cooldown(model):
                errors.append(f"{model}: in cooldown")
                continue
            cached = self._cache_get(request, model)
            if cached is not None:
                return cached, errors
            try:
                provider = model.partition("/")[0]
                self.ratelimiter.wait(provider)
                response = self.adapter.complete(model, request.model_copy(update={"model": model}))
                self._mark_success(model)
                self._cache_put(request, model, response)
                return response, errors
            except ModelError as exc:
                errors.append(f"{model}: {exc}")
                self._mark_failure(model)
                if not exc.retryable:
                    continue
        raise ModelError("All model candidates failed: " + " | ".join(errors), retryable=False)

    def health_report(self) -> list[dict[str, Any]]:
        """Live health snapshot for `agentos models --health` and the API."""
        out: list[dict[str, Any]] = []
        for model in self.models:
            provider = model.partition("/")[0]
            out.append(
                {
                    "model": model,
                    "tier": self.tier_of(model).value,
                    "healthy": not self._in_cooldown(model),
                    "consecutive_failures": self._failures.get(model, 0),
                    "probe": True,
                    "online": self.adapter.health_check(provider),
                }
            )
        return out

    def tier_of(self, model: str) -> ModelTier:
        provider = model.partition("/")[0]
        if provider == OLLAMA:
            return ModelTier.LOCAL
        if provider in _COMPAT_ENDPOINTS:
            return ModelTier.FREE
        return ModelTier.PAID

    def _ordered_candidates(self, preferred: str) -> list[str]:
        pool = list(dict.fromkeys([preferred, *self.models]))
        tiers = {m: self.tier_of(m) for m in pool}
        if self.policy == ModelPolicy.FREE_ONLY:
            order = {ModelTier.FREE: 0, ModelTier.LOCAL: 1, ModelTier.PAID: 9}
        elif self.policy == ModelPolicy.PAID_FIRST:
            order = {ModelTier.PAID: 0, ModelTier.FREE: 1, ModelTier.LOCAL: 2}
        elif self.policy == ModelPolicy.FREE_FIRST:
            order = {ModelTier.FREE: 0, ModelTier.LOCAL: 1, ModelTier.PAID: 2}
        else:  # local-first
            order = {ModelTier.LOCAL: 0, ModelTier.FREE: 1, ModelTier.PAID: 2}

        ordered = sorted(pool, key=lambda m: (order[tiers[m]], pool.index(m)))
        if self.policy == ModelPolicy.FREE_ONLY:
            free_only = [m for m in ordered if tiers[m] != ModelTier.PAID]
            if not free_only:
                raise ModelError(
                    "free-only policy: no free/local models available", retryable=False
                )
            return free_only
        return ordered

    def _in_cooldown(self, model: str) -> bool:
        until = self._cooldown_until.get(model, 0.0)
        return until > time.monotonic()

    def _mark_success(self, model: str) -> None:
        self._failures.pop(model, None)
        self._cooldown_until.pop(model, None)

    def _mark_failure(self, model: str) -> None:
        count = self._failures.get(model, 0) + 1
        self._failures[model] = count
        if count >= self.max_failures:
            self._cooldown_until[model] = time.monotonic() + self.cooldown_seconds
            self._failures[model] = 0

    def _cache_key(self, request: ModelRequest, model: str) -> str:
        payload = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _cache_get(self, request: ModelRequest, model: str) -> ModelResponse | None:
        return self._cache.get(self._cache_key(request, model))

    def _cache_put(self, request: ModelRequest, model: str, response: ModelResponse) -> None:
        key = self._cache_key(request, model)
        if len(self._cache) < 512:
            self._cache[key] = response
