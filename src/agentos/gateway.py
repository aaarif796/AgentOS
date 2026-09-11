from __future__ import annotations

import os
from typing import Protocol

import httpx

from .models import ModelRequest, ModelResponse


class ModelError(Exception):
    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class ProviderAdapter(Protocol):
    provider: str

    def complete(self, model: str, request: ModelRequest) -> ModelResponse: ...


class HttpProviders:
    def complete(self, request: ModelRequest) -> ModelResponse:
        provider, model = request.model.split("/", 1)
        if provider == "openai":
            return self._openai(model, request)
        if provider == "anthropic":
            return self._anthropic(model, request)
        if provider == "gemini":
            return self._gemini(model, request)
        raise ModelError(f"unsupported provider: {provider}", retryable=False)

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
            provider="openai",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    def _anthropic(self, model: str, req: ModelRequest) -> ModelResponse:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ModelError("ANTHROPIC_API_KEY is not configured", False)
        system = "\n".join(m["content"] for m in req.messages if m["role"] == "system")
        messages = [m for m in req.messages if m["role"] != "system"]
        body = {
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
            provider="anthropic",
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
        return ModelResponse(text=text, model=req.model, provider="gemini")


class ModelGateway:
    def __init__(self, models: list[str], adapter: HttpProviders | None = None) -> None:
        self.models = models
        self.adapter = adapter or HttpProviders()

    def complete_with_fallback(self, request: ModelRequest) -> tuple[ModelResponse, list[str]]:
        errors: list[str] = []
        candidates = list(dict.fromkeys([request.model, *self.models]))
        for model in candidates:
            try:
                return self.adapter.complete(
                    model, request.model_copy(update={"model": model})
                ), errors
            except ModelError as exc:
                errors.append(f"{model}: {exc}")
                if not exc.retryable:
                    continue
        raise ModelError("All model candidates failed: " + " | ".join(errors), retryable=False)
