from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def cosine(self, a: list[float], b: list[float]) -> float: ...


class TFIDFEmbedder:
    """Deterministic, offline, fixed-dimension hashed-ngram term vectors.

    Produces comparable vectors across calls without a shared vocabulary, so the
    long-term memory works even when no embedding API key is configured.
    """

    DIM = 384

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self._token_re = re.compile(r"[a-z0-9]+")
        self._ngram_re = re.compile(r"[a-z0-9]{3}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        low = text.lower()
        tokens = self._token_re.findall(low)
        for token in tokens:
            self._bump(vec, "w:" + token, 1.0)
        for ngram in self._ngram_re.findall(low):
            self._bump(vec, "g:" + ngram, 0.5)
        # L2 normalize for cosine comparability
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]

    def _bump(self, vec: list[float], token: str, weight: float) -> None:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign * weight


class OpenAIEmbedder:
    """Optional real embedding provider using the OpenAI API (via httpx)."""

    def __init__(self, model: str = "text-embedding-3-small", timeout: float = 30.0) -> None:
        import httpx

        self._httpx = httpx
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not configured for embedding")
        r = self._httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()["data"]
        ordered = [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
        return ordered

    def cosine(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
