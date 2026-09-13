from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentos.models import utcnow


class MemoryKind(StrEnum):
    SHORT_TERM = "short-term"
    LONG_TERM = "long-term"
    PROCEDURAL = "procedural"


class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""
    kind: MemoryKind
    scope: str  # task id, chat session, skill id, or "global"
    content: str
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
