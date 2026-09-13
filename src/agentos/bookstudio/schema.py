from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agentos.models import utcnow


class ChapterStatus(StrEnum):
    PLANNED = "planned"
    RESEARCHING = "researching"
    DRAFTED = "drafted"
    APPROVED = "approved"
    REJECTED = "rejected"


class ChapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    outline: str = ""
    word_target: int = 1200
    status: ChapterStatus = ChapterStatus.PLANNED
    sources: list[str] = Field(default_factory=list)
    draft: str = ""
    feedback: list[str] = Field(default_factory=list)
    attempts: int = 0


class BookSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""
    title: str
    subtitle: str = ""
    audience: str = "general readers"
    style: str = "professional"
    front_matter: str = ""
    back_matter: str = ""
    chapters: list[ChapterSpec] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def chapter(self, chapter_id: str) -> ChapterSpec | None:
        for ch in self.chapters:
            if ch.id == chapter_id:
                return ch
        return None


class BookManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    status: str = "planning"
    artifacts: dict[str, str] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
