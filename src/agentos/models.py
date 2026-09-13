from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class TaskStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_TOOL = "waiting_tool"
    FALLBACK = "fallback"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class DataClass(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class ModelTier(StrEnum):
    LOCAL = "local"
    FREE = "free"
    PAID = "paid"


class ModelPolicy(StrEnum):
    LOCAL_FIRST = "local-first"
    FREE_FIRST = "free-first"
    HYBRID = "hybrid"
    PAID_FIRST = "paid-first"
    FREE_ONLY = "free-only"


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: str = "1.0.0"
    mission: str
    capabilities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    model_capabilities: list[str] = Field(default_factory=list)
    permissions: dict[str, Any] = Field(default_factory=dict)
    evolution_enabled: bool = False


class SkillSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: str = "1.0.0"
    description: str
    tags: list[str] = Field(default_factory=list)
    instructions: str = ""


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    description: str
    risk: str = "low"
    capabilities: list[str] = Field(default_factory=list)


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    provider: str
    capabilities: list[str] = Field(default_factory=list)
    context_window: int = 128_000
    reliability: float = 0.95
    latency_ms: float = 1000
    enabled: bool = True
    tier: ModelTier = ModelTier.PAID
    brand: str = ""  # human-facing provider name (e.g. "groq", "ollama")
    base_url: str | None = None  # OpenAI-compatible base URL for free/local providers
    api_key_env: str | None = None  # env var holding the API key for this provider
    local: bool = False  # true for models served from the local Ollama runtime


class VerificationIssue(BaseModel):
    severity: str = "warning"  # blocker | warning | nit
    area: str = "general"  # facts|coding|consistency|completeness|style
    detail: str = ""


class VerificationReport(BaseModel):
    task_id: str = ""
    phase: str = "response"
    passed: bool = False
    score: float = 0.0
    issues: list[VerificationIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)
    model: str = ""


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: uuid4().hex)
    goal: str
    status: TaskStatus = TaskStatus.CREATED
    data_class: DataClass = DataClass.INTERNAL
    current_agent: str | None = None
    current_model: str | None = None
    step: int = 0
    messages: list[dict[str, str]] = Field(default_factory=list)
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class RouteNode(BaseModel):
    id: str
    agent_id: str
    skill_ids: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    model_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class ExecutionRoute(BaseModel):
    task_id: str
    strategy: str
    nodes: list[RouteNode]
    confidence: float
    rationale: list[str] = Field(default_factory=list)


class ModelRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.2
    max_tokens: int = 4096


class ModelResponse(BaseModel):
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0


class EvolutionCandidate(BaseModel):
    kind: str
    id: str
    reason: str
    definition: dict[str, Any]
    status: str = "candidate"
