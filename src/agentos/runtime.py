from pathlib import Path

import httpx

from .bookstudio.pipeline import BookPipeline
from .builtin_hooks import AuditHook, ErrorLogHook
from .events import EventBus
from .evolution import EvolutionEngine
from .gateway import ModelGateway
from .hooks import HookManager
from .logging import configure_logging
from .memory.hub import MemoryHub
from .models import AgentSpec, ModelSpec, ModelTier, SkillSpec
from .orchestrator import Orchestrator
from .reflexion import Verifier
from .registry import Registry
from .router import MasterRouter
from .rules import default_rules
from .scraper import Scraper
from .security import SecurityEngine
from .settings import Settings
from .slides.pipeline import PresentationPipeline
from .store import TaskStore
from .tools import ToolRuntime

_AGENT_DATA = [
    (
        "orchestrator",
        "Coordinate multi-agent execution",
        ["orchestration"],
        ["planning", "workflow"],
        ["filesystem", "git"],
    ),
    (
        "planner",
        "Break goals into executable plans",
        ["planning"],
        ["system-design"],
        ["filesystem"],
    ),
    (
        "software-architect",
        "Design robust software",
        ["architecture"],
        ["system-design", "api-design", "database-design"],
        ["filesystem", "git"],
    ),
    (
        "developer",
        "Implement software",
        ["coding", "backend", "frontend"],
        ["python", "javascript", "typescript", "clean-code", "git", "testing"],
        ["filesystem", "terminal", "git"],
    ),
    (
        "researcher",
        "Research and synthesize evidence",
        ["research", "scraping", "web"],
        ["web-research", "web-scraping", "source-evaluation", "summarization"],
        ["http", "filesystem"],
    ),
    (
        "code-reviewer",
        "Review code quality and correctness",
        ["review"],
        ["clean-code", "testing", "owasp"],
        ["filesystem", "git"],
    ),
    (
        "tester",
        "Design and execute tests",
        ["testing"],
        ["unit-testing", "integration-testing", "tdd"],
        ["filesystem", "terminal", "git"],
    ),
    (
        "security-auditor",
        "Audit security",
        ["security"],
        ["owasp", "secrets-management", "dependency-security", "authentication"],
        ["filesystem", "terminal", "git"],
    ),
    (
        "documentation-writer",
        "Write technical documentation",
        ["documentation", "writing"],
        ["technical-documentation", "summarization"],
        ["filesystem", "git"],
    ),
    (
        "devops-engineer",
        "Build delivery infrastructure",
        ["devops"],
        ["docker", "ci-cd", "linux", "git"],
        ["filesystem", "terminal", "git", "http"],
    ),
    (
        "book-supervisor",
        "Supervise full book production: plan, dispatch chapters, review and compile",
        ["book", "books", "writing", "supervision"],
        ["book-writing", "information-architecture", "copywriting"],
        ["filesystem", "http"],
    ),
    (
        "web-researcher",
        "Scrape the web, extract evidence and cite sources for books and decks",
        ["scraping", "web", "research"],
        ["web-scraping", "web-research", "source-evaluation"],
        ["http", "filesystem"],
    ),
    (
        "book-designer",
        "Design book structure, chapter ordering and narrative arc",
        ["book", "design"],
        ["information-architecture", "copywriting"],
        ["filesystem"],
    ),
    (
        "book-writer",
        "Write high-quality book chapters grounded in research",
        ["writing", "book"],
        ["book-writing", "copywriting", "technical-documentation"],
        ["filesystem"],
    ),
    (
        "book-verifier",
        "Fact-check chapters against sources, check consistency and completeness",
        ["verify", "book"],
        ["source-evaluation", "summarization"],
        ["filesystem"],
    ),
    (
        "book-generator",
        "Compile and export books to markdown, docx, pdf and odf",
        ["generate", "book"],
        ["book-writing"],
        ["filesystem"],
    ),
]
_SLIDE_AGENTS = [
    (
        "presentation-supervisor",
        "Supervise premium slide-deck production: plan sections, review and compile pptx",
        ["presentation", "slide", "ppt", "deck", "supervision"],
        ["slide-design", "mermaid-diagrams", "information-architecture"],
        ["filesystem"],
    ),
    (
        "presentation-designer",
        "Design slide narrative, themes and visual hierarchy for decks",
        ["presentation", "design"],
        ["slide-design", "information-architecture"],
        ["filesystem"],
    ),
    (
        "slide-writer",
        "Write crisp slide content: bullets, quotes, data and speaker notes",
        ["presentation", "writing"],
        ["slide-design", "copywriting"],
        ["filesystem"],
    ),
    (
        "diagram-designer",
        "Author mermaid diagrams and data visuals for slides",
        ["diagram", "visual"],
        ["mermaid-diagrams", "data-visualization"],
        ["filesystem"],
    ),
    (
        "ppt-verifier",
        "Verify slide decks for design consistency, completeness and facts",
        ["verify", "presentation"],
        ["slide-design", "source-evaluation"],
        ["filesystem"],
    ),
    (
        "ppt-generator",
        "Build the final .pptx using the theme and diagram engines",
        ["generate", "presentation"],
        ["slide-design"],
        ["filesystem"],
    ),
]

_SKILL_DATA = [
    ("python", "Python engineering patterns", ["programming"]),
    ("javascript", "JavaScript engineering patterns", ["programming"]),
    ("typescript", "TypeScript engineering patterns", ["programming"]),
    ("git", "Safe Git workflows", ["devops"]),
    ("clean-code", "Maintainable software design", ["engineering"]),
    ("system-design", "Architecture and tradeoffs", ["architecture"]),
    ("api-design", "HTTP API design", ["backend"]),
    ("database-design", "Schema and indexing", ["database"]),
    ("unit-testing", "Unit testing practices", ["testing"]),
    ("integration-testing", "Integration testing", ["testing"]),
    ("tdd", "Test-driven development", ["testing"]),
    ("owasp", "Application security assessment", ["security"]),
    ("secrets-management", "Secret handling", ["security"]),
    ("dependency-security", "Dependency risk assessment", ["security"]),
    ("authentication", "Authentication and authorization", ["security"]),
    ("docker", "Container engineering", ["devops"]),
    ("ci-cd", "Continuous delivery", ["devops"]),
    ("linux", "Linux operations", ["system"]),
    ("web-research", "Evidence-based research", ["research"]),
    ("web-scraping", "Safe web scraping with citations", ["research"]),
    ("source-evaluation", "Source quality evaluation", ["research"]),
    ("summarization", "Accurate summarization", ["writing"]),
    ("technical-documentation", "Technical docs", ["writing"]),
    ("book-writing", "Structured book chapter writing", ["writing", "book"]),
    ("information-architecture", "Outline and structure design", ["design", "writing"]),
    ("copywriting", "Persuasive and clear writing", ["writing"]),
    ("slide-design", "Premium slide layout and narrative design", ["design", "presentation"]),
    (
        "mermaid-diagrams",
        "Author mermaid diagrams (flowchart, sequence, gantt)",
        ["diagram", "visual"],
    ),
    ("data-visualization", "Tables and data presentation", ["diagram", "visual"]),
]


def discover_ollama_models(
    ollama_url: str = "http://127.0.0.1:11434", timeout: float = 3.0
) -> list[str]:
    """Probe the local Ollama runtime and auto-register its installed models."""
    try:
        r = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=timeout)
        if r.status_code != 200:
            return []
        return [f"ollama/{m['name']}" for m in r.json().get("models", [])]
    except Exception:
        return []


def build_registry() -> Registry:
    r = Registry()
    for id_, mission, caps, skills, tools in [*_AGENT_DATA, *_SLIDE_AGENTS]:
        r.add_agent(
            AgentSpec(
                id=id_,
                mission=mission,
                capabilities=caps,
                skills=skills,
                tools=tools,
                permissions={t: {} for t in tools},
            )
        )
    for id_, desc, tags in _SKILL_DATA:
        r.add_skill(SkillSpec(id=id_, description=desc, tags=tags, instructions=desc))
    return r


def build_gateway(settings: Settings) -> ModelGateway:
    discovered = discover_ollama_models(settings.ollama_url)
    local_models = list(dict.fromkeys([*discovered, *settings.local_models]))
    models = [
        *local_models,
        *settings.free_models,
        settings.default_model,
        *settings.fallback_models,
    ]
    return ModelGateway(
        models,
        policy=settings.model_policy,
        cooldown_seconds=settings.model_cooldown_seconds,
        max_failures=settings.model_max_consecutive_failures,
    )


_FREE_PROVIDERS = {
    "groq",
    "openrouter",
    "nvidia",
    "deepseek",
    "mistral",
    "together",
    "cerebras",
}


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configure_logging(settings.log_level)
        self.registry = build_registry()
        self.events = EventBus()
        self.hooks = HookManager(self.events)
        self.hooks.register(AuditHook(settings.workspace / ".agentos" / "audit.jsonl"))
        self.hooks.register(ErrorLogHook())
        self.rules = default_rules()
        self.store = TaskStore(settings.database_url)
        self.security = SecurityEngine(Path(settings.workspace))
        self.tools = ToolRuntime(Path(settings.workspace), self.security, settings.terminal_timeout)
        self.gateway = build_gateway(settings)
        self.memory = (
            MemoryHub(
                settings.database_path,
                max_short_term=settings.memory_max_short_term,
                top_k=settings.memory_top_k,
            )
            if settings.memory_enabled
            else None
        )
        self.scraper = Scraper(settings)
        self.verifier: Verifier | None = Verifier(self.gateway, settings, tools=self.tools)
        self.book_pipeline = (
            BookPipeline(self.gateway, self.memory, self.scraper, settings, verifier=self.verifier)
            if self.memory
            else None
        )
        self.slides_pipeline = (
            PresentationPipeline(self.gateway, self.memory, settings, verifier=self.verifier)
            if self.memory
            else None
        )
        self.router = MasterRouter(self.registry)
        self.orchestrator = Orchestrator(
            self.registry,
            self.router,
            self.gateway,
            self.store,
            self.events,
            self.rules,
            settings=settings,
            memory=self.memory,
            tools=self.tools,
            verifier=self.verifier,
            book_pipeline=self.book_pipeline,
            slides_pipeline=self.slides_pipeline,
        )
        self.evolution = EvolutionEngine(self.registry)

    async def init(self) -> None:
        await self.store.init()
        if self.memory is not None:
            await self.memory.init()


def build_runtime() -> Runtime:
    return Runtime(Settings())


def build_model_catalog(settings: Settings) -> list[ModelSpec]:
    """Expose the full model catalog (local + free + paid) with tiers for the API/CLI."""
    ids = [
        *settings.local_models,
        *settings.free_models,
        settings.default_model,
        *settings.fallback_models,
    ]
    specs: list[ModelSpec] = []
    for mid in ids:
        provider = mid.partition("/")[0]
        tier = (
            ModelTier.LOCAL
            if provider == "ollama"
            else ModelTier.FREE
            if provider in _FREE_PROVIDERS
            else ModelTier.PAID
        )
        specs.append(
            ModelSpec(
                id=mid, provider=provider, capabilities=["reasoning"], tier=tier, enabled=True
            )
        )
    return specs
