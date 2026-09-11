from pathlib import Path

from .builtin_hooks import AuditHook, ErrorLogHook
from .events import EventBus
from .evolution import EvolutionEngine
from .gateway import ModelGateway
from .hooks import HookManager
from .logging import configure_logging
from .models import AgentSpec, ModelSpec, SkillSpec
from .orchestrator import Orchestrator
from .registry import Registry
from .router import MasterRouter
from .rules import default_rules
from .security import SecurityEngine
from .settings import Settings
from .store import TaskStore
from .tools import ToolRuntime


def build_registry() -> Registry:
    r = Registry()
    data = [
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
            ["research"],
            ["web-research", "source-evaluation", "summarization"],
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
            ["documentation"],
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
    ]
    for id_, mission, caps, skills, tools in data:
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
    skills = [
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
        ("source-evaluation", "Source quality evaluation", ["research"]),
        ("summarization", "Accurate summarization", ["writing"]),
        ("technical-documentation", "Technical docs", ["writing"]),
    ]
    for id_, desc, tags in skills:
        r.add_skill(SkillSpec(id=id_, description=desc, tags=tags, instructions=desc))
    for id_, provider, caps in [
        ("openai/gpt-4o-mini", "openai", ["coding", "reasoning", "tool_calling"]),
        ("anthropic/claude-3-5-haiku-latest", "anthropic", ["coding", "reasoning", "tool_calling"]),
        ("gemini/gemini-2.0-flash", "gemini", ["coding", "reasoning", "long_context"]),
    ]:
        r.add_model(ModelSpec(id=id_, provider=provider, capabilities=caps))
    return r


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
        self.gateway = ModelGateway(settings.fallback_models)
        self.router = MasterRouter(self.registry)
        self.orchestrator = Orchestrator(
            self.registry, self.router, self.gateway, self.store, self.events, self.rules
        )
        self.evolution = EvolutionEngine(self.registry)

    async def init(self) -> None:
        await self.store.init()


def build_runtime() -> Runtime:
    return Runtime(Settings())
