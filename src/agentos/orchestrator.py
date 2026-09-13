from __future__ import annotations

from typing import Any

import structlog

from agentos.bookstudio.pipeline import BookPipeline
from agentos.events import Event, EventBus
from agentos.gateway import ModelError, ModelGateway
from agentos.memory.hub import MemoryHub
from agentos.models import ModelRequest, Task, TaskStatus, VerificationReport
from agentos.reflexion import Reflector, Verifier, extract_tool_calls
from agentos.registry import Registry
from agentos.router import MasterRouter
from agentos.rules import RuleEngine
from agentos.settings import Settings
from agentos.slides.pipeline import PresentationPipeline
from agentos.store import TaskStore
from agentos.tools import ToolRuntime

log = structlog.get_logger()

AGENTS_WITH_SYSTEM = {
    "book-supervisor",
    "presentation-supervisor",
}


class Orchestrator:
    """Universal Execute -> Verify -> Reflect/Retry loop for every query.

    Every user-facing response passes through the verify gate. Failures feed
    the reflector and retry with an improved prompt up to `settings.max_attempts`.
    """

    def __init__(
        self,
        registry: Registry,
        router: MasterRouter,
        gateway: ModelGateway,
        store: TaskStore,
        events: EventBus,
        rules: RuleEngine,
        settings: Settings | None = None,
        memory: MemoryHub | None = None,
        tools: ToolRuntime | None = None,
        verifier: Verifier | None = None,
        book_pipeline: BookPipeline | None = None,
        slides_pipeline: PresentationPipeline | None = None,
    ) -> None:
        self.registry, self.router, self.gateway = registry, router, gateway
        self.store, self.events, self.rules = store, events, rules
        self.settings = settings or Settings()
        self.memory = memory
        self.tools = tools
        self.verifier = verifier
        self.reflector = Reflector()
        self.book_pipeline = book_pipeline
        self.slides_pipeline = slides_pipeline
        self._tokens_spent = 0

    async def run(self, goal: str) -> tuple[Task, str, Any]:
        task = Task(goal=goal)
        task.status = TaskStatus.PLANNING
        task.checkpoint = {"attempts": 0, "feedback": []}
        await self.store.save(task)
        self.events.publish(Event("TASK_CREATED", {"task_id": task.id}))
        route = self.router.route(task)
        node = route.nodes[0]
        agent = self.registry.agent(node.agent_id)
        task.current_agent = agent.id
        task.status = TaskStatus.RUNNING

        if agent.id == "book-supervisor" and self.book_pipeline is not None:
            return await self._run_book_pipeline(task, goal, route, agent.id)
        if agent.id == "presentation-supervisor" and self.slides_pipeline is not None:
            return await self._run_slides_pipeline(task, goal, route, agent.id)

        system = self._build_system(agent.id, node.skill_ids)
        context = await self._build_memory_context(goal, agent.id, task.id)
        budget = _TokenBudget(self.settings.task_token_budget)

        candidates = node.model_ids or None
        last_output = ""
        feedback = ""
        last_report: VerificationReport | None = None

        for attempt in range(1, self.settings.max_attempts + 1):
            task.checkpoint["attempts"] = attempt
            user_msg = goal + (f"\n\n{feedback}" if feedback else "")
            request = ModelRequest(
                model=self.settings.default_model, messages=_messages(system, context, user_msg)
            )
            if not budget.can_afford(request):
                task.checkpoint["budget_exhausted"] = True
                break
            try:
                response, errors = self.gateway.complete_with_fallback(
                    request, candidates=candidates
                )
            except ModelError as exc:
                task.checkpoint["error"] = str(exc)
                task.status = TaskStatus.FAILED
                await self.store.save(task)
                self.events.publish(Event("TASK_FAILED", {"task_id": task.id, "error": str(exc)}))
                return task, str(exc), route
            task.current_model = response.model
            task.checkpoint["fallback_errors"] = errors
            task.checkpoint["model_used"] = response.model
            budget.spend(request, response)

            output = await self._run_tool_loop(
                task, agent.id, system, context, user_msg, response.text, budget
            )
            last_output = output

            report = await self._verify(task, output)
            last_report = report
            task.checkpoint["verification"] = report.model_dump()

            if report.passed:
                task.status = TaskStatus.COMPLETED
                task.checkpoint["score"] = report.score
                await self._capture(task, goal, output, verified=True)
                await self.store.save(task)
                self.events.publish(
                    Event(
                        "TASK_COMPLETED",
                        {"task_id": task.id, "model": response.model, "score": report.score},
                    )
                )
                return task, output, route

            feedback = self.reflector.build(report, goal, output, attempt)
            task.checkpoint["feedback"] = task.checkpoint.get("feedback", []) + [feedback]
            await self.store.save(task)
            self.events.publish(
                Event(
                    "TASK_VERIFICATION_FAILED",
                    {"task_id": task.id, "attempt": attempt, "score": report.score},
                )
            )

        task.status = TaskStatus.FAILED
        task.checkpoint["final_report"] = last_report.model_dump() if last_report else None
        await self._capture(task, goal, last_output, verified=False, feedback=feedback)
        await self.store.save(task)
        self.events.publish(
            Event("TASK_FAILED", {"task_id": task.id, "error": "verification never passed"})
        )
        return task, last_output, route

    # -- pipeline dispatchers ---------------------------------------------------
    async def _run_book_pipeline(
        self, task: Task, goal: str, route: object, agent_id: str
    ) -> tuple[Task, str, object]:
        pipeline = self.book_pipeline
        if pipeline is None:
            raise RuntimeError("book pipeline unavailable")
        manifest = await pipeline.supervise(goal)
        task.status = TaskStatus.COMPLETED
        task.current_agent = agent_id
        task.artifacts = list(manifest.artifacts.values())
        task.checkpoint = {
            "pipeline": "book",
            "manifest_id": manifest.id,
            "status": manifest.status,
        }
        await self.store.save(task)
        message = f"Book '{manifest.title}' generated.\nStatus: {manifest.status}\n" + "\n".join(
            f"- {p}" for p in manifest.artifacts.values()
        )
        self.events.publish(
            Event("BOOK_COMPLETED", {"task_id": task.id, "manifest_id": manifest.id})
        )
        return task, message, route

    async def _run_slides_pipeline(
        self, task: Task, goal: str, route: object, agent_id: str
    ) -> tuple[Task, str, object]:
        pipeline = self.slides_pipeline
        if pipeline is None:
            raise RuntimeError("slides pipeline unavailable")
        pptx_path = await pipeline.supervise(goal)
        task.status = TaskStatus.COMPLETED
        task.current_agent = agent_id
        task.artifacts = [str(pptx_path)]
        task.checkpoint = {"pipeline": "slides", "path": str(pptx_path)}
        await self.store.save(task)
        message = f"Presentation generated: {pptx_path}"
        self.events.publish(Event("DECK_COMPLETED", {"task_id": task.id, "path": str(pptx_path)}))
        return task, message, route

    # -- core helpers ----------------------------------------------------------
    def _build_system(self, agent_id: str, skill_ids: list[str]) -> str:
        agent = self.registry.agent(agent_id)
        skill_blocks: list[str] = []
        for sid in skill_ids:
            if any(x.id == sid for x in self.registry.skills()):
                skill_blocks.append(f"## {sid}\n{self.registry.skill(sid).instructions}")
        skill_text = "\n".join(skill_blocks)
        tool_hint = ""
        if agent.tools:
            tool_hint = (
                "\nWhen you need external information or to modify files, emit exactly:\n"
                'TOOL_CALL {"tool": "<id>", "args": {...}}\n'
                f"Available tools for you: {', '.join(agent.tools)}\n"
                "The runtime executes the tool and returns TOOL result. Do not fake results."
            )
        return (
            f"You are {agent_id}. Mission: {agent.mission}\n"
            f"Rules:\n{self.rules.system_directives()}\n"
            f"Skills:\n{skill_text}\n{tool_hint}\n"
            "Do not invent actions, tool results, files, tests, or evidence."
        )

    async def _build_memory_context(self, goal: str, agent_id: str, scope: str) -> str:
        if self.memory is None:
            return ""
        return await self.memory.build_context(goal, agent_id, scope)

    async def _run_tool_loop(
        self,
        task: Task,
        agent_id: str,
        system: str,
        context: str,
        user_msg: str,
        initial_text: str,
        budget: _TokenBudget,
    ) -> str:
        if self.tools is None:
            return initial_text
        agent = self.registry.agent(agent_id)
        messages = _messages(system, context, user_msg)
        text = initial_text
        for _step in range(self.settings.max_tool_steps):
            calls = extract_tool_calls(text)
            if not calls:
                return text
            messages.append({"role": "assistant", "content": text})
            observations: list[str] = []
            for tool_id, args in calls:
                try:
                    result = self.tools.execute(agent, tool_id, args)
                except Exception as exc:
                    result = f"ERROR: {exc}"
                observations.append(f"TOOL {tool_id} result: {result[:4000]}")
                self.events.publish(Event("TOOL_EXECUTED", {"task_id": task.id, "tool": tool_id}))
            messages.append(
                {
                    "role": "user",
                    "content": "\n".join(observations) + "\n\nProceed or provide the final answer.",
                }
            )
            request = ModelRequest(
                model=task.current_model or self.settings.default_model, messages=messages
            )
            if not budget.can_afford(request):
                break
            try:
                response, _errors = self.gateway.complete_with_fallback(
                    request, candidates=[task.current_model] if task.current_model else None
                )
            except ModelError as exc:
                return text + f"\n[tool loop stopped: {exc}]"
            budget.spend(request, response)
            text = response.text
        return text

    async def _verify(self, task: Task, output: str) -> VerificationReport:
        if self.verifier is None:
            return VerificationReport(
                task_id=task.id,
                passed=bool(output.strip()),
                score=0.9 if output.strip() else 0.0,
                checks_run=["deterministic-only"],
            )
        return await self.verifier.verify(task, output, model=task.current_model or "")

    async def _capture(
        self, task: Task, goal: str, output: str, verified: bool, feedback: str = ""
    ) -> None:
        if self.memory is None:
            return
        await self.memory.capture(task.id, goal, output, verified, feedback)


def _messages(system: str, context: str, user: str) -> list[dict[str, str]]:
    system_content = system
    if context:
        system_content += f"\n\n## Memory\n{context}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user},
    ]


class _TokenBudget:
    """Rough token accounting to keep long pipelines inside the configured budget."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.spent = 0

    @staticmethod
    def _estimate(messages: list[dict[str, str]]) -> int:
        return sum(len(m["content"]) for m in messages) // 3

    def can_afford(self, request: ModelRequest) -> bool:
        return self.spent + self._estimate(request.messages) <= self.limit

    def spend(self, request: ModelRequest, response: object) -> None:
        used = getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0)
        if not used:
            used = self._estimate(request.messages)
        self.spent += used
