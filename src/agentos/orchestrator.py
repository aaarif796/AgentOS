from __future__ import annotations

import structlog

from .events import Event, EventBus
from .gateway import ModelError, ModelGateway
from .models import ModelRequest, Task, TaskStatus
from .registry import Registry
from .router import MasterRouter
from .rules import RuleEngine
from .store import TaskStore

log = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        registry: Registry,
        router: MasterRouter,
        gateway: ModelGateway,
        store: TaskStore,
        events: EventBus,
        rules: RuleEngine,
    ) -> None:
        self.registry, self.router, self.gateway = registry, router, gateway
        self.store, self.events, self.rules = store, events, rules

    async def run(self, goal: str) -> tuple[Task, str, object]:
        task = Task(goal=goal)
        task.status = TaskStatus.PLANNING
        await self.store.save(task)
        self.events.publish(Event("TASK_CREATED", {"task_id": task.id}))
        route = self.router.route(task)
        node = route.nodes[0]
        agent = self.registry.agent(node.agent_id)
        task.current_agent = agent.id
        task.status = TaskStatus.RUNNING

        skill_text = "\n".join(
            f"## {self.registry.skill(s).id}\n{self.registry.skill(s).instructions}"
            for s in node.skill_ids
            if any(x.id == s for x in self.registry.skills())
        )
        system = (
            f"You are {agent.id}. Mission: {agent.mission}\n"
            f"Rules:\n{self.rules.system_directives()}\n"
            f"Skills:\n{skill_text}\n"
            "Do not invent actions, tool results, files, tests, or evidence."
        )
        task.messages = [{"role": "system", "content": system}, {"role": "user", "content": goal}]
        await self.store.save(task)

        candidates = node.model_ids
        if not candidates:
            candidates = [route.nodes[0].model_ids[0]] if route.nodes[0].model_ids else []

        try:
            if not candidates:
                raise ModelError("No enabled models are registered.", retryable=False)
            request = ModelRequest(model=candidates[0], messages=task.messages)
            response, errors = self.gateway.complete_with_fallback(request)
            task.current_model = response.model
            task.checkpoint = {"model": response.model, "fallback_errors": errors}
            task.messages.append({"role": "assistant", "content": response.text})
            task.status = TaskStatus.COMPLETED
            await self.store.save(task)
            self.events.publish(
                Event("TASK_COMPLETED", {"task_id": task.id, "model": response.model})
            )
            return task, response.text, route
        except ModelError as exc:
            task.status = TaskStatus.FAILED
            task.checkpoint = {"error": str(exc), "messages": task.messages}
            await self.store.save(task)
            self.events.publish(Event("TASK_FAILED", {"task_id": task.id, "error": str(exc)}))
            return task, str(exc), route
