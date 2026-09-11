from .models import ExecutionRoute, RouteNode, Task
from .registry import Registry


class MasterRouter:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def route(self, task: Task, strategy: str = "balanced") -> ExecutionRoute:
        matches = self.registry.matching_agents(task.goal)
        selected = matches[0][1]
        if selected.id == "orchestrator" and len(matches) > 1:
            selected = matches[1][1]
        models = [m.id for m in self.registry.models() if m.enabled][:4]
        node = RouteNode(
            id="main",
            agent_id=selected.id,
            skill_ids=selected.skills,
            tool_ids=selected.tools,
            model_ids=models,
        )
        score = matches[0][0] if matches else 0
        confidence = min(0.98, 0.35 + score / max(1.0, len(task.goal.split())))
        return ExecutionRoute(
            task_id=task.id,
            strategy=strategy,
            nodes=[node],
            confidence=confidence,
            rationale=[
                f"agent capability score={score}",
                f"selected={selected.id}",
                f"models={','.join(models)}",
            ],
        )

    def explain(self, goal: str) -> ExecutionRoute:
        return self.route(Task(goal=goal))
