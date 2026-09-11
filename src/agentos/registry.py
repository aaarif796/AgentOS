from .models import AgentSpec, ModelSpec, SkillSpec, ToolSpec


class Registry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentSpec] = {}
        self._skills: dict[str, SkillSpec] = {}
        self._tools: dict[str, ToolSpec] = {}
        self._models: dict[str, ModelSpec] = {}

    def add_agent(self, value: AgentSpec) -> None:
        self._agents[value.id] = value

    def add_skill(self, value: SkillSpec) -> None:
        self._skills[value.id] = value

    def add_tool(self, value: ToolSpec) -> None:
        self._tools[value.id] = value

    def add_model(self, value: ModelSpec) -> None:
        self._models[value.id] = value

    def agents(self) -> list[AgentSpec]:
        return list(self._agents.values())

    def skills(self) -> list[SkillSpec]:
        return list(self._skills.values())

    def tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def models(self) -> list[ModelSpec]:
        return list(self._models.values())

    def agent(self, key: str) -> AgentSpec:
        return self._agents[key]

    def skill(self, key: str) -> SkillSpec:
        return self._skills[key]

    def matching_agents(self, goal: str) -> list[tuple[float, AgentSpec]]:
        words = {w.strip(".,:;!?()[]{}").lower() for w in goal.split() if len(w) > 3}
        scored: list[tuple[float, AgentSpec]] = []
        for agent in self._agents.values():
            text = " ".join([agent.id, agent.mission, *agent.capabilities, *agent.skills]).lower()
            score = float(sum(word in text for word in words))
            scored.append((score, agent))
        return sorted(scored, key=lambda item: item[0], reverse=True)
