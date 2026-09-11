from .models import EvolutionCandidate
from .registry import Registry


class EvolutionEngine:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def analyze(self, goal: str) -> list[EvolutionCandidate]:
        text = goal.lower()
        existing = {x.id for x in self.registry.agents()} | {x.id for x in self.registry.skills()}
        candidates: list[EvolutionCandidate] = []
        if ("kubernetes" in text or "k8s" in text) and "kubernetes-security-agent" not in existing:
            candidates.append(
                EvolutionCandidate(
                    kind="agent",
                    id="kubernetes-security-agent",
                    reason="Kubernetes security capability gap detected.",
                    definition={
                        "mission": "Audit Kubernetes security",
                        "capabilities": ["kubernetes", "security"],
                        "skills": ["owasp", "network-security"],
                        "tools": ["filesystem", "terminal", "git"],
                    },
                )
            )
        if ("terraform" in text or "iac" in text) and "infrastructure-as-code" not in existing:
            candidates.append(
                EvolutionCandidate(
                    kind="skill",
                    id="infrastructure-as-code",
                    reason="Infrastructure-as-code workflow gap detected.",
                    definition={
                        "description": "Review and test infrastructure-as-code",
                        "tags": ["devops", "iac"],
                    },
                )
            )
        if not candidates:
            candidates.append(
                EvolutionCandidate(
                    kind="skill",
                    id="task-specific-analysis",
                    reason="No specialist matched exactly; propose a bounded candidate skill.",
                    definition={
                        "description": f"Specialized workflow for {goal}",
                        "tags": ["generated", "candidate"],
                    },
                )
            )
        return candidates
