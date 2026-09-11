from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    priority: int = 100
    enabled: bool = True


class RuleEngine:
    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = {r.id: r for r in (rules or []) if r.enabled}

    def system_directives(self) -> str:
        return "\n".join(
            f"- {r.description}" for r in sorted(self.rules.values(), key=lambda x: x.priority)
        )


def default_rules() -> RuleEngine:
    return RuleEngine(
        [
            Rule("security-first", "Never bypass permission checks or security policy."),
            Rule("least-privilege", "Use only tools explicitly granted to the active agent."),
            Rule(
                "truthful-output",
                "Do not claim a tool action happened unless the runtime executed it.",
            ),
            Rule("checkpoint", "Preserve task state before model fallback or recoverable failure."),
            Rule(
                "evolution-gated",
                "Generated agents, skills and workflows are candidates until evaluated and promoted.",
            ),
        ]
    )
