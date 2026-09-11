# AgentOS

Production-oriented foundation for a model-agnostic multi-agent operating system.

## Implemented

- Agent / Skill / Tool / Model / Task contracts
- MasterRouter
- Orchestrator
- Model Gateway with provider adapters
- model fallback
- checkpoint persistence
- EventBus
- lifecycle hooks
- rule engine
- permission engine
- audit hook
- EvolutionEngine
- FastAPI
- CLI
- Ruff
- mypy
- Pyright
- Pytest
- Tach (module boundaries)
- pre-commit
- GitHub Actions

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

agentos doctor
agentos agents
agentos skills
agentos run "Explain this project and identify its highest priority engineering risks."
```

Set at least one provider API key in `.env` for actual model execution.

## Important

This repository is a production-oriented **foundation**, not a claim that every subsystem is production-safe for arbitrary untrusted workloads. In particular, terminal execution must be isolated before exposing AgentOS to untrusted agents/users.
