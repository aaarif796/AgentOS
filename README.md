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

## Inventory

Current runtime contents (see `agentos doctor` / `agentos agents` / `agentos skills`):

| Asset  | Count | Details |
| ------ | :---: | ------- |
| Agents |  10   | orchestrator, planner, software-architect, developer, researcher, code-reviewer, tester, security-auditor, documentation-writer, devops-engineer |
| Skills |  22   | python, javascript, typescript, git, clean-code, system-design, api-design, database-design, unit-testing, integration-testing, tdd, owasp, secrets-management, dependency-security, authentication, docker, ci-cd, linux, web-research, source-evaluation, summarization, technical-documentation |
| Models |   3   | openai/gpt-4o-mini (default), anthropic/claude-3-5-haiku-latest (fallback), gemini/gemini-2.0-flash (fallback) |
| Tools  |   4   | filesystem, terminal, git, http |
| Hooks  |   2   | audit, error-log |
| Rules  |   5   | security-first, least-privilege, truthful-output, checkpoint, evolution-gated |

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
