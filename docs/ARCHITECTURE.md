# AgentOS Production Architecture

```text
CLI / TUI / Studio
       |
      API
       |
 Orchestrator
       |
 MasterRouter ---- Registry
       |             |
       |          Agents/Skills/Tools/Models
       |
 Model Gateway ---- Provider adapters
       |
  checkpointed Task
       |
 EventBus ---- Hooks ---- Audit/Observability
       |
 SecurityEngine ---- ToolRuntime
       |
 EvolutionEngine ---- Candidate -> Sandbox -> Test -> Evaluate -> Promote
```

## Production hardening still required before unrestricted deployment

- PostgreSQL + pgvector migration for multi-user deployments.
- Redis for distributed locks/cache.
- Container/VM sandbox for untrusted terminal and generated code.
- Secret manager integration.
- Authentication/authorization at the API boundary.
- Rate limiting and request quotas.
- Durable distributed event bus for horizontal workers.
- OpenTelemetry traces/metrics.
- Formal model health/quota accounting.
- Human approval for high-risk tools and evolution promotion.
- Full workflow graph execution and resumable parallel steps.
- MCP isolation and per-server permissions.
- Backup/restore and database migrations.
