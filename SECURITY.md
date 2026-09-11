# Security Policy

AgentOS is an automation runtime and must be treated as privileged software.

The LLM is never the authority for permissions. Every tool invocation must pass the SecurityEngine.

For production:
- run untrusted workloads in isolated containers/VMs;
- use least privilege;
- keep credentials outside prompts;
- require approval for destructive/high-risk actions;
- audit all model/tool/task/evolution events;
- do not auto-promote generated code into the trusted runtime without evaluation;
- protect the core runtime, auth, secret store, sandbox boundary and permission engine from autonomous modification.
