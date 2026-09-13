import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

from .runtime import build_model_catalog, build_runtime

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def doctor() -> None:
    rt = build_runtime()
    console.print("[bold]AgentOS production runtime[/bold]")
    console.print(f"agents: {len(rt.registry.agents())}")
    console.print(f"skills: {len(rt.registry.skills())}")
    models = build_model_catalog(rt.settings)
    free_count = sum(1 for m in models if m.tier.value in ("free", "local"))
    console.print(f"models: {len(models)} ({free_count} free/local)")
    console.print(f"hooks: {', '.join(rt.hooks.names())}")
    console.print(f"rules: {len(rt.rules.rules)}")
    console.print(f"memory: {'enabled' if rt.memory else 'disabled'}")
    console.print(f"model policy: {rt.settings.model_policy}")


@app.command()
def agents() -> None:
    rt = build_runtime()
    table = Table("ID", "Mission")
    for a in rt.registry.agents():
        table.add_row(a.id, a.mission)
    console.print(table)


@app.command()
def skills() -> None:
    rt = build_runtime()
    table = Table("ID", "Description")
    for s in rt.registry.skills():
        table.add_row(s.id, s.description)
    console.print(table)


@app.command()
def models(health: bool = typer.Option(False, "--health", help="Probe model availability")) -> None:
    rt = build_runtime()
    table = Table("Model", "Tier", "Healthy")
    rows = build_model_catalog(rt.settings)
    if health:
        report = {item["model"]: item for item in rt.gateway.health_report()}
    for m in rows:
        online = report.get(m.id, {}).get("online", True) if health else (m.enabled is True)
        table.add_row(m.id, m.tier.value, "yes" if online else "no")
    console.print(table)


@app.command()
def run(
    goal: str, no_verify: bool = typer.Option(False, "--no-verify", help="Skip the verify gate")
) -> None:
    async def main() -> None:
        rt = build_runtime()
        if no_verify:
            rt.verifier = None
        await rt.init()
        task, output, route = await rt.orchestrator.run(goal)
        console.print(f"[green]{task.status}[/green] {task.id}")
        if task.checkpoint.get("score") is not None:
            console.print(f"[bold]verified score: {task.checkpoint.get('score'):.2f}[/bold]")
        console.print(output)
        console.print(json.dumps(route.model_dump(), indent=2))

    asyncio.run(main())


@app.command()
def book(
    topic: str,
    resume: str = typer.Option("", "--resume", help="Book project id to resume"),
    approve: bool = typer.Option(True, "--yes", "--no-yes", help="Auto-approve chapters"),
) -> None:
    async def main() -> None:
        rt = build_runtime()
        await rt.init()
        if rt.book_pipeline is None:
            console.print("[red]book pipeline unavailable (memory disabled)[/red]")
            raise typer.Exit(1)
        manifest = await rt.book_pipeline.supervise(
            topic, resume_id=resume or None, approve_all=approve
        )
        table = Table("Format", "Path")
        for fmt, path in manifest.artifacts.items():
            table.add_row(fmt, path)
        console.print(f"[green]Book: {manifest.title}[/green] status={manifest.status}")
        console.print(table)

    asyncio.run(main())


@app.command()
def slides(
    topic: str,
    resume: str = typer.Option("", "--resume", help="Deck project id to resume"),
) -> None:
    async def main() -> None:
        rt = build_runtime()
        await rt.init()
        if rt.slides_pipeline is None:
            console.print("[red]slides pipeline unavailable (memory disabled)[/red]")
            raise typer.Exit(1)
        path = await rt.slides_pipeline.supervise(topic, resume_id=resume or None)
        console.print(f"[green]Presentation generated: {path}[/green]")

    asyncio.run(main())


@app.command()
def router(goal: str = typer.Option("", "--task"), explain: bool = False) -> None:
    rt = build_runtime()
    if explain and goal:
        console.print(json.dumps(rt.router.explain(goal).model_dump(), indent=2))
    else:
        console.print("MasterRouter ready")


@app.command()
def evolution(goal: str) -> None:
    rt = build_runtime()
    for c in rt.evolution.analyze(goal):
        console.print(json.dumps(c.model_dump(), indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("agentos.api:app", host=host, port=port)
