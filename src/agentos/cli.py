import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

from .runtime import build_runtime

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def doctor() -> None:
    rt = build_runtime()
    console.print("[bold]AgentOS production runtime[/bold]")
    console.print(f"agents: {len(rt.registry.agents())}")
    console.print(f"skills: {len(rt.registry.skills())}")
    console.print(f"models: {len(rt.registry.models())}")
    console.print(f"hooks: {', '.join(rt.hooks.names())}")
    console.print(f"rules: {len(rt.rules.rules)}")


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
def run(goal: str) -> None:
    async def main() -> None:
        rt = build_runtime()
        await rt.init()
        task, output, route = await rt.orchestrator.run(goal)
        console.print(f"[green]{task.status}[/green] {task.id}")
        console.print(output)
        console.print(json.dumps(route.model_dump(), indent=2))

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
