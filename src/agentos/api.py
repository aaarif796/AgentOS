from fastapi import FastAPI
from pydantic import BaseModel

from .models import ModelSpec
from .runtime import build_model_catalog, build_runtime

app = FastAPI(title="AgentOS", version="0.2.0")
runtime = build_runtime()


class TaskRequest(BaseModel):
    goal: str


class PipelineRequest(BaseModel):
    topic: str
    approve_all: bool = True
    resume_id: str | None = None


@app.on_event("startup")
async def startup() -> None:
    await runtime.init()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "agentos"}


@app.get("/agents")
async def agents() -> list[dict[str, object]]:
    return [x.model_dump() for x in runtime.registry.agents()]


@app.get("/skills")
async def skills() -> list[dict[str, object]]:
    return [x.model_dump() for x in runtime.registry.skills()]


@app.get("/models")
async def models(health: bool = False) -> list[dict[str, object]]:
    rows: list[ModelSpec] = build_model_catalog(runtime.settings)
    if health:
        report = {item["model"]: item for item in runtime.gateway.health_report()}
        return [{**m.model_dump(), "health": report.get(m.id, {})} for m in rows]
    return [m.model_dump() for m in rows]


@app.post("/tasks")
async def tasks(req: TaskRequest) -> dict[str, object]:
    task, output, route = await runtime.orchestrator.run(req.goal)
    route_data = route.model_dump() if hasattr(route, "model_dump") else {}
    return {
        "task": task.model_dump(mode="json"),
        "output": output,
        "verification": task.checkpoint.get("verification"),
        "route": route_data,
    }


@app.post("/books")
async def books(req: PipelineRequest) -> dict[str, object]:
    if runtime.book_pipeline is None:
        return {"status": "unavailable", "reason": "memory disabled"}
    manifest = await runtime.book_pipeline.supervise(
        req.topic, resume_id=req.resume_id, approve_all=req.approve_all
    )
    return manifest.model_dump(mode="json")


@app.post("/slides")
async def slides(req: PipelineRequest) -> dict[str, object]:
    if runtime.slides_pipeline is None:
        return {"status": "unavailable", "reason": "memory disabled"}
    path = await runtime.slides_pipeline.supervise(req.topic, resume_id=req.resume_id)
    return {"path": str(path)}
