from fastapi import FastAPI
from pydantic import BaseModel

from .runtime import build_runtime

app = FastAPI(title="AgentOS", version="0.2.0")
runtime = build_runtime()


class TaskRequest(BaseModel):
    goal: str


@app.on_event("startup")
async def startup() -> None:
    await runtime.init()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "agentos"}


@app.get("/agents")
async def agents():
    return [x.model_dump() for x in runtime.registry.agents()]


@app.get("/skills")
async def skills():
    return [x.model_dump() for x in runtime.registry.skills()]


@app.get("/models")
async def models():
    return [x.model_dump() for x in runtime.registry.models()]


@app.post("/tasks")
async def tasks(req: TaskRequest):
    task, output, route = await runtime.orchestrator.run(req.goal)
    return {"task": task.model_dump(mode="json"), "output": output, "route": route.model_dump()}
