from agentos.models import AgentSpec, Task
from agentos.registry import Registry
from agentos.router import MasterRouter


def test_router_returns_agent():
    r = Registry()
    r.add_agent(AgentSpec(id="developer", mission="software development", capabilities=["coding"]))
    route = MasterRouter(r).route(Task(goal="build software"))
    assert route.nodes[0].agent_id == "developer"
