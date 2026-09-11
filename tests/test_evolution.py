from agentos.evolution import EvolutionEngine
from agentos.registry import Registry


def test_kubernetes_gap():
    out = EvolutionEngine(Registry()).analyze("audit kubernetes security")
    assert out[0].id == "kubernetes-security-agent"
