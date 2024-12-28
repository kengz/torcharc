# test graph validator
import pytest
from torch import fx

from torcharc.validator.graph import GraphSpec


@pytest.mark.parametrize(
    "spec_dict",
    [
        # basic: str input, str output
        {"input": "x", "modules": {"mlp": ["x"]}, "output": "mlp"},
        # list input, list modules
        {
            "input": ["x_0", "x_1"],
            "modules": {"merge": [["x_0", "x_1"]]},
            "output": "merge",
        },
        # dict modules
        {
            "input": ["src", "tgt"],
            "modules": {"transformer": {"src": "src", "tgt": "tgt"}},
            "output": "transformer",
        },
        # list output
        {"input": "x", "modules": {"mlp": ["x"]}, "output": ["mlp"]},
        # dict output
        {"input": "x", "modules": {"mlp": ["x"]}, "output": {"y": "mlp"}},
    ],
)
def test_basic(spec_dict):
    graph = GraphSpec(**spec_dict).build()
    assert isinstance(graph, fx.Graph)


def test_reuse():
    spec_dict = {
        "input": ["left", "right"],
        "modules": {"conv~left": ["left"], "conv~right": ["right"]},
        "output": {"left": "conv~left", "right": "conv~right"},
    }
    graph = GraphSpec(**spec_dict).build()
    assert isinstance(graph, fx.Graph)
    for n in graph.find_nodes(op="call_module"):
        assert n.target == "conv"
        # target is the module; graph has its own internal node name for node reuse
