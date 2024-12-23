# Pydantic validation for graph spec
from pydantic import BaseModel, Field
from torch import fx


class GraphSpec(BaseModel):
    """
    Graph spec for fx.Graph, where input, modules, and output are specified.
    """

    input: str | list[str] = Field(
        description=" input placeholder nodes of fx.Graph",
        examples=["x", ["x_0, x_1"]],
    )
    modules: dict[str, list[str] | dict[str, str]] = Field(
        description="fx.Graph nodes and their inputs to use in call_module - where key is node name, and value is args if list or kwargs if dict",
        examples=[
            {"mlp": ["x"]},
            {"transformer": {"src": "src_embed", "tgt": "tgt_embed"}},
        ],
    )
    output: str | list[str] | dict[str, str] = Field(
        description="Output of fx.Graph - this can be string (single node), list (tuple), or dict",
        examples=["mlp", ["mlp_0", "mlp_1"], {"y_0": "mlp_0", "y_1": "mlp_1"}],
    )

    def build_input(self, graph: fx.Graph, _nodes: dict) -> None:
        if isinstance(self.input, str):
            _nodes[self.input] = graph.placeholder(self.input)
        else:  # list[str]
            for name in self.input:
                _nodes[name] = graph.placeholder(name)

    def build_modules(self, graph: fx.Graph, _nodes: dict) -> None:
        for name, in_nodes in self.modules.items():
            if isinstance(in_nodes, list):
                args = tuple([_nodes[name] for name in in_nodes])
                _nodes[name] = graph.call_module(name, args=args)
            else:  # dict
                kwargs = {k: _nodes[name] for k, name in in_nodes.items()}
                _nodes[name] = graph.call_module(name, kwargs=kwargs)

    def build_output(self, graph: fx.Graph, _nodes: dict) -> None:
        if isinstance(self.output, str):
            output = _nodes[self.output]
        elif isinstance(self.output, list):
            output = tuple([_nodes[name] for name in self.output])
        else:  # dict
            output = {k: _nodes[name] for k, name in self.output.items()}
        graph.output(output)

    def build(self) -> fx.Graph:
        graph = fx.Graph()
        _nodes = {}
        self.build_input(graph, _nodes)
        self.build_modules(graph, _nodes)
        self.build_output(graph, _nodes)
        return graph
