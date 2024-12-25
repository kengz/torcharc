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
    modules: dict[str, list[str | list[str]] | dict[str, str]] = Field(
        description="fx.Graph nodes and their inputs to use in call_module - where key is node name, and value is args if list or kwargs if dict",
        examples=[
            {"mlp": ["x"]},
            {"mlp": [["x_0", "x_1"]]},
            {"transformer": {"src": "src_embed", "tgt": "tgt_embed"}},
        ],
    )
    output: str | list[str] | dict[str, str] = Field(
        description="Output of fx.Graph - this can be string (single node), list (tuple), or dict",
        examples=["mlp", ["mlp_0", "mlp_1"], {"y_0": "mlp_0", "y_1": "mlp_1"}],
    )

    _reuse_delim: str = "~"  # delimiter for reusing modules with different inputs

    def __parse_reuse_name(self, name: str) -> tuple[str, str]:
        """
        Parse reuse names like `conv~left` (`conv` reused with inputs `left`) into `conv~left` for node name and `conv` for module_name, e.g. for conv~left, conv~right for stereoscopic model
        """
        if "~" in name:
            module_name, _ = name.split(self._reuse_delim)
            node_name = name
            return node_name, module_name
        return name, name

    def build_input(self, graph: fx.Graph, _nodes: dict) -> None:
        if isinstance(self.input, str):
            _nodes[self.input] = graph.placeholder(self.input)
        else:  # list[str]
            for name in self.input:
                _nodes[name] = graph.placeholder(name)

    def map_nodes(self, _nodes: dict, in_nodes: list[str | list[str]]) -> list[fx.Node]:
        result = []
        for n in in_nodes:
            if isinstance(n, list):
                # Recursively process the list if `n` is a list
                result.append(self.map_nodes(_nodes, n))
            else:
                # Process the single node
                result.append(_nodes[n])
        return result

    def build_modules(self, graph: fx.Graph, _nodes: dict) -> None:
        for name, in_nodes in self.modules.items():
            node_name, module_name = self.__parse_reuse_name(name)
            if isinstance(in_nodes, list):
                args, kwargs = tuple(self.map_nodes(_nodes, in_nodes)), {}
            else:  # dict
                args, kwargs = (), {k: _nodes[n] for k, n in in_nodes.items()}
            _nodes[node_name] = graph.call_module(module_name, args=args, kwargs=kwargs)

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
