from pydantic import BaseModel
from torch import fx
import yaml
from pathlib import Path

from torcharc.validator.graph import GraphSpec
from torcharc.validator.modules import ModuleSpec


class Spec(BaseModel):
    """
    TorchArc spec for building neural networks from config using torch.fx, backed by Pydantic validators.

    Builds a torch.fx.GraphModule from a spec of 2 keys:
    - modules: {name: ModuleSpec} to define torch.nn.Module as fx.Graph Nodes. ModuleSpec is either of:
        - NNSpec: {nn_cls_name: kwargs} -> torch.nn.<nn_cls_name>(**kwargs)
        - SequentialSpec: {"Sequential": [NNSpec]} -> torch.nn.Sequential(*[s.build() for s in [NNSpec]])
    - graph: GraphSpec, to define the Graph connections. GraphSpec consists of 3 keys:
        - input: str | list[str] -> input placeholder nodes of fx.Graph, e.g. "x" or ["x_0", "x_1"]
        - modules: {name: str | list[str] | dict[str, str]} -> fx.Graph nodes and their inputs to use in call_module - where key is node name, and value is args if list or kwargs if dict, e.g. {"mlp": "x"} or {"mlp": ["x_0", "x_1"]} or {"transformer": {"src": "src_embed", "tgt": "tgt_embed"}}
        - output: str | list[str] | dict[str, str] -> output of fx.Graph - this can be string (single node), list (tuple), or dict, e.g. "mlp" or ["mlp_0", "mlp_1"] or {"y_0": "mlp_0", "y_1": "mlp_1"}

    The build method of this will build modules = {name: nn.Module} and graph = fx.Graph, and return fx.GraphModule(modules, graph).

    E.g. basic MLP

    import torch
    import yaml
    import torcharc

    yaml_str = '''
    modules:
        mlp:
            Sequential:
                - Linear:
                    in_features: 128
                    out_features: 64
                - ReLU:
                - Linear:
                    in_features: 64
                    out_features: 10

    graph:
        input: x
        modules:
            mlp: [x]
        output: mlp
    '''

    spec = yaml.safe_load(yaml_str)
    model = torcharc.Spec(**spec).build()
    model
    # =>
    # GraphModule(
    #   (mlp): Sequential(
    #     (0): Linear(in_features=128, out_features=64, bias=True)
    #     (1): ReLU()
    #     (2): Linear(in_features=64, out_features=10, bias=True)
    #   )
    # )
    x = torch.randn(1, 128)
    y = model(x)
    y
    # =>
    # tensor([[-0.3924,  0.0313,  0.2628, -0.2336,  0.5610, -0.1078, -0.1165,  0.1895,
    #           0.2207,  0.1977]], grad_fn=<AddmmBackward0>)
    """

    modules: dict[str, ModuleSpec]
    graph: GraphSpec

    def build(self) -> fx.GraphModule:
        modules = {name: module.build() for name, module in self.modules.items()}
        graph = self.graph.build()
        graph.lint()
        gm = fx.GraphModule(modules, graph)
        return gm


def build(spec: dict | str) -> fx.GraphModule:
    """
    Build fx.GraphModule from a spec dict or path to a spec dict, e.g.

    import torch
    import yaml
    import torcharc

    yaml_str = '''
    modules:
        mlp:
            Sequential:
                - Linear:
                    in_features: 128
                    out_features: 64
                - ReLU:
                - Linear:
                    in_features: 64
                    out_features: 10

    graph:
        input: x
        modules:
            mlp: [x]
        output: mlp
    '''

    spec = yaml.safe_load(yaml_str)
    model = torcharc.build(spec)
    model
    # =>
    # GraphModule(
    #   (mlp): Sequential(
    #     (0): Linear(in_features=128, out_features=64, bias=True)
    #     (1): ReLU()
    #     (2): Linear(in_features=64, out_features=10, bias=True)
    #   )
    # )
    x = torch.randn(1, 128)
    y = model(x)
    y
    # =>
    # tensor([[-0.3924,  0.0313,  0.2628, -0.2336,  0.5610, -0.1078, -0.1165,  0.1895,
    #           0.2207,  0.1977]], grad_fn=<AddmmBackward0>)
    """
    if not isinstance(spec, dict):
        with Path(spec).open("r") as f:
            spec = yaml.safe_load(f)
    return Spec(**spec).build()
