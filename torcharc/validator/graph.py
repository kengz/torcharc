# Pydantic validation for graph spec
from pydantic import BaseModel, Field
import torch.fx as fx
# TODO fix import


class GraphSpec(BaseModel):
    input: str | list[str] = Field(
        description="Input nodes, i.e. placeholders of fx.Graph.",
        examples=["x", ["x"], ["x_0, x_1"]],
    )
    modules: dict[str, list[str] | dict[str, str]] = Field(
        description="Modules input spec for fx.Graph, with key as module node name, and value graph call_module input (args if list, kwargs if dict).",
        examples=[
            {"mlp": ["x"]},
            {"transformer": {"src": "src_embed", "tgt": "tgt_embed"}},
        ],
    )
    output: str | list[str] | dict[str, str] = Field(
        description="Output of fx.Graph - this can be string (single node), list (tuple), or dict.",
        examples=["mlp", ["mlp_0", "mlp_1"], {"y_0": "mlp_0", "y_1": "mlp_1"}],
    )

    def build(self) -> fx.Graph:
        pass
