# Pydantic validation for modules spec
from pydantic import RootModel, Field, field_validator
from torch import nn
import yaml


class ModuleSpec(RootModel):
    """
    The basic module spec used to construct a nn.Module, where key = module class name, and value = kwargs, i.e. nn.<key>(**value).
    Must be a single-key dict.
    """

    root: dict[str, dict] = Field(
        description="Module spec for nn.Module, with key as module class name, and value as kwargs.",
        examples=[{"Linear": {"in_features": 128, "out_features": 64}}, {"ReLU": {}}],
    )

    @field_validator("root", mode="before")
    def is_single_key_dict(value: dict) -> dict:
        assert len(value) == 1, "Module spec must be a single-key dict."
        return value

    @field_validator("root", mode="before")
    def null_kwargs_to_dict(value: dict) -> dict:
        key = next(iter(value))
        if value[key] is None:
            value[key] = {}
        return value

    @field_validator("root", mode="after")
    def key_exists_in_nn(value: dict) -> dict:
        class_name = next(iter(value))
        try:
            getattr(nn, class_name)  # will raise AttributeError if not found
            return value
        except AttributeError as e:
            raise ValueError(e) from e

    def build(self) -> nn.Module:
        """Build nn.Module from module spec."""
        class_name = next(iter(self.root))
        cls = getattr(nn, class_name)
        kwargs = self.root[class_name]
        return cls(**kwargs)


module = ModuleSpec({"Linear": {"in_features": 128, "out_features": 64}}).build()
ModuleSpec({"ReLU": None}).build()
ModuleSpec({"ReLU": {}}).build()

ModuleSpec({"foo": {"in_features": 128, "out_features": 64}})

sample_yaml = """
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
  body:
    Linear:
      in_features: 10
      out_features: 1

graph:
  input: x
  modules:
    mlp: [x]
    body: [mlp]
  output: body
"""

# modules is a dict of module_name -> module_spec.
# where module_spec is either {Sequential: [module_spec]} or {str: kwargs}
