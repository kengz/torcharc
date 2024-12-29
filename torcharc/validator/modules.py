# Pydantic validation for modules spec
from pydantic import Field, RootModel, field_validator
from torch import nn


class NNSpec(RootModel):
    """
    The basic spec used to construct a nn.Module, where key = nn class name, and value = kwargs, i.e. nn.<key>(**value).
    Must be a single-key dict.
    E.g.
    Linear:
        in_features: 10
        out_features: 1
    """

    root: dict[str, dict] = Field(
        description="Module spec for nn.Module, with key as nn class name, and value as kwargs.",
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
        """Build nn.Module from nn spec."""
        class_name = next(iter(self.root))
        cls = getattr(nn, class_name)
        kwargs = self.root[class_name]
        return cls(**kwargs)


class SequentialSpec(RootModel):
    """
    Sequential spec where key = Sequential and value = list of NNSpecs.
    E.g.
    Sequential:
        - Linear:
            in_features: 128
            out_features: 64
        - ReLU:
        - Linear:
            in_features: 64
            out_features: 10
    """

    root: dict[str, list[NNSpec]] = Field(
        description="Sequential module spec where value is a list of NNSpec.",
        examples=[
            {
                "Sequential": [
                    {"Linear": {"in_features": 128, "out_features": 64}},
                    {"ReLU": {}},
                    {"Linear": {"in_features": 64, "out_features": 10}},
                ]
            }
        ],
    )

    @field_validator("root", mode="before")
    def is_single_key_dict(value: dict) -> dict:
        return NNSpec.is_single_key_dict(value)

    @field_validator("root", mode="before")
    def key_is_sequential(value: dict) -> dict:
        assert (
            next(iter(value)) == "Sequential"
        ), "Key must be 'Sequential' if using SequentialSpec."
        return value

    def build(self) -> nn.Sequential:
        """Build nn.Sequential from sequential spec."""
        nn_specs = next(iter(self.root.values()))
        return nn.Sequential(*[nn_spec.build() for nn_spec in nn_specs])


class ModuleSpec(RootModel):
    """
    Higher level module spec where value can be either NNSpec or SequentialSpec.
    E.g. (plain NN)
    Linear:
        in_features: 10
        out_features: 1

    E.g. (Sequential)
    Sequential:
        - Linear:
            in_features: 128
            out_features: 64
        - ReLU:
        - Linear:
            in_features: 64
            out_features: 10
    """

    root: SequentialSpec | NNSpec = Field(
        description="Higher level module spec where value can be either NNSpec or SequentialSpec.",
        examples=[
            {"Linear": {"in_features": 128, "out_features": 64}},
            {"Sequential": [{"Linear": {"in_features": 128, "out_features": 64}}]},
        ],
    )

    @field_validator("root", mode="before")
    def is_single_key_dict(value: dict) -> dict:
        return NNSpec.is_single_key_dict(value)

    def build(self) -> nn.Module:
        """Build nn.Module from module spec."""
        return self.root.build()
