# Pydantic validation for modules spec
from pydantic import BaseModel, Field, RootModel, field_validator
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


class CompactLayerSpec(BaseModel):
    """
    Spec to compactly specify multiple layers with common kwargs keys and values list for the layers.
    The following

    type: <torch.nn class name>
    keys: <class kwargs keys>
    args: [class kwargs values]

    expands into

    [{<type>: {<keys>: <arg0>}}, ..., {<type>: {<keys>: <argN>}}]
    """

    type: str = Field(
        description="Name of a torch.nn class", examples=["LazyLinear", "LazyConv2d"]
    )
    keys: list[str] = Field(
        description="The class' kwargs keys, to be expanded and zipped with args.",
        examples=[["out_features"], ["out_channels", "kernel_size"]],
    )
    args: list[list] = Field(
        description="The class' kwargs values for each layer. For convenience this will be casted to list of lists to allow a list of singleton values.",
        examples=[[64, 64, 32, 16], [[16, 2], [32, 3], [64, 4]]],
    )

    @field_validator("args", mode="before")
    def cast_list_of_list(value: list) -> list[list]:
        return [v if isinstance(v, list) else [v] for v in value]


class CompactValueSpec(BaseModel):
    """Intermediate spec defining the values of CompactSpec"""

    prelayer: list[NNSpec] | None = Field(
        None,
        description="The optional list of NNSpec layers that repeat before the mid layer.",
    )
    layer: CompactLayerSpec = Field(
        description="The mid layer to be expanded, wrapped between prelayer and postlayer, and repeated."
    )
    postlayer: list[NNSpec] | None = Field(
        None,
        description="The optional list of NNSpec layers that repeat after the mid layer.",
    )


class CompactSpec(RootModel):
    """
    Higher level compact spec that expands into Sequential spec. This is useful for architecture search.
    Compact spec has the format:

    compact:
        prelayer: [NNSpec]
        layer:
            type: <torch.nn class name>
            keys: <class kwargs keys>
            args: [class kwargs values]
        postlayer: [NNSpec]

    E.g.
    compact:
        layer:
            type: LazyLinear
            keys: [out_features]
            args: [64, 64, 32, 16]
        postlayer:
            - ReLU:

    E.g.
    compact:
        prelayer:
            - LazyBatchNorm2d:
        layer:
            type: LazyConv2d
            keys: [out_channels, kernel_size]
            args: [[16, 2], [32, 3], [64, 4]]
        postlayer:
            - ReLU:
            - Dropout:
                p: 0.1
    """

    root: dict[str, CompactValueSpec] = Field(
        description="Higher level compact spec that expands into Sequential spec.",
        examples=[
            {
                "compact": {
                    "layer": {
                        "type": "LazyLinear",
                        "keys": ["out_features"],
                        "args": [64, 64, 32, 16],
                    },
                    "postlayer": [{"ReLU": {}}],
                }
            },
            {
                "compact": {
                    "prelayer": [{"LazyBatchNorm2d": {}}],
                    "layer": {
                        "type": "LazyConv2d",
                        "keys": ["out_channels", "kernel_size"],
                        "args": [[16, 2], [32, 3], [64, 4]],
                    },
                    "postlayer": [{"ReLU": {}}, {"Dropout": {"p": 0.1}}],
                }
            },
        ],
    )

    @field_validator("root", mode="before")
    def is_single_key_dict(value: dict) -> dict:
        return NNSpec.is_single_key_dict(value)

    @field_validator("root", mode="before")
    def key_is_compact(value: dict) -> dict:
        assert (
            next(iter(value)) == "compact"
        ), "Key must be 'compact' if using CompactSpec."
        return value

    def __expand_spec(self, compact_layer: dict) -> list[dict]:
        class_name = compact_layer["type"]
        keys = compact_layer["keys"]
        args = compact_layer["args"]
        nn_specs = []
        for vals in args:
            nn_spec = {class_name: dict(zip(keys, vals))}
            nn_specs.append(nn_spec)
        return nn_specs

    def expand_to_sequential_spec(self) -> SequentialSpec:
        compact_spec = next(iter(self.root.values())).model_dump()
        prelayer = compact_spec.get("prelayer")
        postlayer = compact_spec.get("postlayer")
        nn_specs = []
        for midlayer in self.__expand_spec(compact_spec["layer"]):
            nn_specs.extend(prelayer) if prelayer else True
            nn_specs.append(midlayer)
            nn_specs.extend(postlayer) if postlayer else True
        return SequentialSpec(**{"Sequential": nn_specs})

    def build(self) -> nn.Sequential:
        """Build nn.Sequential from compact spec expanded into sequential spec"""
        return self.expand_to_sequential_spec().build()


class ModuleSpec(RootModel):
    """
    Higher level module spec where value can be NNSpec, SequentialSpec, or CompactSpec.
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

    E.g. (compact)
    compact:
        layer:
            type: LazyLinear
            keys: [out_features]
            args: [64, 64, 32, 16]
        postlayer:
            - ReLU:
    """

    root: NNSpec | SequentialSpec | CompactSpec = Field(
        description="Higher level module spec where value can be NNSpec, SequentialSpec, or CompactSpec.",
        examples=[
            {"Linear": {"in_features": 128, "out_features": 64}},
            {"Sequential": [{"Linear": {"in_features": 128, "out_features": 64}}]},
            {
                "compact": {
                    "prelayer": [{"LazyBatchNorm2d": {}}],
                    "layer": {
                        "type": "LazyConv2d",
                        "keys": ["out_channels", "kernel_size"],
                        "args": [[16, 2], [32, 3], [64, 4]],
                    },
                    "postlayer": [{"ReLU": {}}, {"Dropout": {"p": 0.1}}],
                }
            },
        ],
    )

    @field_validator("root", mode="before")
    def is_single_key_dict(value: dict) -> dict:
        return NNSpec.is_single_key_dict(value)

    def build(self) -> nn.Module:
        """Build nn.Module from module spec."""
        return self.root.build()
