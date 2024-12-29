import pytest
import torch

from torcharc.validator.modules import ModuleSpec, NNSpec, SequentialSpec


@pytest.mark.parametrize(
    "spec_dict",
    [
        {"Linear": {"in_features": 128, "out_features": 64}},
        {"ReLU": {}},
        {"ReLU": None},
    ],
)
def test_nn_spec(spec_dict):
    module = NNSpec(**spec_dict).build()
    assert isinstance(module, torch.nn.Module)


@pytest.mark.parametrize(
    "spec_dict",
    [
        # multi-key
        {"LazyLinear": {"out_features": 64}, "ReLU": {}},
        # invalid/unregistered nn
        {"Invalid": {}},
        # init error
        {"LazyLinear": {"invalid": 64}},
        # Sequential is a container not a plain module
        {"Sequential": [{"LazyLinear": {"out_features": 64}}, {"ReLU": {}}]},
    ],
)
def test_invalid_nn_spec(spec_dict):
    with pytest.raises(Exception):
        NNSpec(**spec_dict).build()


def test_sequential_spec():
    spec_dict = {
        "Sequential": [
            {"LazyLinear": {"out_features": 64}},
            {"ReLU": {}},
            {"LazyLinear": {"out_features": 10}},
        ]
    }
    module = SequentialSpec(**spec_dict).build()
    assert isinstance(module, torch.nn.Module)


@pytest.mark.parametrize(
    "spec_dict",
    [
        # multi-key
        {"Sequential": [{"LazyLinear": {"out_features": 64}}], "ReLU": {}},
        # non-Sequential
        {"LazyLinear": {"out_features": 64}},
    ],
)
def test_invalid_sequential_spec(spec_dict):
    with pytest.raises(Exception):
        SequentialSpec(**spec_dict).build()


@pytest.mark.parametrize(
    "spec_dict",
    [
        {"LazyLinear": {"out_features": 64}},
        {"Sequential": [{"LazyLinear": {"out_features": 64}}]},
    ],
)
def test_module_spec(spec_dict):
    module = ModuleSpec(root=spec_dict).build()
    assert isinstance(module, torch.nn.Module)
