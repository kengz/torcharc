# test graph validator
import torch
from conftest import SPEC_DIR

from torcharc.validator.spec import Spec, build

spec_dict = {
    "modules": {
        "mlp": {
            "Sequential": [
                {"Linear": {"in_features": 128, "out_features": 64}},
                {"ReLU": None},
                {"Linear": {"in_features": 64, "out_features": 10}},
            ]
        }
    },
    "graph": {
        "input": "x",
        "modules": {"mlp": ["x"]},
        "output": "mlp",
    },
}


def test_spec():
    spec = Spec(**spec_dict)
    model = spec.build()
    assert isinstance(model, torch.nn.Module)

    x = torch.randn(1, 128)
    y = model(x)
    assert y.shape == (1, 10)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x).shape == y.shape
    traced_model = torch.jit.trace(model, (x))
    assert traced_model(x).shape == y.shape


def test_build_from_file():
    spec_file = SPEC_DIR / "basic" / "mlp.yaml"
    model = build(spec_file)
    assert isinstance(model, torch.nn.Module)

    x = torch.randn(1, 128)
    y = model(x)
    assert y.shape == (1, 10)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x).shape == y.shape
    traced_model = torch.jit.trace(model, (x))
    assert traced_model(x).shape == y.shape


def test_build_from_dict():
    model = build(spec_dict)
    assert isinstance(model, torch.nn.Module)

    x = torch.randn(1, 128)
    y = model(x)
    assert y.shape == (1, 10)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x).shape == y.shape
    traced_model = torch.jit.trace(model, (x))
    assert traced_model(x).shape == y.shape
