import pytest
import torch

import torcharc

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
    spec = torcharc.Spec(**spec_dict)
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
    spec_file = torcharc.SPEC_DIR / "basic" / "mlp.yaml"
    model = torcharc.build(spec_file)
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
    model = torcharc.build(spec_dict)
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


def test_register_nn():
    # test register new layer
    class MyLayer(torch.nn.Module):
        def forward(self, x):
            return x

    torcharc.register_nn(MyLayer)
    assert torch.nn.MyLayer == MyLayer


def test_register_conflict():
    class Linear(torch.nn.Module):
        def forward(self, x):
            return x

    # conflict with existing torch.nn.Linear - not allowed
    with pytest.raises(ValueError):
        torcharc.register_nn(Linear)


# test all the example specs
@pytest.mark.parametrize("spec_file", list(torcharc.SPEC_DIR.rglob("*.yaml")))
def test_build_compile(spec_file):
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)
    # Test compatibility with torch.compile
    torch.compile(model)
