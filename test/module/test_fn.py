import pytest
import torch

from torcharc.module import fn


@pytest.mark.parametrize(
    "name, dim",
    [
        ("mean", 1),
        ("mean", [1]),
        ("sum", 1),
        ("prod", 1),
    ],
)
def test_reduce(name, dim):
    model = fn.Reduce(name, dim=dim)
    assert isinstance(model, torch.nn.Module)

    x = torch.randn(4, 3)
    y = model(x)
    assert y.shape == (4,)

    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x).shape == y.shape
    traced_model = torch.jit.trace(model, (x))
    assert traced_model(x).shape == y.shape


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("abs", {}),
        ("sqrt", {}),
        ("topk", {"k": 3}),
        ("mean", {"dim": 1}),
        ("mean", {"dim": [1]}),
    ],
)
def test_torch_fn(name, kwargs):
    model = fn.TorchFn(name, **kwargs)
    assert isinstance(model, torch.nn.Module)

    x = torch.randn(4, 3)
    model(x)

    torch.compile(model)
    torch.jit.trace(model, (x))
