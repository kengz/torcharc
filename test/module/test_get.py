import pytest
import torch

from torcharc.module import get


@pytest.mark.parametrize(
    "input, key, res",
    [
        (  # int key, for list/tuple
            (torch.arange(3).chunk(3)),
            0,
            torch.tensor(0),
        ),
        (  # str key, for dict
            {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)},
            "a",
            torch.tensor(1),
        ),
    ],
)
def test_get(input, key, res):
    model = get.Get(key)
    assert isinstance(model, torch.nn.Module)

    y = model(input)
    assert y == res

    compiled_model = torch.compile(model)
    assert compiled_model(input) == res


def test_narrow():
    model = get.Narrow(1, 1, 2)
    assert isinstance(model, torch.nn.Module)

    x = torch.randn(4, 3)
    y = model(x)
    assert y.shape == (4, 2)

    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x).shape == y.shape
    traced_model = torch.jit.trace(model, (x))
    assert traced_model(x).shape == y.shape


def test_index_select():
    model = get.IndexSelect(1, torch.tensor([0, 2]))
    assert isinstance(model, torch.nn.Module)

    x = torch.randn(4, 3)
    y = model(x)
    assert y.shape == (4, 2)

    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x).shape == y.shape
    traced_model = torch.jit.trace(model, (x))
    assert traced_model(x).shape == y.shape
