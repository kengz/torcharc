import pytest
import torch

from torcharc.module import merge


def test_merge_concat():
    model = merge.MergeConcat()
    assert isinstance(model, torch.nn.Module)

    x_0 = torch.randn(4, 2)
    x_1 = torch.randn(4, 3)
    y = model((x_0, x_1))
    assert y.shape == (4, 5)

    compiled_model = torch.compile(model)
    assert compiled_model((x_0, x_1)).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model((x_0, x_1)).shape == y.shape
    traced_model = torch.jit.trace(model, ((x_0, x_1),))
    assert traced_model((x_0, x_1)).shape == y.shape


def test_merge_stack():
    model = merge.MergeStack()
    assert isinstance(model, torch.nn.Module)

    x_0 = torch.randn(4, 3)
    x_1 = torch.randn(4, 3)
    y = model((x_0, x_1))
    assert y.shape == (4, 2, 3)

    compiled_model = torch.compile(model)
    assert compiled_model((x_0, x_1)).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model((x_0, x_1)).shape == y.shape
    traced_model = torch.jit.trace(model, ((x_0, x_1),))
    assert traced_model((x_0, x_1)).shape == y.shape


@pytest.mark.parametrize(
    "model",
    [
        merge.MergeSum(),
        merge.MergeMean(),
        merge.MergeProd(),
    ],
)
def test_merge_sum(model):
    assert isinstance(model, torch.nn.Module)

    x_0 = torch.randn(4, 3)
    x_1 = torch.randn(4, 3)
    y = model((x_0, x_1))
    assert y.shape == (4, 3)

    compiled_model = torch.compile(model)
    assert compiled_model((x_0, x_1)).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model((x_0, x_1)).shape == y.shape
    traced_model = torch.jit.trace(model, ((x_0, x_1),))
    assert traced_model((x_0, x_1)).shape == y.shape


def test_merge_dot():
    model = merge.MergeDot()
    assert isinstance(model, torch.nn.Module)

    x_0 = torch.randn(4, 3)
    x_1 = torch.randn(4, 3)
    y = model(x_0, x_1)
    assert y.shape == (4,)

    compiled_model = torch.compile(model)
    assert compiled_model(x_0, x_1).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x_0, x_1).shape == y.shape
    traced_model = torch.jit.trace(
        model,
        (
            x_0,
            x_1,
        ),
    )
    assert traced_model(x_0, x_1).shape == y.shape


def test_merge_bmm():
    model = merge.MergeBMM()
    assert isinstance(model, torch.nn.Module)

    m_0 = torch.randn(4, 3, 4)
    m_1 = torch.randn(4, 4, 5)
    y = model(m_0, m_1)
    assert y.shape == (4, 3, 5)

    compiled_model = torch.compile(model)
    assert compiled_model(m_0, m_1).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(m_0, m_1).shape == y.shape
    traced_model = torch.jit.trace(
        model,
        (
            m_0,
            m_1,
        ),
    )
    assert traced_model(m_0, m_1).shape == y.shape


@pytest.mark.parametrize(
    "feature_dim, conditioner_dim, feature, conditioner",
    [
        (3, 8, torch.rand(4, 3), torch.rand(4, 8)),  # vector-vector
        (3, 8, torch.rand(4, 3, 32, 32), torch.rand(4, 8)),  # conv-vector
    ],
)
def test_merge_film(feature_dim, conditioner_dim, feature, conditioner):
    model = merge.MergeFiLM(feature_dim, conditioner_dim)
    assert isinstance(model, torch.nn.Module)

    y = model(feature, conditioner)
    assert y.shape == feature.shape

    compiled_model = torch.compile(model)
    assert compiled_model(feature, conditioner).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(feature, conditioner).shape == y.shape
    traced_model = torch.jit.trace(
        model,
        (
            feature,
            conditioner,
        ),
    )
    assert traced_model(feature, conditioner).shape == y.shape
