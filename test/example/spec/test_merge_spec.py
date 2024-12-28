import pytest
import torch

import torcharc

B = 4  # batch size


@pytest.mark.parametrize(
    "spec_file",
    [
        "concat.yaml",
        "film.yaml",
        "mean.yaml",
        "prod.yaml",
        "sum.yaml",
    ],
)
def test_model(spec_file):
    # Build the model using torcharc
    model = torcharc.build(torcharc.SPEC_DIR / "merge" / spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    x_0 = torch.randn(B, 16)
    x_1 = torch.randn(B, 20)
    y = model(x_0, x_1)
    assert y.shape == (B, 1)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(x_0, x_1).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x_0, x_1).shape == y.shape
    traced_model = torch.jit.trace(model, (x_0, x_1))
    assert traced_model(x_0, x_1).shape == y.shape


def test_bmm():
    # Build the model using torcharc
    model = torcharc.build(torcharc.SPEC_DIR / "merge" / "bmm.yaml")
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    m_0 = torch.randn(B, 3, 4)
    m_1 = torch.randn(B, 4, 5)
    y = model(m_0, m_1)
    assert y.shape == (B, 3, 5)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(m_0, m_1).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(m_0, m_1).shape == y.shape
    traced_model = torch.jit.trace(model, (m_0, m_1))
    assert traced_model(m_0, m_1).shape == y.shape


def test_dot():
    # Build the model using torcharc
    model = torcharc.build(torcharc.SPEC_DIR / "merge" / "dot.yaml")
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    x_0 = torch.randn(B, 16)
    x_1 = torch.randn(B, 20)
    y = model(x_0, x_1)
    assert y.shape == (B,)  # dot product reduces to scalar

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(x_0, x_1).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x_0, x_1).shape == y.shape
    traced_model = torch.jit.trace(model, (x_0, x_1))
    assert traced_model(x_0, x_1).shape == y.shape


def test_stack():
    # Build the model using torcharc
    model = torcharc.build(torcharc.SPEC_DIR / "merge" / "stack.yaml")
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    x_0 = torch.randn(B, 16)
    x_1 = torch.randn(B, 20)
    y = model(x_0, x_1)
    assert y.shape == (B, 2, 10)  # (batch, stack, head dim)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(x_0, x_1).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x_0, x_1).shape == y.shape
    traced_model = torch.jit.trace(model, (x_0, x_1))
    assert traced_model(x_0, x_1).shape == y.shape
