import torch

import torcharc

B = 4  # batch size


def test_get():
    spec_file = torcharc.SPEC_DIR / "get" / "get.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    x = torch.randn(B, 32)
    y = model(x)
    assert len(y) == 2  # tail_0, tail_1

    # Test compatibility with compile and trace
    compiled_model = torch.compile(model)
    assert len(compiled_model(x)) == 2
    traced_model = torch.jit.trace(model, (x))
    assert len(traced_model(x)) == 2


def test_index_select():
    spec_file = torcharc.SPEC_DIR / "get" / "index_select.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    x = torch.randn(B, 32)
    y = model(x)
    assert y.shape == (B, 3)  # select 3 elements

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x).shape == y.shape
    traced_model = torch.jit.trace(model, (x))
    assert traced_model(x).shape == y.shape


def test_narrow():
    spec_file = torcharc.SPEC_DIR / "get" / "narrow.yaml"

    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    x = torch.randn(B, 10, 7)
    y = model(x)
    assert y.shape == (B, 10)

    # Test compatibility with compile
    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
