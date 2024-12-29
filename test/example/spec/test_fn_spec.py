import torch

import torcharc

B = 4  # batch size


# general TorchFn no compat with JIT script
def test_fn_topk():
    spec_file = torcharc.SPEC_DIR / "fn" / "fn_topk.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    x = torch.randn(B, 128)
    y = model(x)
    assert len(y) == 2  # values and indices

    # Test compatibility with compile and trace
    compiled_model = torch.compile(model)
    assert len(compiled_model(x)) == 2
    traced_model = torch.jit.trace(model, (x))
    assert len(traced_model(x)) == 2


def test_reduce_mean():
    spec_file = torcharc.SPEC_DIR / "fn" / "reduce_mean.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    text = torch.randint(0, 1000, (B, 10))
    y = model(text)
    assert y.shape == (B, 128)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(text).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(text).shape == y.shape
    traced_model = torch.jit.trace(model, (text))
    assert traced_model(text).shape == y.shape

