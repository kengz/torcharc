import pytest
import torch
from conftest import SPEC_DIR

import torcharc

B = 4  # batch size


@pytest.mark.parametrize(
    "spec_file, input_shape, output_shape",
    [
        (SPEC_DIR / "basic" / "mlp.yaml", (B, 128), (B, 10)),
        (SPEC_DIR / "basic" / "mlp_lazy.yaml", (B, 128), (B, 10)),
        (SPEC_DIR / "basic" / "conv.yaml", (B, 3, 32, 32), (B, 10)),
        (SPEC_DIR / "basic" / "conv_lazy.yaml", (B, 3, 32, 32), (B, 10)),
    ],
)
def test_model(spec_file, input_shape, output_shape):
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    x = torch.randn(*input_shape)
    y = model(x)
    assert y.shape == output_shape

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(x).shape == y.shape
    traced_model = torch.jit.trace(model, (x))
    assert traced_model(x).shape == y.shape


@pytest.mark.parametrize(
    "spec_file, input_shape, output_shape",
    [
        # RNN input is (batch, seq_len, input_size)
        (SPEC_DIR / "basic" / "rnn.yaml", (B, 10, 7), (B, 10)),
    ],
)
def test_rnn(spec_file, input_shape, output_shape):
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    x = torch.randn(*input_shape)
    y = model(x)
    assert y.shape == output_shape

    # Test compatibility with compile
    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
