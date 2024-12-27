import torch
import pytest
import yaml
import torcharc
from pathlib import Path

SPEC_DIR = Path("torcharc/example/spec")
B = 4  # batch size


@pytest.mark.parametrize(
    "spec_file, input_shape, output_shape",
    [
        (SPEC_DIR / "mlp.yaml", (B, 128), (B, 10)),
        (SPEC_DIR / "mlp_lazy.yaml", (B, 128), (B, 10)),
        (SPEC_DIR / "conv.yaml", (B, 3, 32, 32), (B, 10)),
        (SPEC_DIR / "conv_lazy.yaml", (B, 3, 32, 32), (B, 10)),
    ],
)
def test_basic_model(spec_file, input_shape, output_shape):
    # Load the model specification from the YAML file
    with open(spec_file, "r") as f:
        spec = yaml.safe_load(f)

    # Build the model using torcharc
    model = torcharc.build(spec)

    # Run the model and check the output shape
    x = torch.randn(*input_shape)
    y = model(x)
    assert y.shape == output_shape

    # Test compatibility with JIT script
    scripted_model = torch.jit.script(model)
    assert scripted_model(x).shape == output_shape

    # Test compatibility with JIT trace
    traced_model = torch.jit.trace(model, (x,))
    assert traced_model(x).shape == output_shape

    # Test compatibility with torch.compile
    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == output_shape


@pytest.mark.parametrize(
    "spec_file, input_shape, output_shape",
    [
        # RNN input is (batch, seq_len, input_size)
        (SPEC_DIR / "rnn.yaml", (B, 10, 7), (B, 10)),
    ],
)
def test_rnn(spec_file, input_shape, output_shape):
    # Load the model specification from the YAML file
    with open(spec_file, "r") as f:
        spec = yaml.safe_load(f)

    # Build the model using torcharc
    model = torcharc.build(spec)

    # Run the model and check the output shape
    x = torch.randn(*input_shape)
    y = model(x)
    assert y.shape == output_shape

    # Test compatibility with torch.compile
    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == output_shape


# @pytest.mark.parametrize(
#     "spec_file, input_shape, output_shape",
#     [
#         (SPEC_DIR / "attention/attn.yaml", (B, 128), (B, 10)),

#     ],
# )
