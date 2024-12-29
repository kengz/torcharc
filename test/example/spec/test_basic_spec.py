import pytest
import torch

import torcharc

B = 4  # batch size


@pytest.mark.parametrize(
    "spec_file, input_shape, output_shape",
    [
        ("mlp.yaml", (B, 128), (B, 10)),
        ("mlp_lazy.yaml", (B, 128), (B, 10)),
        ("conv.yaml", (B, 3, 32, 32), (B, 10)),
        ("conv_lazy.yaml", (B, 3, 32, 32), (B, 10)),
    ],
)
def test_model(spec_file, input_shape, output_shape):
    # Build the model using torcharc
    model = torcharc.build(torcharc.SPEC_DIR / "basic" / spec_file)
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


def test_stereo_conv():
    # Build the model using torcharc
    spec_file = torcharc.SPEC_DIR / "basic" / "stereo_conv_reuse.yaml"
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    left_image = right_image = torch.randn(B, 3, 32, 32)
    left, right = model(left_image=left_image, right_image=right_image)
    assert left.shape == (B, 10)
    assert right.shape == (B, 10)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert len(compiled_model(left_image, right_image)) == 2
    scripted_model = torch.jit.script(model)
    assert len(scripted_model(left_image, right_image)) == 2
    traced_model = torch.jit.trace(model, (left_image, right_image))
    assert len(traced_model(left_image, right_image)) == 2


@pytest.mark.parametrize(
    "spec_file, input_shape, output_shape",
    [
        # RNN input is (batch, seq_len, input_size)
        ("rnn.yaml", (B, 10, 7), (B, 10)),
    ],
)
def test_rnn(spec_file, input_shape, output_shape):
    # Build the model using torcharc
    model = torcharc.build(torcharc.SPEC_DIR / "basic" / spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    x = torch.randn(*input_shape)
    y = model(x)
    assert y.shape == output_shape

    # Test compatibility with compile
    compiled_model = torch.compile(model)
    assert compiled_model(x).shape == y.shape
