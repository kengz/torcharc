import pytest
import torch

import torcharc

B = 4  # batch size


@pytest.mark.parametrize(
    "spec_file, input_shape, output_shape",
    [
        ("mlp.yaml", (B, 128), (B, 16)),
        ("mlp_classifier.yaml", (B, 128), (B, 10)),
        ("conv.yaml", (B, 3, 32, 32), (B, 64, 26, 26)),
        ("conv_classifier.yaml", (B, 3, 32, 32), (B, 10)),
    ],
)
def test_model(spec_file, input_shape, output_shape):
    # Build the model using torcharc
    model = torcharc.build(torcharc.SPEC_DIR / "compact" / spec_file)
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
