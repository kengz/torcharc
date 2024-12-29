import pytest
import torch
import torcharc


B = 4  # batch size


@pytest.mark.parametrize(
    "spec_file",
    [
        "dlrm_attn.yaml",
        "dlrm_sum.yaml",
    ],
)
def test_dlrm(spec_file):
    # Build the model using torcharc
    model = torcharc.build(torcharc.SPEC_DIR / "advanced" / spec_file)
    assert isinstance(model, torch.nn.Module)

    # Run the model and check the output shape
    dense = torch.randn(B, 256)
    cat_0 = torch.randint(0, 1000, (B,))
    cat_1 = torch.randint(0, 1000, (B,))
    cat_2 = torch.randint(0, 1000, (B,))
    y = model(dense, cat_0, cat_1, cat_2)
    assert y.shape == (B, 1)

    # Test compatibility with compile
    compiled_model = torch.compile(model)
    assert compiled_model(dense, cat_0, cat_1, cat_2).shape == y.shape


def test_film_image_state():
    spec_file = torcharc.SPEC_DIR / "advanced" / "film_image_state.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    image = torch.randn(B, 3, 32, 32)
    state = torch.randn(B, 4)
    y = model(image=image, state=state)
    assert y.shape == (B, 10)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(image, state).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(image, state).shape == y.shape
    traced_model = torch.jit.trace(model, (image, state))
    assert traced_model(image, state).shape == y.shape


def test_film_image_text():
    spec_file = torcharc.SPEC_DIR / "advanced" / "film_image_text.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    image = torch.randn(B, 3, 32, 32)
    text = torch.randint(0, 1000, (B, 10))
    y = model(image=image, text=text)
    assert y.shape == (B, 10)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(image, text).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(image, text).shape == y.shape
    traced_model = torch.jit.trace(model, (image, text))
    assert traced_model(image, text).shape == y.shape


def test_stereo_conv():
    spec_file = torcharc.SPEC_DIR / "advanced" / "stereo_conv.yaml"
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)

    left_image = right_image = torch.randn(B, 3, 32, 32)
    y = model(left_image=left_image, right_image=right_image)
    assert y.shape == (B, 10)

    # Test compatibility with compile, script and trace
    compiled_model = torch.compile(model)
    assert compiled_model(left_image, right_image).shape == y.shape
    scripted_model = torch.jit.script(model)
    assert scripted_model(left_image, right_image).shape == y.shape
    traced_model = torch.jit.trace(model, (left_image, right_image))
    assert traced_model(left_image, right_image).shape == y.shape
