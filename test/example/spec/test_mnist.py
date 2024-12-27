import pytest
from conftest import SPEC_DIR

from test.example.spec.test_basic import test_model

B = 4  # batch size


@pytest.mark.parametrize(
    "spec_file, input_shape, output_shape",
    [
        (SPEC_DIR / "mnist" / "conv.yaml", (B, 3, 32, 32), (B, 10)),
        (SPEC_DIR / "mnist" / "mlp.yaml", (B, 3, 32, 32), (B, 10)),
    ],
)
def test_mnist(spec_file, input_shape, output_shape):
    test_model(spec_file, input_shape, output_shape)
