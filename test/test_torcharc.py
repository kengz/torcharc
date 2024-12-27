import pytest
import torch
from conftest import SPEC_DIR

import torcharc


@pytest.mark.parametrize("spec_file", list(SPEC_DIR.rglob("*.yaml")))
def test_build_compile(spec_file):
    # Build the model using torcharc
    model = torcharc.build(spec_file)
    assert isinstance(model, torch.nn.Module)
    # Test compatibility with torch.compile
    torch.compile(model)
