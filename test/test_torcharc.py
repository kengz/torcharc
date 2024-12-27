import pytest
import torch
import yaml
from conftest import SPEC_DIR

import torcharc


@pytest.mark.parametrize("spec_file", list(SPEC_DIR.rglob("*.yaml")))
def test_build_compile(spec_file):
    # test build and compat with torch.compile
    # Load the model specification from the YAML file
    with open(spec_file, "r") as f:
        spec = yaml.safe_load(f)
    # Build the model using torcharc
    model = torcharc.build(spec)
    # Test compatibility with torch.compile
    torch.compile(model)
