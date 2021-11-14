from torch import nn
from torcharc import arc_ref, build
import pytest


@pytest.mark.parametrize('name,arc', list(arc_ref.REF_ARCS.items()))
def test_builder(name, arc):
    print('building', name)
    model = build(arc)
    assert isinstance(model, nn.Module)
