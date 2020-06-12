from torcharc import arc_ref, build
from torch import nn


def test_builder():
    for name, arc in arc_ref.REF_ARCS.items():
        print('building', name)
        model = build(arc)
        assert isinstance(model, nn.Module)
