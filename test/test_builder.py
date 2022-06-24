from torch import nn, optim
from torcharc import arc_ref, build, build_criterion, build_optimizer
import pytest


@pytest.mark.parametrize('name,arc', list(arc_ref.REF_ARCS.items()))
def test_builder(name, arc):
    print('building', name)
    model = build(arc)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize('loss_spec', [
    {'type': 'MSELoss'},
    {'type': 'BCEWithLogitsLoss', 'reduction': 'mean', 'pos_weight': 10.0},  # with numeric arg to be converted to tensor
])
def test_build_criterion(loss_spec):
    criterion = build_criterion(loss_spec)
    assert isinstance(criterion, nn.Module)


@pytest.mark.parametrize('optim_spec', [
    {'type': 'SGD', 'lr': 0.1},
    {'type': 'Adam', 'lr': 0.001},
])
def test_build_optimizer(optim_spec):
    model = build(arc_ref.REF_ARCS['Linear'])
    criterion = build_optimizer(optim_spec, model)
    assert isinstance(criterion, optim.Optimizer)
