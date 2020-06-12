from torcharc import optim
from torch import nn
import pytest
import torch


def test_optim_registration():
    assert torch.optim.GlobalAdam == optim.GlobalAdam
    assert torch.optim.GlobalRMSprop == optim.GlobalRMSprop
    assert torch.optim.Lookahead == optim.Lookahead
    assert torch.optim.RAdam == optim.RAdam


@pytest.mark.parametrize('optim_name', [
    'GlobalAdam',
    'GlobalRMSprop',
    'Lookahead',
    'RAdam',
])
def test_optim(optim_name):
    layer = nn.Linear(8, 4)
    params = layer.parameters()
    optim = getattr(torch.optim, optim_name)(params)
    optim.share_memory()
    # test autograd core functions
    layer(torch.rand(1, 8)).mean().backward()
    optim.step()
    assert True
