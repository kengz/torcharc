from torcharc.module.fork import Fork, ReuseFork, SplitFork
import pydash as ps
import pytest
import torch


@pytest.mark.parametrize('names,x', [
    (
        ['reuse_0', 'reuse_1'],
        torch.rand(4, 3),
    )
])
def test_reuse_fork(names, x):
    fork = ReuseFork(names)
    assert isinstance(fork, Fork)
    ys = fork(x)
    assert ps.is_tuple(ys)
    assert ys._fields == tuple(names)


@pytest.mark.parametrize('shapes,x', [
    (
        {'mean': [3], 'std': [3]},
        torch.rand(4, 3 + 3),
    )
])
def test_split_fork(shapes, x):
    fork = SplitFork(shapes)
    assert isinstance(fork, Fork)
    ys = fork(x)
    assert ps.is_tuple(ys)
    assert ys._fields == tuple(shapes.keys())
