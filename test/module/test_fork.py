import pytest
import torch

from torcharc.module import fork


@pytest.mark.parametrize(
    "chunks",
    [
        2,
        3,
    ],
)
def test_fork_chunk(chunks):
    model = fork.ForkChunk(chunks, dim=1)
    assert isinstance(model, torch.nn.Module)

    x = torch.randn(1, 32)
    y = model(x)
    assert len(y) == chunks

    compiled_model = torch.compile(model)
    assert len(compiled_model(x)) == chunks
    traced_model = torch.jit.trace(model, (x))
    assert len(traced_model(x)) == chunks


@pytest.mark.parametrize(
    "split_size_or_sections",
    [
        5,  # each split has size 5
        [6, 4],
    ],
)
def test_fork_split(split_size_or_sections):
    model = fork.ForkSplit(split_size_or_sections, dim=1)
    assert isinstance(model, torch.nn.Module)

    x = torch.randn(1, 10)
    y = model(x)
    assert len(y) == 2

    compiled_model = torch.compile(model)
    assert len(compiled_model(x)) == 2
    traced_model = torch.jit.trace(model, (x))
    assert len(traced_model(x)) == 2
