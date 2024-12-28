import torch
from torch import nn


class Get(nn.Module):
    """
    Get a specific key of an iterable, e.g. (tuple, list, dict) as it can't be split in PyTorch Graph Node
    NOTE For slicing tensor, use Narrow or IndexSelect instead
    """

    def __init__(self, key: int | str = 0):
        super().__init__()
        self.key = key

    def forward(self, input) -> torch.Tensor:
        return input[self.key]


class Narrow(nn.Module):
    """
    Use torch.narrow to 'slice' a tensor:
    Returns a new tensor that is a narrowed version of input tensor. The dimension dim is input from start to start + length. The returned tensor and input tensor share the same underlying storage.
    hint: use Flatten() after slicing to squeeze dimension
    """

    def __init__(self, dim: int, start: int, length: int):
        super().__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.narrow(x, dim=self.dim, start=self.start, length=self.length)


class IndexSelect(nn.Module):
    """
    Use torch.index_select to 'slice' a tensor:
    Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
    The returned tensor has the same number of dimensions as the original tensor (input).
    The returned tensor does not use the same storage as the original tensor.
    hint: use Flatten() after slicing to squeeze dimension
    """

    def __init__(self, dim: int, index: list[int]):
        super().__init__()
        self.dim = dim
        # Convert list to tensor and register as buffer
        self.register_buffer("index", torch.tensor(index, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.index_select(x, dim=self.dim, index=self.index)
