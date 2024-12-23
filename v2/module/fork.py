import torch
from torch import nn


class ForkChunk(nn.Module):
    """Fork a tensor using torch.chunk along a dimension"""

    def __init__(self, chunks: int = 2, dim: int = 1):
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.chunk(x, chunks=self.chunks, dim=self.dim)


class ForkSplit(nn.Module):
    """Fork a tensor using torch.split along a dimension"""

    def __init__(self, split_size_or_sections: int | list[int], dim: int = 1):
        super().__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.split(x, self.split_size_or_sections, dim=self.dim)
