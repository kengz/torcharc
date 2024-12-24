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
        if isinstance(split_size_or_sections, int):
            self.split_size_or_sections = split_size_or_sections
        else:  # Convert list to tensor and register as buffer
            self.register_buffer('split_size_or_sections', torch.tensor(split_size_or_sections, dtype=torch.long))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.split(x, self.split_size_or_sections, dim=self.dim)
