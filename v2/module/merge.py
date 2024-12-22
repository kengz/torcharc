import torch
from torch import nn


class MergeDim(nn.Module):
    """Merge module along a dimension - base class"""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MergeConcat(MergeDim):
    """Merge module using torch.cat along a dimension"""

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return torch.cat(args, dim=self.dim)


class MergeStack(MergeDim):
    """Merge module using torch.stack along a dimension"""

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return torch.stack(args, dim=self.dim)


class MergeSum(MergeDim):
    """Merge module using torch.sum along a dimension"""

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.stack(args, dim=self.dim), dim=self.dim)


class MergeProd(MergeDim):
    """Merge module using torch.prod along a dimension"""

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return torch.prod(torch.stack(args, dim=self.dim), dim=self.dim)


class MergeMean(MergeDim):
    """Merge module using torch.mean along a dimension"""

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.stack(args, dim=self.dim), dim=self.dim)


class MergeDot(nn.Module):
    """Merge module using dot-product, e.g. similarity matrix for CLIP"""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.T)


# TODO FiLM
# TODO attention?
