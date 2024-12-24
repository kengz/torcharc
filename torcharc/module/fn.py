import torch
from torch import nn


class TorchFn(nn.Module):
    """
    Call arbitrary torch function with arguments
    """

    def __init__(self, name: str, **kwargs):
        super().__init__()
        self.fn = getattr(torch, name)
        self.kwargs = kwargs

    def forward(self, x) -> torch.Tensor:
        return self.fn(x, **self.kwargs)
