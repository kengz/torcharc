import torch
from torch import nn


class Reduce(nn.Module):
    """
    Call torch reduction ops https://pytorch.org/docs/stable/torch.html#reduction-ops with common arguments `dim` and `keepdim` (so, applicable to any torch function)
    Uses static arguments instead of dynamic *args and **kwargs to preserve compatibility with JIT script and trace for performance
    """

    def __init__(self, name: str, dim: int | list[int] | None, keepdim: bool = False):
        super().__init__()
        self.fn = getattr(torch, name)
        if isinstance(dim, list):  # Convert list to tensor and register as buffer
            self.register_buffer("dim", torch.tensor(dim, dtype=torch.long))
        else:
            self.dim = dim
        self.keepdim = keepdim

    def forward(self, input) -> torch.Tensor:
        return self.fn(input, dim=self.dim, keepdim=self.keepdim)


class TorchFn(nn.Module):
    """
    Call torch functions generically with tensors as first input and other arguments as kwargs stored during init.
    While torch compile and JIT trace still work here, the caveat is incompatibility with JIT script from the dynamic **kwargs
    """

    def __init__(self, name: str, **kwargs: dict):
        super().__init__()
        self.fn = getattr(torch, name)
        self.kwargs = kwargs or {}
        # basic guard - convert iterable values to tensors
        for key, value in self.kwargs.items():
            if isinstance(value, (list, tuple)):
                self.kwargs[key] = torch.tensor(value)

    def forward(self, input) -> torch.Tensor:
        return self.fn(input, **self.kwargs)
