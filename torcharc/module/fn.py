import torch
from torch import nn


class Reduce(nn.Module):
    """
    Call torch reduction ops https://pytorch.org/docs/stable/torch.html#reduction-ops with common arguments `dim` and `keepdim` - this covers most but not all functions. For general functions, see TorchFn.
    This is compatible with torch compile, JIT script and JIT trace by using static arguments for performance instead of dynamic *args and **kwargs
    """

    def __init__(self, name: str, dim: int | list[int] | None, keepdim: bool = False):
        super().__init__()
        self.fn = getattr(torch, name)
        self.dim = dim  # all reduction ops use dim as int or tuple[int]
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
        # NOTE most iterables values aren't tensors; if needed like in the case of torch.index_select, implement and register a separate torch.nn.Module, like get.IndexSelect

    def forward(self, input) -> torch.Tensor:
        return self.fn(input, **self.kwargs)
