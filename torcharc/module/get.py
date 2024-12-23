# Module to get specific key of module multi-output (tuple/list/dict) as it can't be split in PyTorch Graph Node
from typing import Iterable

import torch
from torch import nn


class Get(nn.Module):
    """Get specific key of multi-output module"""

    def __init__(self, key: int | str = 0):
        super().__init__()
        self.key = key

    def forward(self, x: Iterable) -> torch.Tensor:
        return x[self.key]
