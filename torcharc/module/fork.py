from abc import ABC, abstractmethod
from torch import nn
from typing import Dict, List
import pydash as ps
import torch


class Fork(ABC, nn.Module):
    '''A Fork module forks one tensor into a dict of multiple tensors.'''

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict:  # pragma: no cover
        raise NotImplementedError


class ReuseFork(Fork):
    '''Fork layer to reuse a tensor multiple times via ref in dict'''

    def __init__(self, names: List[str]) -> None:
        super().__init__()
        self.names = names
        self.num_reuse = len(names)

    def forward(self, x: torch.Tensor) -> dict:
        return dict(zip(self.names, [x] * self.num_reuse))


class SplitFork(Fork):
    '''Fork layer to split a tensor along dim=1 into multiple tensors. Reverse of ConcatMerge.'''

    def __init__(self, shapes: Dict[str, List[int]]) -> None:
        super().__init__()
        self.shapes = shapes
        self.split_size = ps.flatten(self.shapes.values())

    def forward(self, x: torch.Tensor) -> dict:
        return dict(zip(self.shapes, x.split(self.split_size, dim=1)))
