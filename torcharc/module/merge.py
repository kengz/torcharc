from abc import ABC, abstractmethod
from torch import nn
from typing import Dict, List, NamedTuple
import torch


class Merge(ABC, nn.Module):
    '''A Merge module merges a dict of tensors into one tensor'''

    @abstractmethod
    def forward(self, xs: NamedTuple) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class ConcatMerge(Merge):
    '''Merge layer to merge a dict of tensors by concatenating along dim=1. Reverse of Split'''

    def forward(self, xs: NamedTuple) -> torch.Tensor:
        return torch.cat(xs, dim=1)


class FiLMMerge(Merge):
    '''
    Merge layer to merge a dict of 2 tensors by Feature-wise Linear Modulation layer https://distill.pub/2018/feature-wise-transformations/
    Takes a feature tensor and conditioning tensor and affine-transforms it with a conditioning tensor:
    output = conditioner_scale * feature + conditioner_shift
    The conditioning tensor is a vector, and will be passed through a Linear layer with out_features = number of features or channels (image), and the operation is element-wise on the features or channels.
    '''

    def __init__(self, names: Dict[str, str], shapes: Dict[str, List[int]]) -> None:
        super().__init__()
        self.feature_name = names['feature']
        self.conditioner_name = names['conditioner']
        assert len(shapes) == 2, f'shapes {shapes} should specify only two keys for feature and conditioner'
        self.feature_size = shapes[self.feature_name][0]
        self.conditioner_size = shapes[self.conditioner_name][0]
        self.conditioner_scale = nn.Linear(self.conditioner_size, self.feature_size)
        self.conditioner_shift = nn.Linear(self.conditioner_size, self.feature_size)

    @classmethod
    def affine_transform(cls, feature: torch.Tensor, conditioner_scale: torch.Tensor, conditioner_shift: torch.Tensor) -> torch.Tensor:
        '''Apply affine transform with safe-broadcast across the entire features/channels of the feature tensor'''
        view_shape = list(conditioner_scale.shape) + [1] * (feature.dim() - conditioner_scale.dim())
        return conditioner_scale.view(*view_shape) * feature + conditioner_shift.view(*view_shape)

    def forward(self, xs: NamedTuple) -> torch.Tensor:
        '''Apply FiLM affine transform on feature using conditioner'''
        feature = getattr(xs, self.feature_name)
        conditioner = getattr(xs, self.conditioner_name)
        conditioner_scale = self.conditioner_scale(conditioner)
        conditioner_shift = self.conditioner_shift(conditioner)
        return self.affine_transform(feature, conditioner_scale, conditioner_shift)
