from collections import namedtuple
from torch import nn
from typing import Dict, List, NamedTuple, Union
import pydash as ps
import torch


def calc_out_shape(module: nn.Module, in_shape: List[int]) -> List[int]:
    '''Calculate the output shape of a module via forward pass given an input shape'''
    x = get_rand_tensor(in_shape)
    with torch.no_grad():
        y = module(x)
    return list(y.shape[1:])  # exclude batch


def get_layer_names(nn_layers: List[nn.Module]) -> List[str]:
    '''Get the class name of each nn.Module in a list'''
    return [nn_layer._get_name() for nn_layer in nn_layers]


def _get_rand_tensor(shape: Union[list, tuple], batch_size: int = 4) -> torch.Tensor:
    '''Get a random tensor given a shape and a batch size'''
    return torch.rand([batch_size] + list(shape))


def get_rand_tensor(shapes: Union[List[int], Dict[str, list]], batch_size: int = 4) -> Union[torch.Tensor, NamedTuple]:
    '''Get a random tensor tuple with default batch size for a dict of shapes'''
    if ps.is_dict(shapes):
        TensorTuple = namedtuple('TensorTuple', shapes.keys())
        return TensorTuple(*[_get_rand_tensor(shape, batch_size) for shape in shapes.values()])
    else:
        return _get_rand_tensor(shapes, batch_size)


def to_namedtuple(data: dict, name='NamedTensor') -> NamedTuple:
    '''Convert a dictionary to namedtuple.'''
    return namedtuple(name, data)(**data)
