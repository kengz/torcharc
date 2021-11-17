from torch import nn
from typing import Dict, List, Union
import pydash as ps
import torch


def build_component(arc: dict, infer_arc: dict, name: str, module):
    '''Helper to build component of module.{name} by combining arc with infer_arc'''
    sub_arc = arc[name]
    kwargs = ps.omit(sub_arc, 'type')
    kwargs.update(infer_arc)
    sub_module = getattr(module, sub_arc['type'])(**kwargs)
    return sub_module


def calc_out_shape(module: nn.Module, in_shape: List[int]) -> List[int]:
    '''Calculate the output shape of a module via forward pass given an input shape'''
    x = get_rand_tensor(in_shape)
    with torch.no_grad():
        y = module(x)
    return list(y.shape[1:])  # exclude batch


def get_layer_names(nn_layers: List[nn.Module]) -> List[str]:
    '''Get the class name of each nn.Module in a list'''
    return [nn_layer._get_name() for nn_layer in nn_layers]


def _get_rand_tensor(shape: Union[list, dict], batch_size: int = 4) -> torch.Tensor:
    '''Get a random tensor given a shape and a batch size'''
    return torch.rand([batch_size] + list(shape))


def get_rand_tensor(shapes: Union[List[int], Dict[str, list]], batch_size: int = 4) -> Union[torch.Tensor, dict]:
    '''Get a random tensor dict with default batch size for a dict of shapes'''
    if isinstance(shapes, dict):
        return {k: _get_rand_tensor(shape, batch_size) for k, shape in shapes.items()}
    else:
        return _get_rand_tensor(shapes, batch_size)
