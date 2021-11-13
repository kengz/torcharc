# module to build Sequential module quickly from arc
from torch import nn
from typing import Any, List, Optional
import pydash as ps


class SpreadSequential(nn.Sequential):
    '''Sequential with auto-spread on multiple input arguments since PyTorch Sequential can't handle it'''

    def forward(self, *inputs):
        for module in self:
            inputs = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
        return inputs


def build_nn_layer(nn_type: str, *args, **kwargs) -> nn.Module:
    '''Build the main layer from arc.type, e.g. linear, conv1d, conv2d, conv3d'''
    return getattr(nn, nn_type)(*args, **kwargs)


def build_sub_layer(k: str, v: Any, dim: int = 1) -> Optional[nn.Module]:
    '''Build the sub layer (activation, batch_norm, dropout) defined in arc'''
    if not v:  # if falsy
        return None
    elif k == 'activation':
        return build_nn_layer(v)
    elif k == 'batch_norm':
        return {
            1: nn.BatchNorm1d,
            2: nn.BatchNorm2d,
            3: nn.BatchNorm3d,
        }[dim](v)
    elif k == 'dropout':
        return {
            1: nn.Dropout,
            2: nn.Dropout2d,
            3: nn.Dropout3d,
        }[dim](v)
    else:  # no action for other keys
        return None


def build_layer_group(arc: dict, idx: int) -> List[nn.Module]:
    '''
    Build a layer group consisting of main layer and sub layers.
    The idx follows that of the arc.layers, where idx = 0 consists of (in_shape[0], layers[idx]), otherwise (layers[idx-1], layers[idx]).

    - main layer: linear, conv1d, conv2d, conv3d (rnn doesn't need this method)
    - sub layers: activation, batch_norm, dropout as ordered by their keys in arc
    '''
    layer_group = []
    nn_type, in_shape, layers = ps.at(arc, *['type', 'in_shape', 'layers'])
    if nn_type == 'Linear':
        in_shape = [arc['in_features']]
    dim = int(ps.find(nn_type, str.isdigit) or 1)
    # in_features for linear or in_channels for conv: use in_shape or the last layer (or first of last layer if conv)
    arg_head = in_shape[0] if idx == 0 else ps.to_list(layers[idx - 1])[0]
    arg_tail = layers[idx]  # int for linear, list for conv
    args = ps.to_list(arg_head) + ps.to_list(arg_tail)  # ensure args is a list

    # main layer
    layer_group.append(build_nn_layer(nn_type, *args))
    # sub layers
    for k, v in arc.items():
        if k == 'batch_norm' and v:  # update v as arg if truthy
            v = args[1]  # always the out_size or out_channel
        layer_group.append(build_sub_layer(k, v, dim))
    return ps.compact(layer_group)


def build(arc: dict) -> nn.Sequential:
    '''The main module method to build a Sequential module given arc using build_layer_group'''
    nn_layers = ps.flatten([build_layer_group(arc, idx) for idx in range(len(arc['layers']))])
    return nn.Sequential(*nn_layers)
