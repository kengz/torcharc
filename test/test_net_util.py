from fixture.net import CONV1D_ARC, CONV2D_ARC, CONV3D_ARC, LINEAR_ARC
from torcharc import module_builder, net_util
from torch import nn
import pydash as ps
import pytest
import torch


@pytest.mark.parametrize('arc', [
    CONV1D_ARC, CONV2D_ARC, CONV3D_ARC, LINEAR_ARC
])
def test_calc_out_shape(arc):
    in_shape = [arc['in_features']] if arc['type'] == 'Linear' else arc['in_shape']
    module = module_builder.build_module(arc)
    module_out_shape = net_util.calc_out_shape(module, in_shape)
    assert len(module_out_shape) == len(in_shape)


def test_get_layer_names():
    assert net_util.get_layer_names([nn.Linear(4, 2), nn.ReLU()]) == ['Linear', 'ReLU']


@pytest.mark.parametrize('shape,batch_size,tensor_shape', [
    ([8], 4, [4, 8]),
    ([8, 6], 4, [4, 8, 6]),
    ((8, 6), 4, [4, 8, 6]),
    (torch.Size([8, 6]), 4, [4, 8, 6]),
    ([8, 6], 32, [32, 8, 6]),
])
def test_get_rand_tensor(shape, batch_size, tensor_shape):
    xs = net_util.get_rand_tensor(shape, batch_size)
    assert isinstance(xs, torch.Tensor)
    assert list(xs.shape) == tensor_shape


@pytest.mark.parametrize('shapes,batch_size,tensor_shapes', [
    ({'vector': [8]}, 4, {'vector': [4, 8]}),
    ({'image': [3, 20, 20], 'vector': [8]}, 4, {'image': [4, 3, 20, 20], 'vector': [4, 8]}),
])
def test_get_rand_tensor_dict(shapes, batch_size, tensor_shapes):
    xs = net_util.get_rand_tensor(shapes, batch_size)
    assert ps.is_tuple(xs)
    for name, tensor_shape in tensor_shapes.items():
        x = getattr(xs, name)
        assert list(x.shape) == tensor_shape
