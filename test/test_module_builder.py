from fixture.net import CONV1D_ARC, CONV2D_ARC, CONV3D_ARC, LINEAR_ARC
from torcharc import module_builder, net_util
from torch import nn
import pydash as ps
import pytest
import torch


@pytest.mark.parametrize('init', [
    None,
    'uniform_',
    'normal_',
    'ones_',
    'zeros_',
    'xavier_uniform_',
    'xavier_normal_',
    'kaiming_uniform_',
    'kaiming_normal_',
    'orthogonal_',
])
@pytest.mark.parametrize('activation', [
    None,
    'ReLU',
    'LeakyReLU',
    'Sigmoid',
    'Tanh',
])
def test_get_init_fn(init, activation):
    init_fn = module_builder.get_init_fn(init, activation)
    assert ps.is_function(init_fn)


@pytest.mark.parametrize('arc,nn_class', [
    (
        CONV1D_ARC,
        nn.Sequential,
    ), (
        CONV2D_ARC,
        nn.Sequential,
    ), (
        CONV3D_ARC,
        nn.Sequential,
    ), (
        LINEAR_ARC,
        nn.Sequential,
    ), (
        {
            'type': 'Linear',
            'in_features': 8,
            'out_features': 4,
        },
        nn.Linear,
    ), (
        {
            'type': 'Flatten',
        },
        nn.Flatten,
    ), (
        {
            'type': 'SplitFork',
            'shapes': {'mean': [2], 'std': [2]}
        },
        nn.SplitFork,
    ), (
        {
            'type': 'FiLMMerge',
            'in_names': ['image', 'vector'],
            'names': {'feature': 'image', 'conditioner': 'vector'},
            'shapes': {'image': [16, 6, 6], 'vector': [8]}
        },
        nn.FiLMMerge,
    ), (
        {
            'type': 'ConcatMerge',
        },
        nn.ConcatMerge,
    ),
])
def test_build_module(arc, nn_class):
    assert module_builder.build_module(arc).__class__ == nn_class


@pytest.mark.parametrize('init', [
    'uniform_',
    'normal_',
    'xavier_uniform_',
    'xavier_normal_',
    'kaiming_uniform_',
    'kaiming_normal_',
    'orthogonal_',
    {'type': 'normal_', 'std': 0.01},  # when using init kwargs
])
@pytest.mark.parametrize('activation', [
    'ReLU',
    'LeakyReLU',
    'Sigmoid',
    'Tanh',
])
@pytest.mark.parametrize('arc', [
    CONV2D_ARC,
    LINEAR_ARC,
])
def test_init_module(init, activation, arc):
    arc = arc.copy()
    arc['init'] = init
    arc['activation'] = activation
    module = module_builder.build_module(arc)
    pre_params = next(module.parameters()).clone()
    module.apply(module_builder.get_init_fn(init))
    post_params = next(module.parameters())
    assert not torch.eq(pre_params, post_params).all()


@pytest.mark.parametrize('arc,xs,in_shape_key,in_shape', [
    (
        {
            'type': 'Conv2d',
            'layers': [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            'batch_norm': True,
            'activation': 'ReLU',
            'dropout': 0.2,
        },
        net_util.get_rand_tensor({'image': [3, 20, 20], 'vector': [8]}),
        'in_shape',
        [3, 20, 20],
    ), (
        {
            'type': 'Linear',
            'layers': [64, 32],
            'batch_norm': True,
            'activation': 'ReLU',
            'dropout': 0.2,
        },
        net_util.get_rand_tensor({'vector': [8], 'image': [3, 20, 20]}),
        'in_features',
        8,
    )
])
def test_infer_in_shape_default(arc, xs, in_shape_key, in_shape):
    assert in_shape_key not in arc
    module_builder.infer_in_shape(arc, xs)
    assert in_shape_key in arc
    assert arc[in_shape_key] == in_shape


@pytest.mark.parametrize('xs', [
    net_util.get_rand_tensor({'image': [3, 20, 20], 'vector': [8]}),
])
@pytest.mark.parametrize('arc,in_shape_key,in_shape', [
    (
        {
            'type': 'Conv2d',
            'layers': [
                [16, 4, 2, 0, 1],
                [16, 4, 1, 0, 1]
            ],
            'batch_norm': True,
            'activation': 'ReLU',
            'dropout': 0.2,
        },
        'in_shape',
        [3, 20, 20],
    ), (
        {
            'type': 'Linear',
            'in_names': ['vector'],
            'layers': [64, 32],
            'batch_norm': True,
            'activation': 'ReLU',
            'dropout': 0.2,
        },
        'in_features',
        8,
    ), (
        {
            'type': 'Linear',
            'in_names': ['vector'],
            'out_features': 4,
        },
        'in_features',
        8,
    ), (
        {
            'type': 'Flatten',
        },
        None,
        None,
    ), (
        {
            'type': 'SplitFork',
            'shapes': {'mean': [2], 'std': [2]}
        },
        None,
        None,
    ), (
        {
            'type': 'FiLMMerge',
            'in_names': ['image', 'vector'],
            'names': {'feature': 'image', 'conditioner': 'vector'},
        },
        'shapes',
        {'image': [3, 20, 20], 'vector': [8]},
    ), (
        {
            'type': 'ConcatMerge',
        },
        None,
        None,
    ),
])
def test_infer_in_shape(arc, xs, in_shape_key, in_shape):
    pre_keys = list(arc.keys())
    module_builder.infer_in_shape(arc, xs)
    if in_shape_key is not None:
        assert list(arc.keys()) != pre_keys
        assert arc[in_shape_key] == in_shape
    else:
        assert list(arc.keys()) == pre_keys


@pytest.mark.parametrize('arc,xs', [
    (
        CONV2D_ARC,
        net_util.get_rand_tensor(CONV2D_ARC['in_shape']),
    ),
    (
        LINEAR_ARC,
        net_util.get_rand_tensor([LINEAR_ARC['in_features']]),
    )
])
def test_carry_forward_tensor(arc, xs):
    module = module_builder.build_module(arc)
    assert isinstance(xs, torch.Tensor)
    ys = module_builder.carry_forward(module, xs)
    assert isinstance(ys, torch.Tensor)


@pytest.mark.parametrize('arc,xs', [
    (
        CONV2D_ARC,
        net_util.get_rand_tensor({'image': CONV2D_ARC['in_shape'], 'vector': [LINEAR_ARC['in_features']]}),
    ),
    (
        LINEAR_ARC,
        net_util.get_rand_tensor({'vector': [LINEAR_ARC['in_features']], 'image': CONV2D_ARC['in_shape']}),
    )
])
def test_carry_forward_tensor_tuple_default(arc, xs):
    module = module_builder.build_module(arc)
    assert ps.is_tuple(xs)
    ys = module_builder.carry_forward(module, xs)
    assert ps.is_tuple(ys)


@pytest.mark.parametrize('xs', [
    net_util.get_rand_tensor({'image': CONV2D_ARC['in_shape'], 'vector': [LINEAR_ARC['in_features']]}),
])
@pytest.mark.parametrize('arc,in_names', [
    (
        CONV2D_ARC,
        ['image'],
    ), (
        LINEAR_ARC,
        ['vector'],
    ), (
        {
            'type': 'SplitFork',
            'shapes': {'mean': [4], 'std': [4]}
        },
        ['vector'],
    ), (
        {
            'type': 'FiLMMerge',
            'in_names': ['image', 'vector'],
            'names': {'feature': 'image', 'conditioner': 'vector'},
            'shapes': {'image': [3, 20, 20], 'vector': [8]}
        },
        ['image', 'vector'],
    )
])
def test_carry_forward_tensor_tuple(arc, xs, in_names):
    module = module_builder.build_module(arc)
    assert ps.is_tuple(xs)
    ys = module_builder.carry_forward(module, xs, in_names)
    assert isinstance(ys, (torch.Tensor, tuple))
