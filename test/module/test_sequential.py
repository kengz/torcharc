from fixture.net import CONV1D_ARC, CONV2D_ARC, CONV3D_ARC, LINEAR_ARC
from torcharc import net_util
from torcharc.module import sequential
from torch import nn
import pytest


@pytest.mark.parametrize('nn_type,nn_class,args,kwargs', [
    ('Linear', nn.Linear, [4, 32], {}),
    ('Conv1d', nn.Conv1d, [3, 8, 4, 1, 0, 1], {}),  # args: in_c, out_c, kernel_size, stride, padding, dilation
    ('Conv2d', nn.Conv2d, [4, 8, 4, 1, 0, 1], {}),
    ('Conv3d', nn.Conv3d, [4, 8, 4, 1, 0, 1], {}),
    ('RNN', nn.RNN, [], {'input_size': 4, 'hidden_size': 8, 'num_layers': 2}),
    ('LSTM', nn.LSTM, [], {'input_size': 4, 'hidden_size': 8, 'num_layers': 2}),
    ('GRU', nn.GRU, [], {'input_size': 4, 'hidden_size': 8, 'num_layers': 2}),
])
def test_build_nn_layer(nn_type, nn_class, args, kwargs):
    assert sequential.build_nn_layer(nn_type, *args, **kwargs).__class__ == nn_class


@pytest.mark.parametrize('k,v,dim,nn_class', [
    ('activation', 'ReLU', None, nn.ReLU),
    ('batch_norm', 8, 1, nn.BatchNorm1d),
    ('batch_norm', 8, 2, nn.BatchNorm2d),
    ('batch_norm', 8, 3, nn.BatchNorm3d),
    ('dropout', 0.2, 1, nn.Dropout),
    ('dropout', 0.2, 2, nn.Dropout2d),
    ('dropout', 0.2, 3, nn.Dropout3d),
    ('unknown', 1, 1, None.__class__),
    ('unknown', None, 1, None.__class__),
])
def test_build_sub_layer(k, v, dim, nn_class):
    assert sequential.build_sub_layer(k, v, dim).__class__ == nn_class


@pytest.mark.parametrize('arc,layer_names', [
    (
        CONV1D_ARC,
        ['Conv1d', 'BatchNorm1d', 'ReLU', 'Dropout', 'Conv1d', 'BatchNorm1d', 'ReLU', 'Dropout'],
    ), (
        CONV2D_ARC,
        ['Conv2d', 'BatchNorm2d', 'ReLU', 'Dropout2d', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Dropout2d'],
    ), (
        CONV3D_ARC,
        ['Conv3d', 'BatchNorm3d', 'ReLU', 'Dropout3d', 'Conv3d', 'BatchNorm3d', 'ReLU', 'Dropout3d'],
    ),
])
def test_build_layer_group_conv(arc, layer_names):
    layer_group_0 = sequential.build_layer_group(arc, idx=0)
    main_layer = layer_group_0[0]
    assert main_layer.in_channels == arc['in_shape'][0]
    assert main_layer.out_channels == arc['layers'][0][0]

    layer_group_1 = sequential.build_layer_group(arc, idx=1)
    main_layer = layer_group_1[0]
    assert main_layer.in_channels == arc['layers'][0][0]
    assert main_layer.out_channels == arc['layers'][1][0]

    assert net_util.get_layer_names(layer_group_0 + layer_group_1) == layer_names


@pytest.mark.parametrize('arc,layer_names', [
    (
        {  # basic, no sub layers
            'type': 'Linear',
            'in_features': 8,
            'layers': [64, 32],
            'batch_norm': False,
            'activation': None,
            'dropout': 0.0,
        },
        ['Linear', 'Linear'],
    ), (
        {  # with sub layers
            'type': 'Linear',
            'in_features': 8,
            'layers': [64, 32],
            'batch_norm': True,
            'activation': 'ReLU',
            'dropout': 0.2,
        },
        ['Linear', 'BatchNorm1d', 'ReLU', 'Dropout', 'Linear', 'BatchNorm1d', 'ReLU', 'Dropout'],
    ), (
        {  # reordered sub layers
            'type': 'Linear',
            'in_features': 8,
            'layers': [64, 32],
            'activation': 'ReLU',
            'batch_norm': True,
            'dropout': 0.2,
        },
        ['Linear', 'ReLU', 'BatchNorm1d', 'Dropout', 'Linear', 'ReLU', 'BatchNorm1d', 'Dropout'],
    ),
])
def test_build_layer_group_linear(arc, layer_names):
    layer_group_0 = sequential.build_layer_group(arc, idx=0)
    main_layer = layer_group_0[0]
    assert main_layer.in_features == arc['in_features']
    assert main_layer.out_features == arc['layers'][0]

    layer_group_1 = sequential.build_layer_group(arc, idx=1)
    main_layer = layer_group_1[0]
    assert main_layer.in_features == arc['layers'][0]
    assert main_layer.out_features == arc['layers'][1]

    assert net_util.get_layer_names(layer_group_0 + layer_group_1) == layer_names


@pytest.mark.parametrize('arc,layer_names', [
    (
        CONV1D_ARC,
        ['Conv1d', 'BatchNorm1d', 'ReLU', 'Dropout', 'Conv1d', 'BatchNorm1d', 'ReLU', 'Dropout'],
    ), (
        CONV2D_ARC,
        ['Conv2d', 'BatchNorm2d', 'ReLU', 'Dropout2d', 'Conv2d', 'BatchNorm2d', 'ReLU', 'Dropout2d'],
    ), (
        CONV3D_ARC,
        ['Conv3d', 'BatchNorm3d', 'ReLU', 'Dropout3d', 'Conv3d', 'BatchNorm3d', 'ReLU', 'Dropout3d'],
    ), (
        LINEAR_ARC,
        ['Linear', 'BatchNorm1d', 'ReLU', 'Dropout', 'Linear', 'BatchNorm1d', 'ReLU', 'Dropout'],
    )
])
def test_build_sequential(arc, layer_names):
    module = sequential.build(arc)
    assert module.__class__ == nn.Sequential
    assert net_util.get_layer_names(module.children()) == layer_names
