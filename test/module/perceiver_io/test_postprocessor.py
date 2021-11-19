from torcharc.module.perceiver_io import postprocessor
import pytest
import torch


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('in_shape', [(3, 9)])
@pytest.mark.parametrize('out_dim', [10])
def test_projection_postprocessor(batch, in_shape, out_dim):
    x = torch.rand(batch, *in_shape)
    module = postprocessor.ProjectionPostprocessor(in_shape, out_dim)
    out = module(x)
    assert [in_shape[0], out_dim] == module.out_shape
    assert list(out.shape) == [batch, *module.out_shape]
    assert not out.isnan().any()
    out.mean().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('in_shape', [(1, 9), (3, 9)])
@pytest.mark.parametrize('out_dim', [10])
def test_classification_postprocessor(batch, in_shape, out_dim):
    x = torch.rand(batch, *in_shape)
    module = postprocessor.ClassificationPostprocessor(in_shape, out_dim)
    out = module(x)
    out_shape = [out_dim] if in_shape[0] == 1 else [in_shape[0], out_dim]
    assert out_shape == module.out_shape
    assert list(out.shape) == [batch, *module.out_shape]
    assert not out.isnan().any()
    out.mean().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('decoder_out_shape', [[100, 9]])
@pytest.mark.parametrize('in_shapes', [
    # NOTE use classifier_ts to simplify shape assert without squeeze dim
    {'classifier_ts': [4, 9], 'ts_1': [64, 9], 'ts_2': [32, 9]},
])
@pytest.mark.parametrize('arc', [
    {
        'classifier_ts': {
            'type': 'ClassificationPostprocessor',
            'out_dim': 10,
        },
        'ts_1': {
            'type': 'ProjectionPostprocessor',
            'out_dim': 4,
        },
        'ts_2': {
            'type': 'ProjectionPostprocessor',
            'out_dim': 4,
        },
    },
])
def test_multimodal_postprocessor(batch, decoder_out_shape, in_shapes, arc):
    x = torch.rand(batch, *decoder_out_shape)
    module = postprocessor.MultimodalPostprocessor(in_shapes, arc)
    outs = module(x)
    for mode, out in outs.items():
        assert [in_shapes[mode][0], arc[mode]['out_dim']] == module.out_shapes[mode]
        assert list(out.shape) == [batch, *module.out_shapes[mode]]
