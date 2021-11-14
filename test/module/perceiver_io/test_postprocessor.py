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
