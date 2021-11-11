from torcharc.module.perceiver_io import postprocessor
import pytest
import torch


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('output_shape', [(3, 9)])
@pytest.mark.parametrize('proj_dim', [10])
def test_projection_postprocessor(batch, output_shape, proj_dim):
    out = torch.rand(batch, *output_shape)
    module = postprocessor.ProjectionPostprocessor(output_shape, proj_dim)
    post_out = module(out)
    assert list(post_out.shape) == [batch, output_shape[0], proj_dim]
    assert not post_out.isnan().any()
    post_out.mean().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('output_shape', [(1, 9)])
@pytest.mark.parametrize('num_classes', [10])
def test_classification_postprocessor(batch, output_shape, num_classes):
    out = torch.rand(batch, *output_shape)
    module = postprocessor.ClassificationPostprocessor(output_shape, num_classes)
    post_out = module(out)
    assert list(post_out.shape) == [batch, num_classes]
    assert not post_out.isnan().any()
    post_out.mean().backward()
