from torcharc.module.perceiver_io import decoder
import pytest
import torch


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('latent_shape', [[4, 11]])
@pytest.mark.parametrize('out_shape', [[3, 9]])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('v_head_dim', [None, 32])
@pytest.mark.parametrize('cross_attn_num_heads', [1, 8])
@pytest.mark.parametrize('cross_attn_widening_factor', [1, 4])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
def test_perceiver_decoder(
        batch, latent_shape, out_shape, head_dim, v_head_dim,
        cross_attn_num_heads, cross_attn_widening_factor, dropout_p):
    latent = torch.rand(batch, *latent_shape)
    module = decoder.PerceiverDecoder(
        latent_shape=latent_shape,
        out_shape=out_shape,
        head_dim=head_dim, v_head_dim=v_head_dim,
        cross_attn_num_heads=cross_attn_num_heads,
        cross_attn_widening_factor=cross_attn_widening_factor,
        dropout_p=dropout_p)
    out = module(latent)
    assert out_shape == module.out_shape
    assert list(out.shape) == [batch, *module.out_shape]
    assert not out.isnan().any()
    out.mean().backward()
