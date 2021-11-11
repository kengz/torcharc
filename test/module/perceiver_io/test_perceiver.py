from torcharc.module.perceiver_io import perceiver
import pytest
import torch


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('latent_shape', [(4, 11)])
@pytest.mark.parametrize('x', [torch.rand((2, 3, 13))])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('v_head_dim', [None, 32])
@pytest.mark.parametrize('cross_attn_num_heads', [1, 8])
@pytest.mark.parametrize('cross_attn_widening_factor', [1, 4])
@pytest.mark.parametrize('num_self_attn_blocks', [8])
@pytest.mark.parametrize('num_self_attn_per_block', [1, 4])
@pytest.mark.parametrize('self_attn_num_heads', [1, 8])
@pytest.mark.parametrize('self_attn_widening_factor', [1, 4])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
def test_perceiver_encoder(
        batch, latent_shape, x, head_dim, v_head_dim,
        cross_attn_num_heads, cross_attn_widening_factor,
        num_self_attn_blocks, num_self_attn_per_block,
        self_attn_num_heads, self_attn_widening_factor, dropout_p):
    input_dim = x.shape[-1]
    module = perceiver.PerceiverEncoder(
        latent_shape=latent_shape,
        input_dim=input_dim,
        head_dim=head_dim, v_head_dim=v_head_dim,
        cross_attn_num_heads=cross_attn_num_heads,
        cross_attn_widening_factor=cross_attn_widening_factor,
        num_self_attn_blocks=num_self_attn_blocks,
        num_self_attn_per_block=num_self_attn_per_block,
        self_attn_num_heads=self_attn_num_heads,
        self_attn_widening_factor=self_attn_widening_factor,
        dropout_p=dropout_p)
    assert 2 == len(list(module.encoder_processor.children()))  # self_attn_blocks is shared/reused
    out = module(x)
    assert list(out.shape) == [batch, *latent_shape]
    assert not out.isnan().any()
    out.mean().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('latent_shape', [(4, 11)])
@pytest.mark.parametrize('output_shape', [(3, 9)])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('v_head_dim', [None, 32])
@pytest.mark.parametrize('cross_attn_num_heads', [1, 8])
@pytest.mark.parametrize('cross_attn_widening_factor', [1, 4])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
def test_perceiver_decoder(
        batch, latent_shape, output_shape, head_dim, v_head_dim,
        cross_attn_num_heads, cross_attn_widening_factor, dropout_p):
    latent = torch.rand(batch, *latent_shape)
    module = perceiver.PerceiverDecoder(
        latent_shape=latent_shape,
        output_shape=output_shape,
        head_dim=head_dim, v_head_dim=v_head_dim,
        cross_attn_num_heads=cross_attn_num_heads,
        cross_attn_widening_factor=cross_attn_widening_factor,
        dropout_p=dropout_p)
    out = module(latent)
    assert list(out.shape) == [batch, *output_shape]
    assert not out.isnan().any()
    out.mean().backward()
