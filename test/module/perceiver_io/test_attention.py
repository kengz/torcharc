from torch import nn
from torcharc.module.perceiver_io import attention
import pytest
import torch


@pytest.mark.parametrize('embed', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('widening_factor', [1, 4])
def test_transformer_mlp(embed, widening_factor):
    embed_dim = embed.shape[-1]
    module = attention.TransformerMLP(embed_dim, widening_factor)
    out = module(embed)
    assert out.shape == embed.shape
    assert not out.isnan().any()
    out.mean().backward()


@pytest.mark.parametrize('embed', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
def test_residual(embed, dropout_p):
    embed_dim = embed.shape[-1]
    mlp = nn.Linear(embed_dim, embed_dim)
    module = attention.Residual(mlp, dropout_p)
    out = module(embed)
    assert out.shape == embed.shape
    assert not out.isnan().any()
    out.mean().backward()


@pytest.mark.parametrize('shape,mask', [
    ((2, 4, 8, 16), None),
    ((2, 4, 8, 16), torch.rand((2, 4, 4)).round().bool()),
])
def test_attend(shape, mask):
    # shape (batch index num_heads head_dim)
    q = torch.rand(shape)
    k = torch.rand(shape)
    v = torch.rand(shape)
    z = attention.attend(q, k, v, mask)
    assert z.shape == shape
    assert not z.isnan().any()


@pytest.mark.parametrize('embed,context', [
    (torch.rand((2, 4, 11)), None),   # shape (batch index embed_dim)
    (torch.rand((2, 4, 11)), torch.rand((2, 4, 13))),
])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('v_head_dim', [None, 32])
@pytest.mark.parametrize('num_heads', [1, 8])
def test_attention(embed, context, head_dim, v_head_dim, num_heads):
    embed_dim = embed.shape[-1]
    context_dim = None if context is None else context.shape[-1]
    module = attention.Attention(embed_dim, context_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads)
    attn = module(embed, context)
    assert attn.shape == embed.shape
    attn.mean().backward()


@pytest.mark.parametrize('embed', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('v_head_dim', [None, 32])
@pytest.mark.parametrize('num_heads', [1, 8])
def test_self_attention(embed, head_dim, num_heads, v_head_dim):
    embed_dim = embed.shape[-1]
    module = attention.SelfAttention(embed_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads)
    out = module(embed)
    assert out.shape == embed.shape
    assert not out.isnan().any()
    out.mean().backward()


@pytest.mark.parametrize('embed', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('context', [torch.rand((2, 4, 13))])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('v_head_dim', [None, 32])
@pytest.mark.parametrize('num_heads', [1, 8])
def test_cross_attention(embed, context, head_dim, v_head_dim, num_heads):
    embed_dim = embed.shape[-1]
    context_dim = context.shape[-1]
    module = attention.CrossAttention(embed_dim, context_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads)
    out = module(embed, context)
    assert out.shape == embed.shape
    assert not out.isnan().any()
    out.mean().backward()


@pytest.mark.parametrize('embed', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('v_head_dim', [None, 32])
@pytest.mark.parametrize('num_heads', [1, 8])
@pytest.mark.parametrize('widening_factor', [1, 4])
def test_build_self_attn_layer(embed, head_dim, v_head_dim, num_heads, widening_factor):
    embed_dim = embed.shape[-1]
    module = attention.build_self_attn_layer(embed_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads, widening_factor=widening_factor)
    out = module(embed)
    assert out.shape == embed.shape
    assert not out.isnan().any()
    out.mean().backward()


@pytest.mark.parametrize('num_self_attn_per_block', [1, 4])
@pytest.mark.parametrize('embed', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('v_head_dim', [None, 32])
@pytest.mark.parametrize('num_heads', [1, 8])
@pytest.mark.parametrize('widening_factor', [1, 4])
def test_build_self_attn_block(num_self_attn_per_block, embed, head_dim, v_head_dim, num_heads, widening_factor):
    embed_dim = embed.shape[-1]
    module = attention.build_self_attn_block(num_self_attn_per_block, embed_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads, widening_factor=widening_factor)
    out = module(embed)
    assert out.shape == embed.shape
    assert not out.isnan().any()
    out.mean().backward()


@pytest.mark.parametrize('embed', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('context', [torch.rand((2, 4, 13))])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('v_head_dim', [None, 32])
@pytest.mark.parametrize('num_heads', [1, 8])
@pytest.mark.parametrize('widening_factor', [1, 4])
def test_build_cross_attn_layer(embed, context, head_dim, v_head_dim, num_heads, widening_factor):
    embed_dim = embed.shape[-1]
    context_dim = context.shape[-1]
    module = attention.build_cross_attn_layer(embed_dim, context_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads, widening_factor=widening_factor)
    out = module(embed, context)
    assert out.shape == embed.shape
    assert not out.isnan().any()
    out.mean().backward()


@pytest.mark.parametrize('embed', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('context', [torch.rand((2, 4, 13))])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('v_head_dim', [None, 32])
@pytest.mark.parametrize('cross_attn_num_heads', [1, 8])
@pytest.mark.parametrize('cross_attn_widening_factor', [1, 4])
@pytest.mark.parametrize('num_self_attn_blocks', [8])
@pytest.mark.parametrize('num_self_attn_per_block', [1, 4])
@pytest.mark.parametrize('self_attn_num_heads', [1, 8])
@pytest.mark.parametrize('self_attn_widening_factor', [1, 4])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
def test_build_perceiver_layer(
        embed, context, head_dim, v_head_dim,
        cross_attn_num_heads,
        cross_attn_widening_factor,
        num_self_attn_blocks,
        num_self_attn_per_block,
        self_attn_num_heads,
        self_attn_widening_factor,
        dropout_p):
    embed_dim = embed.shape[-1]
    context_dim = context.shape[-1]
    module = attention.build_perceiver_layer(
        embed_dim, context_dim,
        head_dim=head_dim, v_head_dim=v_head_dim,
        cross_attn_num_heads=cross_attn_num_heads,
        cross_attn_widening_factor=cross_attn_widening_factor,
        num_self_attn_blocks=num_self_attn_blocks,
        num_self_attn_per_block=num_self_attn_per_block,
        self_attn_num_heads=self_attn_num_heads,
        self_attn_widening_factor=self_attn_widening_factor,
        dropout_p=dropout_p)
    assert 2 == len(list(module.children()))  # self_attn_blocks is shared/reused
    out = module(embed, context)
    assert out.shape == embed.shape
    assert not out.isnan().any()
    out.mean().backward()
