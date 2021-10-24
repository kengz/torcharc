from torcharc.module.perceiver import attention
import pytest
import torch


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


@pytest.mark.parametrize('x,context', [
    (torch.rand((2, 4, 11)), None),   # shape (batch index embed_dim)
    (torch.rand((2, 4, 11)), torch.rand((2, 4, 13))),
])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('num_heads', [1, 8])
def test_attention(x, context, head_dim, num_heads):
    embed_dim = x.shape[-1]
    context_dim = None if context is None else context.shape[-1]
    module = attention.Attention(embed_dim, context_dim, head_dim, num_heads)
    attn = module(x, context)
    assert attn.shape == x.shape


@pytest.mark.parametrize('x', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('widening_factor', [1, 4])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
def test_post_attention_mlp(x, widening_factor, dropout_p):
    embed_dim = x.shape[-1]
    module = attention.PostAttentionMLP(embed_dim, widening_factor, dropout_p)
    out = module(x)
    assert out.shape == x.shape
    assert not out.isnan().any()


@pytest.mark.parametrize('x', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('num_heads', [1, 8])
@pytest.mark.parametrize('widening_factor', [1, 4])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
def test_self_attention(x, head_dim, num_heads, widening_factor, dropout_p):
    embed_dim = x.shape[-1]
    module = attention.SelfAttention(embed_dim, head_dim=head_dim, num_heads=num_heads, widening_factor=widening_factor, dropout_p=dropout_p)
    out = module(x)
    assert out.shape == x.shape
    assert not out.isnan().any()


@pytest.mark.parametrize('x', [torch.rand((2, 4, 11))])
@pytest.mark.parametrize('context', [torch.rand((2, 4, 13))])
@pytest.mark.parametrize('use_query_residual', [True, False])
@pytest.mark.parametrize('head_dim', [64])
@pytest.mark.parametrize('num_heads', [1, 8])
@pytest.mark.parametrize('widening_factor', [1, 4])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
def test_cross_attention(x, context, use_query_residual, head_dim, num_heads, widening_factor, dropout_p):
    embed_dim = x.shape[-1]
    context_dim = context.shape[-1]
    module = attention.CrossAttention(embed_dim, context_dim, use_query_residual=use_query_residual, head_dim=head_dim, num_heads=num_heads, widening_factor=widening_factor, dropout_p=dropout_p)
    out = module(x, context)
    assert out.shape == x.shape
    assert not out.isnan().any()