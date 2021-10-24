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


@pytest.mark.parametrize('x_shape,context_shape', [
    (
        (2, 4, 11),  # shape (batch index embed_dim)
        None,
    ),
    (
        (2, 4, 11),
        (2, 4, 13),
    ),
])
@pytest.mark.parametrize('head_dim,num_heads', [
    (64, 1),
    (64, 8),
])
def test_attention(x_shape, context_shape, head_dim, num_heads):
    x = torch.rand(x_shape)
    embed_dim = x.shape[-1]
    context = torch.rand(context_shape) if context_shape else None
    context_dim = context.shape[-1] if context_shape else None
    module = attention.Attention(embed_dim, context_dim, head_dim, num_heads)
    attn = module(x, context)
    assert attn.shape == x.shape
