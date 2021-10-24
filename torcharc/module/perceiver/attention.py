# basic modules for Perceiver
from torch import nn
import einops
import torch


def attend(q, k, v, mask=None):
    '''
    Compute multi-head attention using query, key, value.
    softmax((q.k)/sqrt(head_dim)).v
    @param q: query tensor of shape [batch, q_idx, num_heads, head_dim]
    @param k: key tensor of shape [batch, kv_idx, num_heads, head_dim] (kv_idx means k_idx = v_idx)
    @param v: value tensor of shape [batch, kv_idx, num_heads, head_dim] (kv_idx means k_idx = v_idx)
    @param mask: tensor of shape [batch, q_idx, kv_idx] indicating which attentions are valid
    @return: z tensor of shape [batch, q_idx, num_heads, head_dim]
    '''
    _batch, _q_idx, _num_heads, head_dim = q.shape
    scale = head_dim ** -0.5
    score = torch.einsum('b t h d, b T h d -> b h t T', q, k) * scale

    if mask is not None:
        max_neg_value = -torch.finfo(score.dtype).max
        score.masked_fill_(einops.rearrange(~mask, 'b t T -> b () t T'), max_neg_value)

    norm_score = score.softmax(dim=-1)
    z = torch.einsum('b h t T, b T h d -> b t h d', norm_score, v)
    return z


class Attention(nn.Module):
    '''
    Multi-headed {cross, self}-attention.
    Cross attention when context is different from x (embedding)
    Ref: https://jalammar.github.io/illustrated-transformer/
    single-head: z = softmax((q.k)/sqrt(head_dim)).v
    multi-head: z = (softmax((q.k)/sqrt(head_dim)).v) . W_z
    '''

    def __init__(self, embed_dim, context_dim=None, head_dim=64, num_heads=8):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        context_dim = embed_dim if context_dim is None else context_dim  # for k,v
        multi_head_dim = head_dim * num_heads

        # the main weights Q, K, V, and final MLP layer. Flatten (head_dim, num_heads) to multi_head_dim for efficiency
        # NOTE conv1d with kernel 1 and linear are the same
        # NOTE can trust PyTorch default init for now it uses lecun uniform
        self.to_flat_q = nn.Linear(embed_dim, multi_head_dim, bias=True)
        # forward-pass together then split in half to k,v for efficiency
        self.to_flat_kv = nn.Linear(context_dim, 2 * multi_head_dim, bias=True)
        # to reduce multi-head z to one z
        self.to_z = nn.Linear(multi_head_dim, embed_dim)

    def forward(self, x, context=None, mask=None):
        '''Compute multi-head attention'''
        flat_q = self.to_flat_q(x)
        context = x if context is None else context
        flat_k, flat_v = self.to_flat_kv(context).chunk(2, dim=-1)
        # unflatten to (num_heads, head_dim) for attention, shape (batch index num_heads head_dim)
        q = einops.rearrange(flat_q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = einops.rearrange(flat_k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = einops.rearrange(flat_v, 'b t (h d) -> b t h d', h=self.num_heads)

        multi_z = attend(q, k, v, mask=mask)
        # now combine multi-head z's into one z
        flat_z = einops.rearrange(multi_z, 'b t h d -> b t (h d)')
        z = self.to_z(flat_z)
        return z
