# basic modules for Perceiver based on https://github.com/deepmind/deepmind-research/tree/master/perceiver
from einops import rearrange
from torch import nn
from torcharc.module.sequential import SpreadSequential
import torch


class TransformerMLP(nn.Module):
    '''Transformer-style MLP to follow attention'''

    def __init__(self, embed_dim: int, widening_factor: int = 4):
        super().__init__()
        self.module = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * widening_factor),
            nn.GELU(),
            nn.Linear(embed_dim * widening_factor, embed_dim),
        )

    def forward(self, embed: torch.Tensor):
        return self.module(embed)


class Residual(nn.Module):
    '''Compute residual with dropout, i.e. residual = x + dropout(module(x))'''

    def __init__(self, module: nn.Module, dropout_p: float = 0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout_p)
        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    '''
    Compute multi-head attention using query, key, value.
    softmax((q.k)/sqrt(head_dim)).v
    NOTE no dropout is implemented here since Deepmind finds it deteriorates performance
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
        score.masked_fill_(rearrange(~mask, 'b t T -> b () t T'), max_neg_value)

    norm_score = score.softmax(dim=-1)
    z = torch.einsum('b h t T, b T h d -> b t h d', norm_score, v)
    return z


class Attention(nn.Module):
    '''
    Custom Multi-headed {cross, self}-attention that allows you to specify head_dim vs v_head_dim
    Cross attention is when context is different from x (embedding)
    Ref: https://jalammar.github.io/illustrated-transformer/
    single-head: z = softmax((q.k)/sqrt(head_dim)).v
    multi-head: z = (softmax((q.k)/sqrt(head_dim)).v) . W_z
    '''

    def __init__(self, embed_dim: int, context_dim: int = None, head_dim: int = 64, v_head_dim: int = None, num_heads: int = 8):
        super().__init__()
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim or head_dim  # optional different head_dim for v for perceiver output
        self.num_heads = num_heads
        context_dim = embed_dim if context_dim is None else context_dim  # for k,v
        multi_head_dim = self.head_dim * num_heads
        multi_v_head_dim = self.v_head_dim * num_heads

        # the main weights Q, K, V, and final MLP layer. Flatten (head_dim, num_heads) to multi_head_dim for efficiency
        # NOTE conv1d with kernel 1 and linear are the same
        # NOTE can trust PyTorch default init for now it uses lecun uniform
        self.to_flat_q = nn.Linear(embed_dim, multi_head_dim, bias=True)
        self.to_flat_k = nn.Linear(context_dim, multi_head_dim, bias=True)
        self.to_flat_v = nn.Linear(context_dim, multi_v_head_dim, bias=True)
        # to reduce multi-head z to one z
        self.to_z = nn.Linear(multi_v_head_dim, embed_dim)

    def forward(self, embed: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None):
        '''Compute multi-head attention'''
        flat_q = self.to_flat_q(embed)
        context = embed if context is None else context
        flat_k = self.to_flat_k(context)
        flat_v = self.to_flat_v(context)
        # unflatten to (num_heads, head_dim) for attention, shape (batch index num_heads head_dim)
        q = rearrange(flat_q, 'b t (h d) -> b t h d', h=self.num_heads)
        k = rearrange(flat_k, 'b t (h d) -> b t h d', h=self.num_heads)
        v = rearrange(flat_v, 'b t (h d) -> b t h d', h=self.num_heads)

        multi_z = attend(q, k, v, mask=mask)
        # now combine multi-head z's into one z
        flat_z = rearrange(multi_z, 'b t h d -> b t (h d)')
        z = self.to_z(flat_z)
        return z


class SelfAttention(nn.Module):
    '''Self-attention: the OG attention layer where q,k,v are generated from the  same embedding.'''

    def __init__(self, embed_dim: int, head_dim: int = 64, v_head_dim: int = None, num_heads: int = 8):
        super().__init__()
        self.attn = Attention(embed_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, embed: torch.Tensor, mask: torch.Tensor = None):
        return self.attn(self.layer_norm(embed), mask=mask)


class CrossAttention(nn.Module):
    '''Cross-attention: when k,v are generated from a different embedding (context) than q'''

    def __init__(self, embed_dim: int, context_dim: int, head_dim: int = 64, v_head_dim: int = None, num_heads: int = 8):
        super().__init__()
        self.attn = Attention(embed_dim, context_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads)
        self.embed_layer_norm = nn.LayerNorm(embed_dim)
        self.context_layer_norm = nn.LayerNorm(context_dim)

    def forward(self, embed: torch.Tensor, context: torch.Tensor, mask: torch.Tensor = None):
        return self.attn(self.embed_layer_norm(embed), context=self.context_layer_norm(context), mask=mask)


def build_self_attn_layer(embed_dim: int, head_dim: int = 64, v_head_dim: int = None, num_heads: int = 8, widening_factor: int = 4, dropout_p: float = 0.0) -> nn.Sequential:
    '''Build a self-attention layer as SelfAttention->Residual->TransformerMLP->Residual'''
    return nn.Sequential(
        Residual(SelfAttention(embed_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads), dropout_p),
        Residual(TransformerMLP(embed_dim, widening_factor), dropout_p)
    )


def build_self_attn_block(num_self_attn_per_block: int, embed_dim: int, head_dim: int = 64, v_head_dim: int = None, num_heads: int = 8, widening_factor: int = 4, dropout_p: float = 0.0) -> nn.Sequential:
    '''Build a block composed of multiple self-attention layer, i.e. n * [SelfAttention->Residual->TransformerMLP->Residual]'''
    return nn.Sequential(*[build_self_attn_layer(embed_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads, widening_factor=widening_factor, dropout_p=dropout_p) for _ in range(num_self_attn_per_block)])


def build_cross_attn_layer(embed_dim: int, context_dim: int, head_dim: int = 64, v_head_dim: int = None, num_heads: int = 8, widening_factor: int = 4, dropout_p: float = 0.0) -> SpreadSequential:
    '''Build a cross-attention layer as CrossAttention->Residual->TransformerMLP->Residual'''
    return SpreadSequential(
        Residual(CrossAttention(embed_dim, context_dim, head_dim=head_dim, v_head_dim=v_head_dim, num_heads=num_heads), dropout_p),
        Residual(TransformerMLP(embed_dim, widening_factor), dropout_p)
    )
