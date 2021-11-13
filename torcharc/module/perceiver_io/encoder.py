from einops import repeat
from torch import nn
from torcharc.module.perceiver_io.attention import build_cross_attn_layer, build_self_attn_block
from torcharc.module.sequential import SpreadSequential
import torch


class PerceiverEncoder(nn.Module):
    '''
    Perceiver IO: https://arxiv.org/abs/2107.14795
    The Encoder-Processor part of Perceiver model
    Build a Perceiver layer as cross-attention layer -> num_blocks * self-attention block,
    where the cross-attention layer encodes, and the self-attention block process for L times (Figure 2 in https://arxiv.org/abs/2107.14795)
    More detailed breakdown of Perceiver layer:
    cross_attn_layer: CrossAttention->Residual->TransformerMLP->Residual
    -> self_attn_block: n * [SelfAttention->Residual->TransformerMLP->Residual] -> (reuse the self_attn_block for total of num_self_attn_blocks times)

    NOTE notation mapping from ours to Deepmind's implementation in https://github.com/deepmind/deepmind-research/blob/master/perceiver/perceiver.py:
    latent_shape[-1], embed_dim, embed.shape[-1] -> inputs_q.shape[-1], latent_shape[-1], num_z_channels
    in_dim, context_dim -> inputs_kv.shape[-1], in_shape[-1]
    head_dim -> qk_channels / num_heads, q_head_dim
    v_head_dim -> v_channels / num_heads, v_head_dim

    @example
    latent_shape = [4, 11]
    x = torch.rand([2, 3, 13])
    in_dim = x.shape[-1]
    module = PerceiverEncoder(latent_shape=latent_shape, in_dim=in_dim)
    out = module(x)
    '''

    def __init__(
        self,
        latent_shape: list,  # (N, D) for latent array
        in_dim: int,  # the C of (M, C) for input array
        head_dim: int = 32,
        v_head_dim: int = None,
        cross_attn_num_heads: int = 1,
        cross_attn_widening_factor: int = 1,
        num_self_attn_blocks: int = 8,
        num_self_attn_per_block: int = 6,
        self_attn_num_heads: int = 8,
        self_attn_widening_factor: int = 1,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        latent_n, latent_d = latent_shape
        # Initialize the latent array for the initial cross-attend q
        self.latent = nn.Parameter(torch.empty(latent_n, latent_d))
        nn.init.trunc_normal_(self.latent, mean=0.0, std=0.02)  # Deepmind's init

        embed_dim = latent_d
        # encoder
        cross_attn_layer = build_cross_attn_layer(
            embed_dim=embed_dim, context_dim=in_dim,
            head_dim=head_dim, v_head_dim=v_head_dim,
            num_heads=cross_attn_num_heads,
            widening_factor=cross_attn_widening_factor,
            dropout_p=dropout_p)
        # processor
        self_attn_block = build_self_attn_block(
            num_self_attn_per_block=num_self_attn_per_block,
            embed_dim=embed_dim,
            head_dim=head_dim, v_head_dim=v_head_dim,
            num_heads=self_attn_num_heads,
            widening_factor=self_attn_widening_factor,
            dropout_p=dropout_p)
        # shared weights for self_attn_block to process L times
        shared_self_attn_blocks = num_self_attn_blocks * [self_attn_block]
        self.encoder_processor = SpreadSequential(cross_attn_layer, *shared_self_attn_blocks)
        self.out_shape = latent_shape

    def forward(self, x, mask=None):
        latent = repeat(self.latent, '... -> b ...', b=x.shape[0])  # repeat for batch
        return self.encoder_processor(latent, x, mask)
