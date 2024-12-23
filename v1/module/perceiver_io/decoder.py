from einops import repeat
from torch import nn
from torcharc.module.perceiver_io.attention import build_cross_attn_layer
import torch


class PerceiverDecoder(nn.Module):
    '''
    The Decoder part of Perceiver model
    This is just a cross-attention layer with the latent array from PerceiverEncoder as context and an initialized output_query as embed
    cross_attn_layer: CrossAttention->Residual->TransformerMLP->Residual

    @example
    latent_shape = [4, 11]
    out_shape = [3, 9]
    latent = torch.rand(2, *latent_shape)
    module = PerceiverDecoder(latent_shape=latent_shape, out_shape=out_shape)
    out = module(latent)
    '''

    def __init__(
        self,
        latent_shape: list,  # (N, D) for latent array
        out_shape: list,  # (O, E) for output query array
        head_dim: int = 32,
        v_head_dim: int = None,
        cross_attn_num_heads: int = 1,
        cross_attn_widening_factor: int = 1,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        _latent_n, latent_d = self.latent_shape = latent_shape
        # Initialize the output query array for the initial cross-attend q
        self.output_query = nn.Parameter(torch.empty(*out_shape))
        nn.init.trunc_normal_(self.output_query, mean=0.0, std=0.02)  # Deepmind's init

        embed_dim = out_shape[-1]
        self.decoder = build_cross_attn_layer(
            embed_dim=embed_dim, context_dim=latent_d,
            head_dim=head_dim, v_head_dim=v_head_dim,
            num_heads=cross_attn_num_heads,
            widening_factor=cross_attn_widening_factor,
            dropout_p=dropout_p)
        self.out_shape = out_shape

    def forward(self, latent):
        output_query = repeat(self.output_query, '... -> b ...', b=latent.shape[0])  # repeat for batch
        return self.decoder(output_query, latent)
