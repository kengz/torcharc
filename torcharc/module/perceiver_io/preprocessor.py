from einops import repeat
from torch import nn
import torch


class TextPreprocessor(nn.Module):
    '''Standard text preprocessing for transformer by embedding a tokenized tensor, then adding a learned position encoding.'''

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # learned position encoding
        self.pos_encoding = nn.Parameter(torch.empty(max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, mean=0.0, std=0.02)  # Deepmind's init
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        batch, seq_len = x.shape
        pe = repeat(self.pos_encoding[:seq_len], '... -> b ...', b=batch)  # repeat for batch
        return self.embedding(x) * self.scale + pe
