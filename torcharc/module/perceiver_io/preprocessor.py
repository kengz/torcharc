from einops import repeat, rearrange
from torch import nn
import math
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
        self.output_dim = embed_dim

    def forward(self, x):
        batch, seq_len = x.shape
        pe = repeat(self.pos_encoding[:seq_len], '... -> b ...', b=batch)  # repeat for batch
        return self.embedding(x) * self.scale + pe


class FourierPreprocessor(nn.Module):
    '''
    Spatial input preprocessor for PerceiverEncoder using Fourier positional encoding for any dimensions of spatial tensor with channels, i.e. shape (x, y, ..., c)
    This builds Fourier pos_encoding for coordinates of N-dimensional spatial data as a meshgrid
    e.g. for image of shape (x, y, c) -> get a meshgrid of shape (x, y, d=2), where each slice at d is a meshgrid of a dimension
    then generate the sin, cos frequencies, stack [pos, sin, cos],
    then flatten the meshgrid's spatial dimension into 1D to get the final pos_encoding of shape (x*y*..., d*(2*num_freq_bands+1)).
    When encoding, this flattens the spatial dimensions of input into 1D, e.g. (x, y, ..., c) into (x*y*..., c), then concat it with the pos_encoding, so the final output tensor is a stack of the [flattened input with channels, pos_encoding with d*(2*num_freq_bands+1).
    The output shape is (x*y*..., output_dim), where output_dim = (c+d*(2*num_freq_bands+1))

    @example
    '''

    def __init__(self, input_shape: list, num_freq_bands: int, cat_pos: bool = True):
        super().__init__()
        *self.spatial_shape, num_c = self.input_shape = list(input_shape)  # shape excluding batch
        self.num_freq_bands = num_freq_bands
        self.cat_pos = cat_pos
        # create fourier positional encoding
        pos = self.build_positions()
        self.pos_encoding = self.build_pos_encoding(pos)
        self.output_dim = num_c + self.get_pos_encoding_dim()  # input_dim to PerceiverEncoder; we stack pos_encoding with top of channels

    def build_positions(self, start: float = -1.0, end: float = 1.0):
        '''Build spatial coordinates as a meshgrid, i.e. coordinates laid out such that values along the channel is a point in coordinate, e.g. shape = (x, y, 2)'''
        x_y = [torch.linspace(start, end, steps=s) for s in self.spatial_shape]
        return torch.stack(torch.meshgrid(*x_y), dim=len(self.spatial_shape))

    def build_pos_encoding(self, pos: torch.Tensor, max_reso: list = None) -> torch.Tensor:
        '''
        Generate a Fourier frequency position encoding with linear spacing.
        @param pos: meshgrid position coordinates of shape (x, y, d=len(shape)), e.g. (x, y, 2), or (x, y, z, 3) etc. in general
        @param max_reso: maximum resolution (pixels) per dimension.
        @param cat_pos: whether to concat pos before the fourier encoding
        @return position encodings tensor of shape (x, y,... d*(2*num_freq_bands+1))
        '''
        max_reso = max_reso or pos.shape[:-1]
        freq_bands = torch.stack([torch.linspace(1.0, max_r / 2.0, steps=self.num_freq_bands) for max_r in max_reso])
        pos_freqs = rearrange(torch.einsum('...d,df->d...f', pos, freq_bands), 'd ... f -> ... (d f)')

        encodings = [pos] if self.cat_pos else []
        encodings += [torch.sin(math.pi * pos_freqs), torch.cos(math.pi * pos_freqs)]
        spatial_encoding = torch.cat(encodings, dim=-1)  # shape (x, y,... d*(2*num_freq_bands+1))
        # flatten spatial dimensions into 1D
        pos_encoding = rearrange(spatial_encoding, '... c -> (...) c')
        return pos_encoding

    def get_pos_encoding_dim(self) -> int:
        return len(self.spatial_shape) * (2 * self.num_freq_bands + int(self.cat_pos))

    def forward(self, x):
        batch, *x_in_shape = x.shape
        assert x_in_shape == self.input_shape, f'input shape {x_in_shape} != expected {self.input_shape}'

        pos_encoding = repeat(self.pos_encoding, '... -> b ...', b=batch)  # repeat for batch
        x = rearrange(x, 'b ... c -> b (...) c')  # flatten spatial dimensions into 1D
        return torch.cat([x, pos_encoding], dim=-1)  # stack 1D input with pos_encoding
