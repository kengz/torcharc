from einops import repeat, rearrange
from torch import nn
from torcharc import net_util
import math
import pydash as ps
import sys
import torch


def build_learned_pos_encoding(max_seq_len: int, embed_dim: int):
    '''Build learned positional encoding with Deepmind's init'''
    # learned position encoding
    pos_encoding = nn.Parameter(torch.empty(max_seq_len, embed_dim))
    nn.init.trunc_normal_(pos_encoding, mean=0.0, std=0.02)  # Deepmind's init
    return pos_encoding


class Identity(nn.Identity):
    def __init__(self, in_shape: list):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = in_shape


class TextPreprocessor(nn.Module):
    '''Standard text preprocessing for transformer by embedding a tokenized tensor, then adding a learned position encoding.'''

    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int = 512, **_kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # learned position encoding
        self.pos_encoding = build_learned_pos_encoding(max_seq_len, embed_dim)
        self.scale = embed_dim ** 0.5
        self.out_shape = [max_seq_len, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    The output shape is (x*y*..., out_dim), where out_dim = (c+d*(2*num_freq_bands+1))

    @example
    batch = 2
    in_shape = [64, 3]
    num_freq_bands = 32
    x = torch.rand(batch, *in_shape)
    module = FourierPreprocessor(in_shape, num_freq_bands)
    out = module(x)
    assert [math.prod(in_shape[:-1]), module.out_dim] == module.out_shape
    assert list(out.shape) == [batch, *module.out_shape]
    '''

    def __init__(self, in_shape: list, num_freq_bands: int, max_reso: list = None, cat_pos: bool = True):
        super().__init__()
        *self.spatial_shape, num_c = self.in_shape = list(in_shape)  # shape excluding batch
        self.num_freq_bands = num_freq_bands
        self.cat_pos = cat_pos
        # create fourier positional encoding
        pos = self.build_positions()
        self.pos_encoding = self.build_pos_encoding(pos, max_reso=max_reso)
        flat_dim = math.prod(in_shape[:-1])
        self.out_dim = num_c + self.get_pos_encoding_dim()  # in_dim to PerceiverEncoder; we stack pos_encoding with top of channels
        self.out_shape = [flat_dim, self.out_dim]

    def build_positions(self, start: float = -1.0, end: float = 1.0):
        '''Build spatial coordinates as a meshgrid, i.e. coordinates laid out such that values along the channel is a point in coordinate, e.g. shape = (x, y, 2)'''
        x_y = [torch.linspace(start, end, steps=s) for s in self.spatial_shape]
        return torch.stack(torch.meshgrid(*x_y), dim=len(self.spatial_shape))

    def build_pos_encoding(self, pos: torch.Tensor, max_reso: list = None) -> torch.Tensor:
        '''
        Generate a Fourier frequency position encoding with linear spacing.
        @param pos: meshgrid position coordinates of shape (x, y, d=len(shape)), e.g. (x, y, 2), or (x, y, z, 3) etc. in general
        @param max_reso: maximum resolution (pixels) per dimension. Useful when input such as picture varies in size
        @param cat_pos: whether to concat pos before the fourier encoding
        @return position encodings tensor of shape (x, y,... d*(2*num_freq_bands+1))
        '''
        max_reso = max_reso or pos.shape[:-1]
        assert len(max_reso) == len(pos.shape[:-1]), f'max_reso len(shape) must match pos len(shape), but got {len(max_reso)} instead of {len(pos.shape[:-1])}'
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, *x_in_shape = x.shape
        assert x_in_shape == self.in_shape, f'input shape {x_in_shape} != expected {self.in_shape}'

        pos_encoding = repeat(self.pos_encoding, '... -> b ...', b=batch)  # repeat for batch
        x = rearrange(x, 'b ... c -> b (...) c')  # flatten spatial dimensions into 1D
        return torch.cat([x, pos_encoding], dim=-1)  # stack 1D input with pos_encoding


class MultimodalPreprocessor(nn.Module):
    '''
    Multimodal preprocessor for multimodal input {mode: x}
    This recursively builds a preprocessor for each mode, and applies them to the multimodal inputs in order.
    To combine the multimodal preprocessed outputs,
    first note that each output is a 2D array of (max_seq_len, channel) or (M, C) of Perceiver input array.
    They are padded with trainable position encoding (1 position per mode, broadcasted) to have the same common_channels (max_channels + pad_channels), before getting concatenated along the sequences for transformer to attend to.
    The output shape is [total_seq_len, common_channels]
    '''

    def __init__(self, in_shapes: dict, arc: dict, pad_channels: int = 2):
        super().__init__()
        self.preprocessors = nn.ModuleDict({
            mode: net_util.build_component(arc, {'in_shape': in_shape}, mode, sys.modules[__name__])
            for mode, in_shape in in_shapes.items()
        })
        self.out_shapes = {mode: preprocessor.out_shape for mode, preprocessor in self.preprocessors.items()}
        total_seq_len = ps.sum_by(self.out_shapes, ps.head)
        max_channels = ps.max_by(self.out_shapes, ps.last)[-1]
        common_channels = max_channels + pad_channels
        self.pos_encodings = {
            mode: build_learned_pos_encoding(1, common_channels - out_shape[-1])
            for mode, out_shape in self.out_shapes.items()
        }
        self.out_shape = [total_seq_len, common_channels]

    def pos_encoding_pad(self, mode: str, out: torch.Tensor) -> torch.Tensor:
        '''
        Pad output to ensure they result in shape [batch, seq_len, common_channels]
        The padding channels ensured by pad_channels are used to stack learned pos_encoding of shape [1, common_channels - out_dim] (broadcasted) for each mode,
        i.e. each mode has 1 encoded position for transformer to differentiate
        '''
        pos_encoding = self.pos_encodings[mode]
        batch, seq_len, _channel = out.shape
        padding = pos_encoding.broadcast_to((batch, seq_len, pos_encoding.shape[-1]))
        return torch.cat([out, padding], dim=2)  # concat along channel to result in common_channels

    def forward(self, xs: dict) -> torch.Tensor:
        outs = []
        for mode, x in xs.items():
            out = self.preprocessors[mode](x)
            padded_out = self.pos_encoding_pad(mode, out)
            outs.append(padded_out)
        # NOTE concat along seq_len to result in [total_seq_len, common_channels] since transformer attention is along seq_len, not channel
        return torch.cat(outs, dim=1)
