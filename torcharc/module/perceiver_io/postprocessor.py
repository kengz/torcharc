from torch import nn
from torcharc import net_util
from torcharc.module.perceiver_io.preprocessor import Identity  # noqa
import pydash as ps
import sys
import torch


class ProjectionPostprocessor(nn.Module):
    '''Postprocessing to use a linear layer to project PerceiverDecoder output's last dimension, i.e. shape (O, E) -> (O, out_dim).'''

    def __init__(self, in_shape: list, out_dim: int):
        super().__init__()
        self.in_shape = in_shape
        decoder_o, decoder_e = in_shape  # PerceiverDecoder output shape (O, E)
        self.out_dim = out_dim
        self.linear = nn.Linear(decoder_e, self.out_dim)
        self.out_shape = [decoder_o, out_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ClassificationPostprocessor(ProjectionPostprocessor):
    '''
    Classification postprocessing for Perceiver.
    This is just ProjectionPostprocessor except it tries to squeeze dim=1 whenever possible, i.e. when PerceiverDecoder output shape:
    - (O=1, E>=1): out_shape = [num_classes]; useful for plain classifier
    - (O>1, E>=1): out_shape = [O, num_classes]; useful for sequence-classifier, i.e. O=max_seq_len, num_classes=vocab_size
    '''

    def __init__(self, in_shape: list, out_dim: int):
        super().__init__(in_shape, out_dim)
        self.in_shape = in_shape
        self.out_shape = [out_dim] if in_shape[0] == 1 else [in_shape[0], out_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).squeeze(dim=1)  # output logits


class MultimodalPostprocessor(nn.Module):
    '''
    Multimodal postprocessor for multimodal output {mode: y}
    This recursively builds a postprocessor for each mode,
    then splits the input tensor by the seq_len (shape[1]) of each postprocessor,
    and applies them to each postprocessor and collect outputs in {mode: y}
    The output shape is {mode: postprocessor.out_shape}
    '''

    def __init__(self, in_shapes: dict, arc: dict):
        super().__init__()
        self.in_shapes = in_shapes
        assert len(ps.uniq([in_shape[-1] for in_shape in in_shapes.values()])) == 1, f'in_shape[-1] must be uniform, but got {in_shapes}'
        self.split_sizes = [shape[0] for shape in in_shapes.values()]
        self.total_seq_len = sum(self.split_sizes)
        self.postprocessors = nn.ModuleDict({
            mode: net_util.build_component(arc, {'in_shape': in_shape}, mode, sys.modules[__name__])
            for mode, in_shape in in_shapes.items()
        })
        self.out_shapes = {mode: preprocessor.out_shape for mode, preprocessor in self.postprocessors.items()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.total_seq_len, f'input shape[1]: {x.shape[1]} != {self.total_seq_len}'
        outs = {}
        split_xs = x.split(self.split_sizes, dim=1)  # x shape (O, E)
        for idx, (mode, postprocessor) in enumerate(self.postprocessors.items()):
            outs[mode] = postprocessor(split_xs[idx])
        return outs
