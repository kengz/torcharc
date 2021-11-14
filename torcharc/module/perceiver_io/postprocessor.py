from torch import nn
from torcharc.module.perceiver_io.preprocessor import Identity  # noqa


class ProjectionPostprocessor(nn.Module):
    '''Postprocessing to use a linear layer to project PerceiverDecoder output's last dimension, i.e. shape (O, E) -> (O, out_dim).'''

    def __init__(self, in_shape: list, out_dim: int):
        super().__init__()
        decoder_o, decoder_e = in_shape  # PerceiverDecoder output shape (O, E)
        self.out_dim = out_dim
        self.linear = nn.Linear(decoder_e, self.out_dim)
        self.out_shape = [decoder_o, out_dim]

    def forward(self, x):
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
        self.out_shape = [out_dim] if in_shape[0] == 1 else [in_shape[0], out_dim]

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)  # output logits

# TODO multimodal postprocessor by partitioning in_shape[0] to each postprocessor
