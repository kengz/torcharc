# Implement Perceiver IO https://arxiv.org/abs/2107.14795
# inspired by https://github.com/deepmind/deepmind-research/blob/master/perceiver/perceiver.py and https://github.com/krasserm/perceiver-io
from torch import nn
from torcharc.module.perceiver_io import encoder, decoder, postprocessor, preprocessor
from torcharc.module.perceiver_io.attention import SpreadSequential
from torcharc.net_util import build_component
from typing import Union
import pydash as ps


class Perceiver(nn.Module):
    '''
    Perceiver module, composed of preprocessor -> PerceiverEncoder -> PerceiverDecoder -> postprocessor
    See the encoder and decoder modules for more implementation details
    '''

    def __init__(self, arc: dict, in_shape: list = None, in_shapes: dict = None):
        super().__init__()
        preprocessor_infer_arc = {'in_shapes': in_shapes} if in_shapes else {'in_shape': in_shape}  # only infer for non-multimodal
        self._preprocessor = build_component(arc, preprocessor_infer_arc, 'preprocessor', preprocessor)
        self._encoder = build_component(arc, {'in_dim': self._preprocessor.out_shape[-1]}, 'encoder', encoder)
        self._decoder = build_component(arc, {'latent_shape': self._encoder.out_shape}, 'decoder', decoder)

        postprocessor_infer_arc = {} if ps.get(arc, 'postprocessor.in_shapes') else {'in_shape': self._decoder.out_shape}  # only infer for non-multimodal
        self._postprocessor = build_component(arc, postprocessor_infer_arc, 'postprocessor', postprocessor)
        self.module = SpreadSequential(self._preprocessor, self._encoder, self._decoder, self._postprocessor)

    def forward(self, x):
        return self.module(x)
