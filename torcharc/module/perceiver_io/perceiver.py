# Implement Perceiver IO https://arxiv.org/abs/2107.14795
# inspired by https://github.com/deepmind/deepmind-research/blob/master/perceiver/perceiver.py and https://github.com/krasserm/perceiver-io
from torch import nn
from torcharc.module.perceiver_io import encoder, decoder, postprocessor, preprocessor
from torcharc.module.perceiver_io.attention import SpreadSequential
import pydash as ps


def build_component(arc: dict, infer_arc: dict, name: str,  module):
    '''Helper to build component of Perceiver'''
    sub_arc = arc[name]
    kwargs = ps.omit(sub_arc, 'type')
    kwargs.update(infer_arc)
    sub_module = getattr(module, sub_arc['type'])(**kwargs)
    return sub_module


class Perceiver(nn.Module):
    '''
    Perceiver module, composed of preprocessor -> PerceiverEncoder -> PerceiverDecoder -> postprocessor
    See the encoder and decoder modules for more implementation details
    '''

    def __init__(self, in_shape: list, arc: dict, **_kwargs):
        super().__init__()
        self._preprocessor = build_component(arc, {'in_shape': in_shape}, 'preprocessor', preprocessor)
        self._encoder = build_component(arc, {'in_dim': self._preprocessor.out_shape[-1]}, 'encoder', encoder)
        self._decoder = build_component(arc, {'latent_shape': self._encoder.out_shape}, 'decoder', decoder)
        self._postprocessor = build_component(arc, {'in_shape': self._decoder.out_shape}, 'postprocessor', postprocessor)
        self.module = SpreadSequential(self._preprocessor, self._encoder, self._decoder, self._postprocessor)

    def forward(self, x):
        return self.module(x)
