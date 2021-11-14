from torcharc.module.perceiver_io import perceiver
import pytest
import torch


@pytest.mark.parametrize('batch', [2])
def test_perceiver(batch):
    arc = {
        'type': 'Perceiver',
        'in_shape': [64, 64, 3],
        'arc': {
            'preprocessor': {
                'type': 'FourierPreprocessor',
                'num_freq_bands': 32,
                'cat_pos': True,
            },
            'encoder': {
                'type': 'PerceiverEncoder',
                'latent_shape': [4, 11],
                'head_dim': 32,
                'v_head_dim': None,
                'cross_attn_num_heads': 1,
                'cross_attn_widening_factor': 1,
                'num_self_attn_blocks': 8,
                'num_self_attn_per_block': 6,
                'self_attn_num_heads': 8,
                'self_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'decoder': {
                'type': 'PerceiverDecoder',
                'out_shape': [1, 16],
                'head_dim': 32,
                'v_head_dim': None,
                'cross_attn_num_heads': 1,
                'cross_attn_widening_factor': 1,
                'dropout_p': 0.0,
            },
            'postprocessor': {
                'type': 'ClassificationPostprocessor',
                'out_dim': 10,
            }
        }
    }
    in_shape = arc['in_shape']
    x = torch.rand(batch, *in_shape)
    module = perceiver.Perceiver(**arc)
    out = module(x)
    assert list(out.shape) == [batch, *module._postprocessor.out_shape]
    assert not out.isnan().any()
    out.mean().backward()
