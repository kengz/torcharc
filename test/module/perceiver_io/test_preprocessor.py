from torcharc.module.perceiver_io import preprocessor
import math
import pytest
import torch


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('vocab_size', [1024])
@pytest.mark.parametrize('embed_dim', [256])
@pytest.mark.parametrize('max_seq_len', [512])
def test_text_preprocessor(batch, vocab_size, embed_dim, max_seq_len):
    x = torch.randint(vocab_size, (batch, max_seq_len))
    module = preprocessor.TextPreprocessor(vocab_size, embed_dim, max_seq_len)
    assert embed_dim == module.output_dim
    out = module(x)
    assert [max_seq_len, embed_dim] == module.out_shape
    assert list(out.shape) == [batch, *module.out_shape]
    out.mean().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('in_shape', [
    [64, 3],  # series
    [64, 64, 3],  # image
    [64, 64, 64, 3]  # volume
])
@pytest.mark.parametrize('num_freq_bands', [32])
def test_text_preprocessor(batch, in_shape, num_freq_bands):
    x = torch.rand(batch, *in_shape)
    max_reso = [2 * r for r in in_shape[:-1]]  # max resolution twice the input size
    module = preprocessor.FourierPreprocessor(in_shape, num_freq_bands, max_reso)
    out = module(x)
    assert [math.prod(in_shape[:-1]), module.out_dim] == module.out_shape
    assert list(out.shape) == [batch, *module.out_shape]
