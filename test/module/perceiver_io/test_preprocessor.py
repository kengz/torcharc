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
    assert list(out.shape) == [batch, max_seq_len, module.output_dim]
    out.mean().backward()


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('dim_shape', [
    [64, 3],  # series
    [64, 64, 3],  # image
    [64, 64, 64, 3]  # volume
])
@pytest.mark.parametrize('num_freq_bands', [32])
def test_text_preprocessor(batch, dim_shape, num_freq_bands):
    x = torch.rand(batch, *dim_shape)
    module = preprocessor.FourierPreprocessor(dim_shape, num_freq_bands)
    out = module(x)
    flat_dim = math.prod(dim_shape[:-1])
    assert list(out.shape) == [batch, flat_dim, module.output_dim]
