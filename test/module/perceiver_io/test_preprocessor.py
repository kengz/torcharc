from torcharc.module.perceiver_io import preprocessor
import pytest
import torch


@pytest.mark.parametrize('batch', [2])
@pytest.mark.parametrize('vocab_size', [1024])
@pytest.mark.parametrize('embed_dim', [256])
@pytest.mark.parametrize('max_seq_len', [512])
def test_text_preprocessor(batch, vocab_size, embed_dim, max_seq_len):
    x = torch.randint(vocab_size, (batch, max_seq_len))
    module = preprocessor.TextPreprocessor(vocab_size, embed_dim, max_seq_len)
    out = module(x)
    assert list(out.shape) == [batch, max_seq_len, embed_dim]
    out.mean().backward()
