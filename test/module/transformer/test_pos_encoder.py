from torcharc.module.transformer import pos_encoder
import pytest
import torch


@pytest.mark.parametrize('d_model,batch_size,channels,seq_len', [
    (512, 4, 1, 32),
    (512, 4, 5, 32),
])
@pytest.mark.parametrize('embedding_type', [
    'Linear',
    'Conv1d',
    'unknown',
])
@pytest.mark.parametrize('PE', [
    pos_encoder.SinusoidPE,
    pos_encoder.PeriodicPE,
    pos_encoder.LearnedPE,
])
def test_pos_encoder(d_model, batch_size, channels, seq_len, embedding_type, PE):
    x = torch.rand(batch_size, seq_len, channels)

    if embedding_type == 'unknown':
        with pytest.raises(ValueError):
            in_embedding = pos_encoder.get_in_embedding(embedding_type, channels, d_model)
        return
    in_embedding = pos_encoder.get_in_embedding(embedding_type, channels, d_model)
    emb_x = in_embedding(x)
    assert emb_x.shape == torch.Size((batch_size, seq_len, d_model))

    pe = PE(d_model)
    y = pe(emb_x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size((batch_size, seq_len, d_model))
