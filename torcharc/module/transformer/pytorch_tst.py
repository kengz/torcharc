from torcharc.module.transformer import pos_encoder
from torch import nn
import torch


class PTTSTransformer(nn.Module):
    '''
    Transformer for time series using plain PyTorch transformers
    Adaptations from text transformer:
    - input embedding -> linear layer
    - limit on attention window
    - output layer is chosen according to problem

    Notable Parameters
    ----------
    d_model:
        Dimension of the input vector.
    pe:
        Type of positional encoding to add.
        Must be one of ``'sinusoid'``, ``'periodic'``, ``'learned'`` or ``None``. Default is ``None``.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    '''

    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dropout: float = 0.3,
            dim_feedforward: int = 2048,
            activation: str = 'relu',
            in_embedding: str = 'Linear',
            pe: str = None,
            attention_size: int = None,
            in_channels: int = 1,
            out_channels: int = 1,
            q: int = 8,  # Dimension of queries and keys.
            v: int = 8,  # Dimension of values.
            chunk_mode: bool = 'chunk',
            scalar_output: bool = False,
    ) -> None:
        super().__init__()

        self.in_embedding = pos_encoder.get_in_embedding(in_embedding, in_channels, d_model)
        # position encoder
        pe_classes = {
            'sinusoid': pos_encoder.SinusoidPE,
            'periodic': pos_encoder.PeriodicPE,
            'learned': pos_encoder.LearnedPE,
            None: None,
        }
        self.pe = pe_classes[pe](d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.encoders = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.decoders = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.out_linear = nn.Linear(d_model, out_channels)
        self.scalar_output = scalar_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input shape: (batch_size, seq_len, in_channels)
        output shape: (batch_size, seq_len, out_channels)
        '''
        encoding = self.in_embedding(x)
        if self.pe is not None:  # position encoding
            encoding = self.pe(encoding)
        encoding = self.encoders(encoding.transpose(0, 1))

        decoding = encoding
        if len(self.decoders.layers):
            if self.pe is not None:  # position encoding
                decoding = self.pe(decoding.transpose(0, 1)).transpose(0, 1)
            decoding = self.decoders(decoding, encoding).transpose(0, 1)

        if self.scalar_output:  # if want scalar instead of seq output, take the first index from seq
            decoding = decoding[:, 0, :]

        output = self.out_linear(decoding)
        return output
