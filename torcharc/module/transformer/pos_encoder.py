from torch import nn
import numpy as np
import torch


# input embedding before position encoder: Linear or Conv1d

class Transpose(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


def get_in_embedding(nn_type: str, channels: int, d_model: int) -> nn.Module:
    '''Get the embedding for time series input: either Linear or Conv1d'''
    if nn_type == 'Linear':
        in_embedding = nn.Linear(channels, d_model)
    elif nn_type == 'Conv1d':
        # transpose input shape from NLC to NCL
        in_embedding = nn.Sequential(Transpose(), nn.Conv1d(in_channels=channels, out_channels=d_model, kernel_size=1, bias=False), Transpose())
    else:
        raise ValueError(f'nn_type {nn_type} not supported; try: Linear, Conv1d')
    return in_embedding


# position encoder

class SinusoidPE(nn.Module):
    '''Sinusoidal positional encoder (not learnable)'''

    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x shape is NLC (batch, seq_len, channels) to follow transformer convention of seq_len-first'''
        with torch.no_grad():  # this encoder has no learnable weights
            # NOTE sqrt of x used in NLP
            # x = x * torch.sqrt(self.d_model)
            seq_len = x.shape[1]
            x = x + self.pe[:, :seq_len]
            return self.dropout(x)


class PeriodicPE(nn.Module):
    '''Periodic positional encoder (not learnable) with a specified period (must match the granularity of input, e.g. hourly data, period = 24'''

    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 128, period: int = 24) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        pe = torch.sin(pos * 2 * np.pi / period)
        pe = pe.repeat(1, d_model).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x shape is NLC (batch, seq_len, channels) to follow transformer convention of seq_len-first'''
        with torch.no_grad():  # this encoder has no learnable weights
            seq_len = x.shape[1]
            x = x + self.pe[:, :seq_len]
            return self.dropout(x)


class LearnedPE(nn.Module):
    '''Learnable positional encoding using embedding'''

    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 128) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        idxs = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('idxs', idxs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x shape is NLC (batch, seq_len, channels) to follow transformer convention of seq_len-first'''
        idxs = self.idxs[0, :x.shape[1]].expand(x.shape[:2])
        pe = self.pos_embedding(idxs)
        x = x + pe
        x = self.layer_norm(x)
        return self.dropout(x)
