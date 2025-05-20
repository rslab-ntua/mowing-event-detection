import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, emb_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_len, 2) * (-math.log(10000.0) / emb_len))
        pe = torch.zeros(1,seq_len, emb_len)
        pe[0,:,0::2] = torch.sin(position * div_term)
        pe[0,:,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe
        return self.dropout(x)
		
		
		


