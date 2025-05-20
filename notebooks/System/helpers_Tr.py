# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:23:12 2021

@author: bjorn

script holding for all ulility functions
"""
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(2, 1)
        #pe = pe.permute(0,2,1)
        #print('pe initial shape:',pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe[:x.size(1), :].squeeze(1)
        #print('x shape in pe forward:',x.shape)
        #print('pe initial shape:',self.pe.shape)
        x = x + self.pe[:x.size(0), :]
        # return self.dropout(x)
        return x

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep