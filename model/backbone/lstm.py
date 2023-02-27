# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from ..model_config import ModelConfig


class LSTMConfig(ModelConfig):
    """
        model_name format: lstm_(scales)
    """
    seq_len = 224
    n_channel = 52
    num_layer = 2
    hidden_dim = 256
    dropout = 0.1
    batch_first = True
    bidirectional = False

    def __init__(self, model_name: str):
        super(LSTMConfig, self).__init__(model_name)
        _, scales = model_name.split('_')
        if scales == '1':
            self.hidden_dim = 128
            self.num_layer = 1
        elif scales == '2':
            self.hidden_dim = 256
            self.num_layer = 2
        elif scales == '3':
            self.hidden_dim = 512
            self.num_layer = 3


class LSTM(nn.Module):
    """
    LSTM不对数据切块，通过模型观察数据global情况
    """

    def __init__(self, config: LSTMConfig):
        super(LSTM, self).__init__()
        self.model_name = config.model_name
        self.seq_len = config.seq_len
        self.n_channel = config.n_channel
        self.hidden_dim = config.hidden_dim
        self.bidirectional = config.bidirectional

        self.core = nn.LSTM(input_size=config.hidden_dim, hidden_size=config.hidden_dim,
                            num_layers=config.num_layer, batch_first=config.batch_first,
                            dropout=config.dropout, bidirectional=config.bidirectional)

        self.embedding = nn.Linear(config.n_channel, config.hidden_dim, bias=False)

        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        # batch_size, n_channels, seq_len -> batch_size, seq_len, n_channels
        x = x.permute(0, 2, 1)
        x = self.embedding(x)

        x = self.core(x)[0]
        return x

    def get_output_size(self):
        return self.hidden_dim * 2 if self.bidirectional else self.hidden_dim

    def get_model_name(self):
        return self.model_name


def lstm(config: LSTMConfig):
    return LSTM(config)
