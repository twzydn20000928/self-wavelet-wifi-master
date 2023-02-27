# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from ..model_config import ModelConfig


class DSLSTMConfig(ModelConfig):
    """
        model_name format: ds_lstm_(layers)
    """
    seq_len = [128, 128]
    n_channel = [64, 64]
    num_layer = 2
    hidden_dim = 256
    dropout = 0.1
    batch_first = True
    bidirectional = False

    def __init__(self, model_name: str):
        super(DSLSTMConfig, self).__init__(model_name)
        # ds_lstm_(scales)
        _, __, scales = model_name.split('_')
        if scales == '1':
            self.hidden_dim = 128
            self.num_layer = 1
        elif scales == '2':
            self.hidden_dim = 256
            self.num_layer = 2
        elif scales == '3':
            self.hidden_dim = 512
            self.num_layer = 3


class DSLSTM(nn.Module):
    """
    LSTM不对数据切块，通过模型观察数据global情况
    """

    def __init__(self, config: DSLSTMConfig):
        super(DSLSTM, self).__init__()
        self.model_name = config.model_name

        self.seq_len = config.seq_len
        self.n_channel = config.n_channel

        self.hidden_dim = config.hidden_dim
        self.bidirectional = config.bidirectional

        self.branch_1 = nn.LSTM(input_size=config.hidden_dim, hidden_size=config.hidden_dim,
                                num_layers=config.num_layer, batch_first=config.batch_first,
                                dropout=config.dropout, bidirectional=config.bidirectional)
        self.branch_2 = nn.LSTM(input_size=config.hidden_dim, hidden_size=config.hidden_dim,
                                num_layers=config.num_layer, batch_first=config.batch_first,
                                dropout=config.dropout, bidirectional=config.bidirectional)

        self.embedding_1 = nn.Linear(config.n_channel[0], config.hidden_dim, bias=False)
        self.embedding_2 = nn.Linear(config.n_channel[1], config.hidden_dim, bias=False)

        nn.init.xavier_uniform_(self.embedding_1.weight)
        nn.init.xavier_uniform_(self.embedding_2.weight)

    def forward(self, batch_data):
        # batch_size, n_channels, seq_len -> batch_size, seq_len, n_channels
        for i in range(2):
            batch_data[i] = batch_data[i].permute(0, 2, 1)

        embedding_1, embedding_2 = self.embedding_1(batch_data[0]), self.embedding_2(batch_data[1])

        feature_1, feature_2 = self.branch_1(embedding_1)[0], self.branch_2(embedding_2)[0]
        feature_1, feature_2 = torch.mean(feature_1, dim=1), torch.mean(feature_2, dim=1)
        return torch.cat([feature_1, feature_2], dim=-1)

    def get_output_size(self):
        return self.hidden_dim * 2 * 2 if self.bidirectional else self.hidden_dim * 2

    def get_model_name(self):
        return self.model_name


def ds_lstm(config: DSLSTMConfig):
    return DSLSTM(config)
