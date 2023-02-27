# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from ..model_config import ModelConfig


class ViTConfig(ModelConfig):
    """
        model_name format: vit_(scale)_(patch_size)
    """
    seq_len = 224
    n_channel = 52
    patch_size = 16
    d_model = 768
    num_layer = 12
    n_head = 12
    expansion_factor = 4
    dropout = 0.1
    pooling = False
    MAX_PATCH_NUMS = 1000
    def __init__(self, model_name: str):
        super(ViTConfig, self).__init__(model_name)
        # vit_s_16_0.5
        _, scale, patch_size, dropout = model_name.split('_')
        self.patch_size = int(patch_size)
        self.dropout = float(dropout)
        if scale == 'es':
            # Extra Small
            self.d_model = 128
            self.num_layer = 2
            self.n_head = 4
        elif scale == 'ms':
            # Medium Small
            self.d_model = 256
            self.num_layer = 4
            self.n_head = 8
        elif scale == 's':
            # Small
            self.d_model = 512
            self.num_layer = 8
            self.n_head = 8
        elif scale == 'b':
            # Base
            self.d_model = 768
            self.num_layer = 12
            self.n_head = 12


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head=12, d_ff=2048, dropout=0.5):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.msa = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, batch_data):
        # batch_size, 1 + num_patches, d_model
        residual = batch_data
        batch_data = self.norm1(batch_data)
        batch_data = self.msa(batch_data, batch_data, batch_data)[0]
        batch_data = self.dropout(batch_data)
        batch_data = residual + batch_data

        residual = batch_data
        batch_data = self.norm2(batch_data)
        batch_data = self.activation(self.linear1(batch_data))
        batch_data = self.dropout1(batch_data)
        batch_data = self.linear2(batch_data)
        batch_data = self.dropout2(batch_data)
        batch_data = residual + batch_data
        return batch_data


class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super(ViT, self).__init__()
        self.model_name = config.model_name
        self.MAX_PATCH_NUMS = config.MAX_PATCH_NUMS

        self.n_channel = config.n_channel
        self.seq_len = config.seq_len

        self.patch_size = config.patch_size

        self.num_layer = config.num_layer
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_ff = config.d_model * config.expansion_factor

        self.dropout = config.dropout
        self.pooling = config.pooling

        self.cls_embedding = nn.Parameter(torch.empty((1, 1, self.d_model)), requires_grad=True)
        self.embedding = nn.Linear(self.n_channel * self.patch_size, self.d_model, bias=False)

        self.position_embedding = nn.Parameter(torch.empty((1, 1 + self.MAX_PATCH_NUMS, self.d_model)),
                                               requires_grad=True)

        self.encoders = nn.Sequential(
            *[TransformerEncoder(self.d_model, self.n_head, self.d_ff, self.dropout) for _ in
              range(self.num_layer)]
        )

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.cls_embedding.data)
        nn.init.xavier_uniform_(self.position_embedding.data)

        self.norm = nn.LayerNorm(self.d_model)

    def _pickup_patching(self, batch_data):
        # batch_size, n_channels, seq_len
        batch_size, n_channels, seq_len = batch_data.size()
        assert seq_len % self.patch_size == 0
        batch_data = batch_data.view(batch_size, n_channels, seq_len // self.patch_size, self.patch_size)
        batch_data = batch_data.permute(0, 2, 1, 3)
        batch_data = batch_data.reshape(batch_size, seq_len // self.patch_size, n_channels * self.patch_size)
        return batch_data

    def forward(self, x):
        x = self._pickup_patching(x)
        x = self.embedding(x)
        batch_size, num_patches, _ = x.size()

        # 拼接CLS向量
        x = torch.cat((self.cls_embedding.repeat(batch_size, 1, 1), x), dim=1)
        # 加上Position Embedding
        x = x + self.position_embedding.repeat(batch_size, 1, 1)[:, :1 + num_patches, :]

        x = self.encoders(x)

        x = self.norm(x)
        if self.pooling:
            x = torch.mean(x, dim=1)
        else:
            x = x[:, 0, :]
        return x

    def get_output_size(self):
        return self.d_model

    def get_model_name(self):
        return self.model_name


def vit(config: ViTConfig):
    return ViT(config)
