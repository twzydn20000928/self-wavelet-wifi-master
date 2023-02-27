# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from .vit import TransformerEncoder
from ..model_config import ModelConfig


class DSViTConfig(ModelConfig):
    """
        model_name format: ds_vit_(scale)_(patch_size)_(patch_size)_(fusion)
    """
    seq_len = [224, 224]
    n_channel = [64, 64]
    patch_size = [8, 8]
    d_model = 768
    num_layer = 12
    n_head = 12
    expansion_factor = 4
    dropout = 0.1
    pooling = False
    # early|fully|late
    fusion = 'early'
    MAX_PATCH_NUMS = 1000

    def __init__(self, model_name: str):
        super(DSViTConfig, self).__init__(model_name)
        # ds_vit_s_8_8_fully
        _, __, scale, patch_size_1, patch_size_2, fusion = model_name.split('_')
        self.patch_size = [int(patch_size_1), int(patch_size_2)]
        self.fusion = fusion
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


class DSViT(nn.Module):
    def __init__(self, config: DSViTConfig):
        super(DSViT, self).__init__()
        self.model_name = config.model_name
        # basic parameter
        self.fusion = config.fusion
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

        # data embedding layer
        self.cls_embedding = nn.ParameterList()
        if self.fusion == 'late':
            # late fusion: two cls vector
            self.cls_embedding.append(nn.Parameter(torch.empty((1, 1, self.d_model)), requires_grad=True))
            self.cls_embedding.append(nn.Parameter(torch.empty((1, 1, self.d_model)), requires_grad=True))
        else:
            # early and fully fusion: one cls vector
            self.cls_embedding.append(nn.Parameter(torch.empty((1, 1, self.d_model)), requires_grad=True))

        self.embedding = nn.ModuleList([
            nn.Linear(self.n_channel[0] * self.patch_size[0], self.d_model, bias=False),
            nn.Linear(self.n_channel[1] * self.patch_size[1], self.d_model, bias=False)
        ])

        # encoder layer
        self.encoders = nn.ModuleList()
        if self.fusion == 'early':
            # early fusion: one branch
            self.encoders.append(nn.ModuleList([TransformerEncoder(self.d_model, self.n_head, self.d_ff, self.dropout)
                                                for _ in range(self.num_layer)]))
        else:
            # late and fully fusion: two branch
            self.encoders.append(nn.ModuleList([TransformerEncoder(self.d_model, self.n_head, self.d_ff, self.dropout)
                                                for _ in range(self.num_layer)]))
            self.encoders.append(nn.ModuleList([TransformerEncoder(self.d_model, self.n_head, self.d_ff, self.dropout)
                                                for _ in range(self.num_layer)]))

        self.transfer = nn.ModuleList()
        if self.fusion == 'fully':
            # fully fusion has middle transfer layer
            self.transfer.append(
                nn.ModuleList([nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(self.num_layer - 1)])
            )
            self.transfer.append(
                nn.ModuleList([nn.Linear(self.d_model, self.d_model, bias=False) for _ in range(self.num_layer - 1)])
            )

            for transfer in self.transfer:
                for i in range(self.num_layer - 1):
                    nn.init.xavier_uniform_(transfer[i].weight)

        # init
        for cls_embedding in self.cls_embedding:
            nn.init.xavier_uniform_(cls_embedding.data)
        for embedding in self.embedding:
            nn.init.xavier_uniform_(embedding.weight)

        self.position_embedding = nn.Parameter(torch.empty((1, 1 + self.MAX_PATCH_NUMS, self.d_model)),
                                               requires_grad=True)
        nn.init.xavier_uniform_(self.position_embedding.data)

        self.norm = nn.LayerNorm(self.d_model)

    def _pickup_patching(self, batch_data, patch_size):
        # batch_size, n_channels, seq_len
        batch_size, n_channels, seq_len = batch_data.size()
        assert seq_len % patch_size == 0
        batch_data = batch_data.view(batch_size, n_channels, seq_len // patch_size, patch_size)
        batch_data = batch_data.permute(0, 2, 1, 3)
        batch_data = batch_data.reshape(batch_size, seq_len // patch_size, n_channels * patch_size)
        return batch_data

    def forward(self, batch_data):
        batch_size = batch_data[0].size(0)
        assert batch_size == batch_data[1].size(0)
        '''embedding'''
        num_patches = [0, 0]
        # 首先分别做embedding
        for i in range(2):
            batch_data[i] = self._pickup_patching(batch_data[i], self.patch_size[i])
            batch_data[i] = self.embedding[i](batch_data[i])
            num_patches[i] = batch_data[i].size(1)

        # 分别加上cls embedding和position embedding
        # late fusion有两个cls
        if self.fusion == 'late':
            for i in range(2):
                # 拼接CLS向量
                batch_data[i] = torch.cat((self.cls_embedding[i].repeat(batch_size, 1, 1), batch_data[i]), dim=1)
                # 加上Position Embedding
                batch_data[i] = batch_data[i] + \
                                self.position_embedding.repeat(batch_size, 1, 1)[:, :1 + num_patches[i], :]
        # early fusion 和 fully fusion就将embedding和拼接在一起且，只有一个cls embedding
        else:
            # 首先加上PE
            for i in range(2):
                batch_data[i] = batch_data[i] + \
                                self.position_embedding.repeat(batch_size, 1, 1)[:, 1: 1 + num_patches[i], :]
            # 拼接两个数据
            batch_data = torch.cat(batch_data, dim=1)
            cls_embedding = self.cls_embedding[0].repeat(batch_size, 1, 1) + \
                            self.position_embedding.repeat(batch_size, 1, 1)[:, 0, :]
            batch_data = torch.cat((cls_embedding, batch_data), dim=1)

        '''feature extraction and fusion'''
        if self.fusion == 'fully':
            batch_data = [batch_data, batch_data]
            for layer in range(self.num_layer):
                for i in range(2):
                    batch_data[i] = self.encoders[i][layer](batch_data[i])
                if layer != self.num_layer - 1:
                    next_batch_data = [
                        batch_data[0] + self.transfer[1][layer](batch_data[1]),
                        batch_data[1] + self.transfer[0][layer](batch_data[0]),
                    ]
                    batch_data = next_batch_data
            for i in range(2):
                batch_data[i] = self.norm(batch_data[i])
            if self.pooling:
                return (torch.mean(batch_data[0], dim=1) + torch.mean(batch_data[1], dim=1)) / 2
            else:
                return (batch_data[0][:, 0, :] + batch_data[1][:, 0, :]) / 2
        elif self.fusion == 'early':
            for layer in range(self.num_layer):
                batch_data = self.encoders[0][layer](batch_data)
            batch_data = self.norm(batch_data)
            if self.pooling:
                return torch.mean(batch_data, dim=1)
            else:
                return batch_data[:, 0, :]
        elif self.fusion == 'late':
            for layer in range(self.num_layer):
                for i in range(2):
                    batch_data[i] = self.encoders[i][layer](batch_data[i])
            for i in range(2):
                batch_data[i] = self.norm(batch_data[i])
            if self.pooling:
                return (torch.mean(batch_data[0], dim=1) + torch.mean(batch_data[1], dim=1)) / 2
            else:
                return (batch_data[0][:, 0, :] + batch_data[1][:, 0, :]) / 2

    def get_output_size(self):
        return self.d_model

    def get_model_name(self):
        return self.model_name


def ds_vit(config: DSViTConfig):
    return DSViT(config)
