import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_, Mlp
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, LayerScale, Attention
import math
import numpy as np

from ..function import (
    WaveAttention,
    StdAttention,
    WaveAttention_wave2,
    WaveAttention_lh,
    WaveAttention_2,
    WaveAttention_lh2,
    WaveAttention2_2,
    WaveAttention_test,
    WaveAttention_wave2_2,
WaveAttention_lh_res,
WaveAttention_lh_l,
WaveAttention_res
)
from ..model_config import ModelConfig

class WaveVitConfig(ModelConfig):
    """
        model_name format: wavevit_(attn_type)_(attn_type_ratio)_(scale)_(patch_size)_dropout_droppath
    """
    num_classes = 7
    seq_len = 224
    n_channel = 52
    patch_size = 16
    d_model = 768
    num_layer = 12
    n_head = 12
    dim_mlp_hidden = 2048
    expansion_factor = 4
    dropout = 0.5
    pooling = False
    MAX_PATCH_NUMS = 1000

    attn_type = 'wave'
    norm_layer = nn.LayerNorm

    high_ratio = 1.0

    def __init__(self, model_name: str):
        super(WaveVitConfig, self).__init__(model_name)
        # wavevit_wave_4_s_16_0.5
        _, attn_type, attn_type_layer, scale, patch_size, dropout, droppath, *list = model_name.split('_')
        if (len(list) != 0):
            self.high_ratio = float(list[0])

        self.attn_type_layer = int(attn_type_layer)
        self.patch_size = int(patch_size)
        self.attn_type = attn_type
        self.dropout = float(dropout)
        self.droppath = float(droppath)
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
        elif scale == 'test':
            # Base
            self.d_model = 768
            self.num_layer = 1
            self.n_head = 8

class SimpleSpanCLSHead(nn.Module):
    def __init__(self, hidden_dim, n_class):
        super(SimpleSpanCLSHead, self).__init__()

        self.head_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head_2 = nn.Linear(hidden_dim, n_class, bias=False)

        nn.init.xavier_uniform_(self.head_1.weight)
        nn.init.xavier_uniform_(self.head_2.weight)

    def forward(self, features):
        return self.head_2(self.head_1(features))


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 N_dim,
                 dim_mlp_hidden = 2048,
                 dropout = 0.5,
                 norm_layer = nn.LayerNorm,
                 attn_type = 'wave',
                 act_layer = nn.GELU,
                 qkv_bias = True,
                 attn_drop = 0.5,
                 init_values=None,
                 drop_path=0.1,
                 sr_ratio=1,
                 high_ratio=1.0
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(in_features=dim, hidden_features=dim_mlp_hidden, act_layer=act_layer, drop=dropout)

        # --------------------------------------------------------------------------
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # --------------------------------------------------------------------------
        if attn_type == 'wave':
            self.attn = WaveAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, sr_ratio=sr_ratio)
        elif attn_type == 'timm':
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop)
        elif attn_type == 'wave2':
            self.attn = WaveAttention_wave2(dim=dim,  N_dim=N_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop)
        elif attn_type == 'wavelh':
            self.attn = WaveAttention_lh(dim=dim,  N_dim=N_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop)
        elif attn_type == 'wavelh2':
            self.attn = WaveAttention_lh2(dim=dim,  N_dim=N_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop)
        elif attn_type == 'test':
            self.attn = WaveAttention_test(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop)
        elif attn_type == 'wave22':
            self.attn = WaveAttention_wave2_2(dim=dim, N_dim=N_dim, num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop, proj_drop=attn_drop)
        elif attn_type == 'wavelhres':
            self.attn = WaveAttention_lh_res(dim=dim, N_dim=N_dim, num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop, proj_drop=attn_drop)
        elif attn_type == 'wavel':
            self.attn = WaveAttention_lh_l(dim=dim, N_dim=N_dim, num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop, proj_drop=attn_drop)
        elif attn_type == 'waveres':
            self.attn = WaveAttention_res(dim=dim, N_dim=N_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=attn_drop, high_ratio=high_ratio)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class WaveVit(nn.Module):

    def __init__(self,
                 model_name = 'wavevit_timm_s_16',
                 num_classes = 10,
                 n_channel = 90,
                 seq_len = 2000,
                 patch_size = 16,
                 embed_dim = 768,
                 num_head = 12,
                 # dim_mlp_hidden = 2048,
                 dropout = 0.5,
                 drop_path = 0.1,
                 high_ratio=1.0,
                 expansion_factor=4,
                 depth = 12,
                 MAX_PATCH_NUMS = 1000,
                 pooling = False,
                 attn_type='wave',
                 attn_type_layer = 4,
                 norm_layer=nn.LayerNorm):

        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.n_channel = n_channel
        self.seq_len = seq_len

        self.N_dim = (seq_len // patch_size) if (seq_len % patch_size == 0) else (seq_len // patch_size + 1)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.depth = depth
        self.MAX_PATCH_NUMS = MAX_PATCH_NUMS
        self.pooling = pooling
        self.dim_mlp_hidden = embed_dim * expansion_factor
        self.dropout = dropout
        # --------------------------------------------------------------------------
        #
        self.cls_embed = nn.Parameter(torch.empty((1, 1, embed_dim)), requires_grad=True)
        self.pos_embed = nn.Parameter(torch.empty((1, 1 + MAX_PATCH_NUMS, embed_dim)),
                                      requires_grad=True)

        self.embedding = nn.Linear(n_channel * patch_size, embed_dim, bias=False)

        model_list = []
        for i in range(depth):
            self.attn_type = 'timm' if i >= attn_type_layer else attn_type
            model_list.append(Block(dim=embed_dim,
                                    num_heads=num_head,
                                    N_dim=self.N_dim,
                                    dim_mlp_hidden=self.dim_mlp_hidden,
                                    dropout=dropout,
                                    attn_drop=dropout,
                                    # attn_type='timm',
                                    drop_path=drop_path,
                                    attn_type=self.attn_type,
                                    sr_ratio=1,
                                    high_ratio=high_ratio))

        self.blocks = nn.ModuleList(model_list)
        self.norm = norm_layer(embed_dim)

        # self.head = SimpleSpanCLSHead(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.xavier_uniform_(self.cls_embed.data)
        nn.init.xavier_uniform_(self.pos_embed.data)

        self.apply(self._init_weights)
        # --------------------------------------------------------------------------

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pickup_patching(self, batch_data):
        # batch_size, n_channels, seq_len
        batch_size, n_channels, seq_len = batch_data.size()
        if (seq_len % self.patch_size != 0):
            batch_data = F.pad(batch_data,
                               (0, self.patch_size - (seq_len % self.patch_size)),
                               'constant', 0)
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
        x = torch.cat((self.cls_embed.repeat(batch_size, 1, 1), x), dim=1)
        # 加上Position Embedding
        x = x + self.pos_embed.repeat(batch_size, 1, 1)[:, :1 + num_patches, :]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.pooling:
            x = torch.mean(x, dim=1)
        else:
            x = x[:, 0, :]

        # x = self.head(x)

        return x

    def get_output_size(self):
        return self.embed_dim

    def get_model_name(self):
        return self.model_name

def waveVit_wifi(config: WaveVitConfig):
    model = WaveVit(
        dropout=config.dropout,
        drop_path=config.droppath,
        high_ratio=config.high_ratio,
        model_name=config.model_name,
        num_classes=config.num_classes,
        n_channel=config.n_channel,
        seq_len=config.seq_len,
        patch_size=config.patch_size,
        embed_dim=config.d_model,
        num_head=config.n_head,
        expansion_factor=config.expansion_factor,
        depth=config.num_layer,
        MAX_PATCH_NUMS=config.MAX_PATCH_NUMS,
        pooling=config.pooling,
        attn_type=config.attn_type,
        attn_type_layer=config.attn_type_layer,
        norm_layer=config.norm_layer
    )
    return model

if __name__ == '__main__':
    # seq_len // self.patch_size 必须得是一个奇数

    # inputs = np.ones((2, 30, 1584))
    # inputs = torch.from_numpy(inputs).float().to(torch.device('cuda'))
    # print(inputs.shape)
    # wave_vit = WaveVit(embed_dim = 480,
    #                    num_head = 8,
    #                    n_channel = 30).to(torch.device('cuda'))
    # wave_vit(inputs)

    inputs = np.ones((2, 5, 6))
    inputs = np.array([
        [
            [1,2,3,4,5],
            [6,7,8,9,10],
            [11,12,13,14,15]
        ],
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ]
    ])
    print(inputs.shape)
    # print(inputs)
    inputs = torch.from_numpy(inputs).float()
    batch_data = F.pad(inputs,
                       (0, 4 - (5 % 4)),
                       'constant', 0)
    # print(batch_data)
    batch_size, n_channels, seq_len = batch_data.size()
    # print(batch_data.shape)
    # print(batch_data)
    batch_data = batch_data.view(batch_size, n_channels, seq_len // 4, 4)

    batch_data = batch_data.permute(0, 2, 1, 3)
    batch_data = batch_data.reshape(batch_size, seq_len // 4, n_channels * 4)
    print(batch_data.shape)
    print(batch_data.transpose(1,2))
    print(batch_data)


