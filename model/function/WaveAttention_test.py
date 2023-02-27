import torch
import torch.nn as nn
from .torch_wavelets_1D import DWT_1D, IDWT_1D
from timm.models.vision_transformer import _cfg, LayerScale, Attention

class WaveAttention_test(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.dwt = DWT_1D(wave='haar')
        self.idwt = IDWT_1D(wave='haar')

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        
        # x = self.dwt(x)
        # x = self.idwt(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WaveAttention_wave2_2(nn.Module):
    def __init__(self,
                 dim,
                 N_dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.5,
                 proj_drop=0.5
                 ):
        super().__init__()
        # assert dim % 2 == 0, "dim should be divisible by 2"
        assert (dim // 2) % num_heads == 0, 'dim should be divisible by num_heads // 2'
        self.num_heads = num_heads
        head_dim = (dim // 2) // num_heads
        self.scale = head_dim ** -0.5
        # --------------------------------------------------------------------------
        self.dwt = DWT_1D(wave='haar')
        self.idwt = IDWT_1D(wave='haar')

        # print(f'N_dim {N_dim}')
        self.filter = nn.Sequential(
            nn.Conv1d((N_dim+1)*2, (N_dim+1)*2, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm1d((N_dim+1)*2),
            nn.ReLU(inplace=True)
        )
        self.reduce = nn.Sequential(
            nn.Conv1d((N_dim+1)*2, (N_dim+1), kernel_size=1, padding=0, stride=1),
            nn.BatchNorm1d((N_dim+1)),
            nn.ReLU(inplace=True)
        )
        # --------------------------------------------------------------------------

        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim + dim // 2, dim)
        self.proj_drop = nn.Dropout(proj_drop)
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
        # --------------------------------------------------------------------------
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
                                    # [128, 126, 768]   [B, N, D]
        x = self.dwt(x)             # [128, 252, 384]   [B, 2*N, D//2]
        x = self.filter(x)     # [128, 252, 384]   [B, 2*N, D//2]
        x_idwt = self.idwt(x)  # [128, 126, 768]   [B, N, D]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)     # [128, 8, 252, 48]   [B, heads, 2*N, D//2 // heads]
        attn = (q @ k.transpose(-2, -1)) * self.scale   # [128, 8, 252, 252]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # [128, 252, 384] [B, 2*N, D//2]

        x = self.reduce(x)                              # [128, 126, 384] [B, N, D//2]
        x = self.proj(torch.cat([x, x_idwt], dim=-1))   # [128, 126, 1152] [B, N, 3D//2]
        x = self.proj_drop(x)                           # [128, 126, 768] [B, N, D]

        return x

class WaveAttention_lh_res(nn.Module):
    def __init__(self,
                 dim,
                 N_dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.5,
                 proj_drop=0.5
                 ):
        super().__init__()
        assert dim % 2 == 0, "dim should be divisible by 2"
        assert dim % num_heads == 0, 'dim should be divisible by num_heads // 2'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # --------------------------------------------------------------------------
        self.dwt = DWT_1D(wave='haar')
        self.idwt = IDWT_1D(wave='haar')

        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim // 2, dim // 2)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # [128, 126, 768]   [B, N, D]
        B, N, _ = x.shape
        x = self.dwt(x)             # [128, 252, 384]   [B, 2*N, D//2]
        x_h = x[:,N:,:]
        x = x[:,:N,:]
        _, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = torch.cat([x,x_h], dim=1)
        x = self.idwt(x)
        return x

class WaveAttention_lh_l(nn.Module):
    def __init__(self,
                 dim,
                 N_dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.5,
                 proj_drop=0.5
                 ):
        super().__init__()
        assert dim % 2 == 0, "dim should be divisible by 2"
        assert dim % num_heads == 0, 'dim should be divisible by num_heads // 2'
        self.dwt = DWT_1D(wave='haar')
        self.idwt = IDWT_1D(wave='haar')
        self.attn_l = Attention(dim // 2, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop)
        self.attn_h = Attention(dim // 2, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=attn_drop)

    def forward(self, x):
        # [128, 126, 768]   [B, N, D]
        B, N, _ = x.shape
        x = self.dwt(x)             # [128, 252, 384]   [B, 2*N, D//2]
        x_h = x[:,N:,:]
        x = x[:,:N,:]

        x = self.attn_l(x)
        x_h = self.attn_h(x_h)

        x = torch.cat([x,x_h], dim=1)
        x = self.idwt(x)
        return x

if __name__ == '__main__':

    import numpy as np
    # def _pickup_patching(batch_data):
    #     # batch_size, n_channels, seq_len
    #     batch_size, n_channels, seq_len = batch_data.size()
    #     patch_size = 16
    #     assert seq_len % patch_size == 0
    #     batch_data = batch_data.view(batch_size, n_channels, seq_len // patch_size, patch_size)
    #     batch_data = batch_data.permute(0, 2, 1, 3)
    #     batch_data = batch_data.reshape(batch_size, seq_len // patch_size, n_channels * patch_size)
    #     return batch_data
    # inputs = np.ones((2, 30, 1600))
    # inputs = torch.from_numpy(inputs).float().to(torch.device('cuda'))
    # print(inputs.shape)
    # inputs = _pickup_patching(inputs)
    # print(inputs.shape)
    # wave_attn = WaveAttention2(dim=480).to(torch.device('cuda'))
    # wave_attn(inputs)