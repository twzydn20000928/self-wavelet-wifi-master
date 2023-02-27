from .resnet1d import resnet1d, ResNet1DConfig
from .resnet2d import resnet2d, ResNet2DConfig
from .lstm import lstm, LSTMConfig
from .vit import vit, ViTConfig
from .ds_resnet1d import ds_resnet1d, DSResNet1DConfig
from .ds_resnet2d import ds_resnet2d, DSResNet2DConfig
from .ds_lstm import ds_lstm, DSLSTMConfig
from .ds_vit import ds_vit, DSViTConfig
from .fs_resnet1d import fs_resnet1d, FSResNet1DConfig
from .wavevit import WaveVitConfig, waveVit_wifi

__all__ = [
    resnet1d, ResNet1DConfig,
    resnet2d, ResNet2DConfig,
    lstm, LSTMConfig,
    vit, ViTConfig,
    ds_resnet1d, DSResNet1DConfig,
    ds_resnet2d, DSResNet2DConfig,
    ds_lstm, DSLSTMConfig,
    ds_vit, DSViTConfig,
    fs_resnet1d, FSResNet1DConfig,
    WaveVitConfig, waveVit_wifi
]

'''
backbone只提取数据中高维特征，后续无论是整段分类还是逐帧分类通过head和strategy组合完成
'''
