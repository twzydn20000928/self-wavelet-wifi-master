import os
import argparse
import logging
import math
import numpy as np


from data_process import (
    WiFiARDatasetConfig,
    WiFiVioDatasetConfig,
)

from model import ModelConfig
from model import (
    ResNet1DConfig,
    ResNet2DConfig,
    LSTMConfig,
    ViTConfig,
    DSResNet1DConfig,
    DSResNet2DConfig,
    DSLSTMConfig,
    DSViTConfig,
    FSResNet1DConfig,
    WaveVitConfig,
)
from model import (
    ARILSpanCLSConfig,
    WiFiARSpanCLSConfig,
    HTHISpanCLSConfig,
    ARILLateFusionSpanCLSConfig,
    WiFiARLateFusionSpanCLSConfig,
)

from util import augmentation

import strategy
import strategy.single_stream

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)


def init_dataset(dataset_name: str, datasource_path: os.path, check_point_path: os.path, is_test: bool):
    # name, *_ = dataset_name.split('_')
    if dataset_name.startswith('WiAR'):
        '''
        WiAR_(train_val_rate)
        '''
        _, train_val_rate = dataset_name.split('_')
        train_val_rate = float(train_val_rate)

        Data_path = os.path.join("Data.pt")

        dataset_config = WiFiARDatasetConfig(os.path.join(datasource_path, Data_path), train_val_rate)
        from data_process import load_WiAR_dataset, WiFiARDataset
        train_dataset, test_dataset = load_WiAR_dataset(dataset_config)
        train_dataset, test_dataset = WiFiARDataset(train_dataset), WiFiARDataset(test_dataset)
        return train_dataset, test_dataset

    if dataset_name.startswith('WiVio'):
        '''
        WiVio
        '''
        if dataset_name.startswith('WiVioAUG'):
            name, *augs = dataset_name.split('_')
            if len(augs) == 0:
                augs_list = None
            else:
                augs_list = []
                for aug in augs:
                    if aug == 'jitter':
                        augs_list.append(augmentation.jitter)
                    elif aug == 'scaling':
                        augs_list.append(augmentation.scaling)
                    elif aug == 'rotate':
                        augs_list.append(augmentation.rotation)
                    elif aug == 'magwarp':
                        augs_list.append(augmentation.magnitude_warp)
                    elif aug == 'window-s':
                        augs_list.append(augmentation.window_slice)
                    elif aug == 'window-w':
                        augs_list.append(augmentation.window_warp)
                    elif aug == 'mean':
                        augs_list.append('mean-mix')
                    elif aug == 'w-s-mag':
                        augs_list.append('window-s_magwarp')
                    elif aug == 'w-s-w-w':
                        augs_list.append('window-s_window-w')
                    elif aug == 'w-s-mean':
                        augs_list.append('window-s_mean-mix')
        else:
            augs_list = None

        dataset_config = WiFiVioDatasetConfig(os.path.join(datasource_path), os.path.join(check_point_path))
        from data_process import load_wifi_Vio_data, WiFiVioDataset
        train_dataset, test_dataset = load_wifi_Vio_data(dataset_config)
        train_dataset, test_dataset = WiFiVioDataset(train_dataset, is_test=is_test, augs_list=augs_list), WiFiVioDataset(test_dataset, is_test=is_test)
        return train_dataset, test_dataset



def _init_backbone_config(backbone_name: str):
    config = None
    if backbone_name.startswith('resnet1d'):
        config = ResNet1DConfig(backbone_name)
    elif backbone_name.startswith('resnet2d'):
        config = ResNet2DConfig(backbone_name)
    elif backbone_name.startswith('lstm'):
        config = LSTMConfig(backbone_name)
    elif backbone_name.startswith('vit'):
        config = ViTConfig(backbone_name)
    elif backbone_name.startswith('ds_resnet1d'):
        config = DSResNet1DConfig(backbone_name)
    elif backbone_name.startswith('ds_resnet2d'):
        config = DSResNet2DConfig(backbone_name)
    elif backbone_name.startswith('ds_lstm'):
        config = DSLSTMConfig(backbone_name)
    elif backbone_name.startswith('ds_vit'):
        config = DSViTConfig(backbone_name)
    elif backbone_name.startswith('fs_resnet1d'):
        config = FSResNet1DConfig(backbone_name)
    elif backbone_name.startswith('wavevit'):
        config = WaveVitConfig(backbone_name)
    return config


def _init_head_config(head_name: str):
    config = None
    if head_name == 'aril_span_cls':
        config = ARILSpanCLSConfig(head_name)
    elif head_name == 'wifi_ar_span_cls':
        config = WiFiARSpanCLSConfig(head_name)
    elif head_name == 'hthi_span_cls':
        config = HTHISpanCLSConfig(head_name)
    elif head_name == 'aril_late_fusion_span_cls':
        config = ARILLateFusionSpanCLSConfig(head_name)
    elif head_name == 'wifi_ar_late_fusion_span_cls':
        config = WiFiARLateFusionSpanCLSConfig(head_name)
    return config


def _init_backbone(config: ModelConfig):
    logger.info('初始化模型：%s' % config.model_name)
    if isinstance(config, ResNet1DConfig):
        from model import resnet1d
        return resnet1d(config)
    elif isinstance(config, ResNet2DConfig):
        from model import resnet2d
        return resnet2d(config)
    elif isinstance(config, LSTMConfig):
        from model import lstm
        return lstm(config)
    elif isinstance(config, ViTConfig):
        from model import vit
        return vit(config)
    elif isinstance(config, DSResNet1DConfig):
        from model import ds_resnet1d
        return ds_resnet1d(config)
    elif isinstance(config, DSResNet2DConfig):
        from model import ds_resnet2d
        return ds_resnet2d(config)
    elif isinstance(config, DSLSTMConfig):
        from model import ds_lstm
        return ds_lstm(config)
    elif isinstance(config, DSViTConfig):
        from model import ds_vit
        return ds_vit(config)
    elif isinstance(config, FSResNet1DConfig):
        from model import fs_resnet1d
        return fs_resnet1d(config)
    elif isinstance(config, WaveVitConfig):
        from model import waveVit_wifi
        return waveVit_wifi(config)


def _init_head(hidden_dim, config: ModelConfig):
    logger.info('初试化预测头：%s' % config.model_name)
    if isinstance(config, ARILSpanCLSConfig):
        from model import ARILSpanCLS
        return ARILSpanCLS(hidden_dim, config)
    elif isinstance(config, WiFiARSpanCLSConfig):
        from model import WiFiARSpanCLS
        return WiFiARSpanCLS(hidden_dim, config)
    elif isinstance(config, HTHISpanCLSConfig):
        from model import HTHISpanCLS
        return HTHISpanCLS(hidden_dim, config)
    elif isinstance(config, ARILLateFusionSpanCLSConfig):
        from model import ARILLateFusionSpanCLS
        return ARILLateFusionSpanCLS(hidden_dim, config)
    elif isinstance(config, WiFiARLateFusionSpanCLSConfig):
        from model import WiFiARLateFusionSpanCLS
        return WiFiARLateFusionSpanCLS(hidden_dim, config)


def _init_strategy_config(strategy_name: str):
    mapping = {
        'r': 'raw', 'f': 'freq',
        't': 'time', 'c': 'channel',
    }
    """解析strategy_name中的参数"""
    config = None
    '''Single Stream'''
    if strategy_name.startswith('resnet1d_span_cls'):
        '''resnet1d_span_cls_(conv_data)_(conv_dim)'''
        config = strategy.single_stream.Resnet1DSpanCLSConfig(strategy_name)

        _, __, ___, conv_data, conv_dim = strategy_name.split('_')
        config.conv_data = conv_data
        config.conv_dim = conv_dim
    elif strategy_name.startswith('early_cat'):
        '''early_cat_(conv_data)_(stack_dim)_(stack_num)'''
        config = strategy.single_stream.EarlyCatConfig(strategy_name)

        _, __, conv_data, stack_dim, stack_num = strategy_name.split('_')
        config.conv_data = conv_data
        config.stack_dim = stack_dim
        config.stack_num = int(stack_num)
    elif strategy_name.startswith('late_fusion'):
        '''late_fusion_(conv_data)_(conv_dim)_(pooling_method)_(downsample_factor)'''
        config = strategy.single_stream.LateFusionConfig(strategy_name)

        _, __, conv_data, conv_dim, pooling_method, downsample_factor = strategy_name.split('_')
        config.conv_data = conv_data
        config.conv_dim = conv_dim
        config.pooling_method = pooling_method
        config.downsample_factor = int(downsample_factor)
    elif strategy_name.startswith('lstm_span_cls'):
        '''lstm_span_cls_(calc_data)'''
        config = strategy.single_stream.LSTMSpanCLSConfig(strategy_name)

        _, __, ___, calc_data = strategy_name.split('_')
        config.conv_data = calc_data
    elif strategy_name.startswith('resnet2d_span_cls'):
        '''resnet2d_span_cls_(calc_data)'''
        config = strategy.single_stream.Resnet2DSpanCLSConfig(strategy_name)

        _, __, ___, calc_data = strategy_name.split('_')
        config.conv_data = calc_data
    elif strategy_name.startswith('vit_span_cls'):
        '''vit_span_cls_(calc_data)'''
        config = strategy.single_stream.ViTSpanCLSConfig(strategy_name)

        _, __, ___, calc_data = strategy_name.split('_')
        config.conv_data = calc_data
    elif strategy_name.startswith('ds_resnet1d_span_cls'):
        '''ds_resnet1d_span_cls_((conv_data)(conv_dim)(conv_data)(conv_dim))'''
        config = strategy.dual_stream.DSResnet1DSpanCLSConfig(strategy_name)
        _, __, ___, ____, focus = strategy_name.split('_')
        for i in range(2):
            config.conv_data[i] = mapping[focus[i * 2]]
            config.conv_dim[i] = mapping[focus[i * 2 + 1]]
    elif strategy_name.startswith('ds_resnet2d_span_cls'):
        '''ds_resnet2d_span_cls'''
        config = strategy.dual_stream.DSResnet2DSpanCLSConfig(strategy_name)
    elif strategy_name.startswith('ds_lstm_span_cls'):
        '''ds_lstm_span_cls'''
        config = strategy.dual_stream.DSLSTMSpanCLSConfig(strategy_name)
    elif strategy_name.startswith('ds_vit_span_cls'):
        '''ds_vit_span_cls'''
        config = strategy.dual_stream.DSViTSpanCLSConfig(strategy_name)
    elif strategy_name.startswith('fs_resnet1d_span_cls'):
        '''fs_resnet1d_span_cls'''
        config = strategy.four_stream.FSResnet1DSpanCLSConfig(strategy_name)
    return config


def init_strategy(backbone_name: str, head_name: str, strategy_name: str, n_channels: dict, seq_lens: dict):
    logger.info('初试化训练策略：%s' % strategy_name)
    backbone_config = _init_backbone_config(backbone_name)
    head_config = _init_head_config(head_name)
    strategy_config = _init_strategy_config(strategy_name)

    if isinstance(strategy_config, strategy.single_stream.Resnet1DSpanCLSConfig):
        from strategy.single_stream import Resnet1DSpanCLS
        if strategy_config.conv_data == 'raw':
            if strategy_config.conv_dim == 'time':
                backbone_config.in_channel = n_channels['data']
            elif strategy_config.conv_dim == 'channel':
                backbone_config.in_channel = seq_lens['data']
        elif strategy_config.conv_data == 'freq':
            if strategy_config.conv_dim == 'time':
                backbone_config.in_channel = n_channels['freq_data']
            elif strategy_config.conv_dim == 'channel':
                backbone_config.in_channel = seq_lens['freq_data']
        backbone = _init_backbone(backbone_config)

        head = _init_head(backbone.get_output_size(), head_config)

        return Resnet1DSpanCLS(backbone, head, strategy_config)
    elif isinstance(strategy_config, strategy.single_stream.EarlyCatConfig):
        from strategy.single_stream import EarlyCat
        if strategy_config.conv_data == 'raw':
            if strategy_config.stack_dim == 'time':
                backbone_config.in_channel = n_channels['data'] // strategy_config.stack_num
            elif strategy_config.stack_dim == 'channel':
                backbone_config.in_channel = seq_lens['data'] // strategy_config.stack_num
        elif strategy_config.conv_data == 'freq':
            if strategy_config.stack_dim == 'time':
                backbone_config.in_channel = n_channels['freq_data'] // strategy_config.stack_num
            elif strategy_config.stack_dim == 'channel':
                backbone_config.in_channel = seq_lens['freq_data'] // strategy_config.stack_num
        backbone = _init_backbone(backbone_config)

        head = _init_head(backbone.get_output_size(), head_config)

        return EarlyCat(backbone, head, strategy_config)
    elif isinstance(strategy_config, strategy.single_stream.LateFusionConfig):
        from strategy.single_stream import LateFusion
        if strategy_config.conv_data == 'raw':
            if strategy_config.conv_dim == 'time':
                backbone_config.in_channel = n_channels['data']
            elif strategy_config.conv_dim == 'channel':
                backbone_config.in_channel = seq_lens['data']
        elif strategy_config.conv_data == 'freq':
            if strategy_config.conv_dim == 'time':
                backbone_config.in_channel = n_channels['freq_data']
            elif strategy_config.conv_dim == 'channel':
                backbone_config.in_channel = seq_lens['freq_data']
        backbone = _init_backbone(backbone_config)

        head_config.pooling_method = strategy_config.pooling_method
        head_config.downsample_factor = strategy_config.downsample_factor
        if strategy_config.pooling_method != 'conv':
            head_config.conv_in_channel = 0
        else:
            factor = 32
            if strategy_config.conv_data == 'raw':
                if strategy_config.conv_dim == 'time':
                    head_config.conv_in_channel = math.ceil(seq_lens['data'] / factor)
                elif strategy_config.conv_dim == 'channel':
                    head_config.conv_in_channel = math.ceil(n_channels['data'] / factor)
            elif strategy_config.conv_data == 'freq':
                if strategy_config.conv_dim == 'time':
                    head_config.conv_in_channel = math.ceil(seq_lens['freq_data'] / factor)
                elif strategy_config.conv_dim == 'channel':
                    head_config.conv_in_channel = math.ceil(n_channels['freq_data'] / factor)
        head = _init_head(backbone.get_output_size(), head_config)

        return LateFusion(backbone, head, strategy_config)
    elif isinstance(strategy_config, strategy.single_stream.Resnet2DSpanCLSConfig):
        from strategy.single_stream import Resnet2DSpanCLS
        backbone = _init_backbone(backbone_config)
        head = _init_head(backbone.get_output_size(), head_config)
        return Resnet2DSpanCLS(backbone, head, strategy_config)
    elif isinstance(strategy_config, strategy.single_stream.LSTMSpanCLSConfig):
        from strategy.single_stream import LSTMSpanCLS
        if strategy_config.calc_data == 'raw':
            backbone_config.n_channel = n_channels['data']
            backbone_config.seq_len = seq_lens['data']
        elif strategy_config.calc_data == 'freq':
            backbone_config.n_channel = n_channels['freq_data']
            backbone_config.seq_len = seq_lens['freq_data']
        backbone = _init_backbone(backbone_config)
        head = _init_head(backbone.get_output_size(), head_config)
        return LSTMSpanCLS(backbone, head, strategy_config)
    elif isinstance(strategy_config, strategy.single_stream.ViTSpanCLSConfig):
        from strategy.single_stream import ViTSpanCLS
        if strategy_config.calc_data == 'raw':
            backbone_config.n_channel = n_channels['data']  # 90
            backbone_config.seq_len = seq_lens['data']      # 5000
        elif strategy_config.calc_data == 'freq':
            backbone_config.n_channel = n_channels['freq_data']
            backbone_config.seq_len = seq_lens['freq_data']
        strategy_config.patch_size = backbone_config.patch_size
        backbone = _init_backbone(backbone_config)
        head = _init_head(backbone.get_output_size(), head_config)
        return ViTSpanCLS(backbone, head, strategy_config)
    elif isinstance(strategy_config, strategy.dual_stream.DSResnet1DSpanCLSConfig):
        from strategy.dual_stream import DSResnet1DSpanCLS
        for i in range(2):
            if strategy_config.conv_data[i] == 'raw':
                if strategy_config.conv_dim[i] == 'time':
                    backbone_config.in_channel[i] = n_channels['data']
                elif strategy_config.conv_dim[i] == 'channel':
                    backbone_config.in_channel[i] = seq_lens['data']
            elif strategy_config.conv_data[i] == 'freq':
                if strategy_config.conv_dim[i] == 'time':
                    backbone_config.in_channel[i] = n_channels['freq_data']
                elif strategy_config.conv_dim[i] == 'channel':
                    backbone_config.in_channel[i] = seq_lens['freq_data']
        backbone = _init_backbone(backbone_config)
        head = _init_head(backbone.get_output_size(), head_config)
        return DSResnet1DSpanCLS(backbone, head, strategy_config)
    elif isinstance(strategy_config, strategy.dual_stream.DSResnet2DSpanCLSConfig):
        from strategy.dual_stream import DSResnet2DSpanCLS
        backbone = _init_backbone(backbone_config)
        head = _init_head(backbone.get_output_size(), head_config)
        return DSResnet2DSpanCLS(backbone, head, strategy_config)
    elif isinstance(strategy_config, strategy.dual_stream.DSLSTMSpanCLSConfig):
        from strategy.dual_stream import DSLSTMSpanCLS
        backbone_config.seq_len[0] = seq_lens['data']
        backbone_config.n_channel[0] = n_channels['data']
        backbone_config.seq_len[1] = seq_lens['freq_data']
        backbone_config.n_channel[1] = n_channels['freq_data']
        backbone = _init_backbone(backbone_config)
        head = _init_head(backbone.get_output_size(), head_config)
        return DSLSTMSpanCLS(backbone, head, strategy_config)
    elif isinstance(strategy_config, strategy.dual_stream.DSViTSpanCLSConfig):
        from strategy.dual_stream import DSViTSpanCLS
        backbone_config.seq_len[0] = seq_lens['data']
        backbone_config.n_channel[0] = n_channels['data']
        backbone_config.seq_len[1] = seq_lens['freq_data']
        backbone_config.n_channel[1] = n_channels['freq_data']
        strategy_config.patch_size = backbone_config.patch_size
        backbone = _init_backbone(backbone_config)
        head = _init_head(backbone.get_output_size(), head_config)
        return DSViTSpanCLS(backbone, head, strategy_config)
    elif isinstance(strategy_config, strategy.four_stream.FSResnet1DSpanCLSConfig):
        from strategy.four_stream import FSResnet1DSpanCLS
        backbone_config.in_channel = [
            n_channels['data'],
            seq_lens['data'],
            n_channels['freq_data'],
            seq_lens['freq_data'],
        ]
        backbone = _init_backbone(backbone_config)
        head = _init_head(backbone.get_output_size(), head_config)
        return FSResnet1DSpanCLS(backbone, head, strategy_config)
