import torch
import torch.nn as nn
from typing import List

from .resnet1d import ResNet1DConfig, resnet1d, ResNet1D
from ..model_config import ModelConfig


class FSResNet1DConfig(ModelConfig):
    """
        model_name format: fs_resnet1d_(layers)
    """
    stream_num = 4
    in_channel = [64, 64, 64, 64]
    inplane = 64

    def __init__(self, model_name: str):
        super(FSResNet1DConfig, self).__init__(model_name)


class FSResNet1D(nn.Module):
    def __init__(self, branches: List[ResNet1D], config: FSResNet1DConfig):
        super(FSResNet1D, self).__init__()
        self.model_name = config.model_name

        self.stream_nums = config.stream_num
        self.branches = nn.ModuleList(branches)

        self.output_size = 0
        for branch in branches:
            self.output_size += branch.get_output_size()

    def forward(self, batch_data):
        features = []
        for i in range(self.stream_nums):
            features.append(self.branches[i](batch_data[i]))

        for i in range(self.stream_nums):
            features[i] = torch.mean(features[i], dim=-1)

        return torch.cat(features, dim=-1)

    def get_output_size(self):
        return self.output_size

    def get_model_name(self):
        return self.model_name


def fs_resnet1d(config: FSResNet1DConfig):
    # fs_resnet1d_18
    _, __, layers = config.model_name.split('_')
    branches_config = []
    for i in range(config.stream_num):
        branches_config.append(ResNet1DConfig('resnet1d_%s' % layers))
    for i in range(config.stream_num):
        branches_config[i].in_channel = config.in_channel[i]
    return FSResNet1D(
        [resnet1d(branches_config[i]) for i in range(config.stream_num)],
        config=config,
    )
