import torch
import torch.nn as nn

from .resnet1d import ResNet1DConfig, resnet1d, ResNet1D
from ..model_config import ModelConfig


class DSResNet1DConfig(ModelConfig):
    """
        model_name format: ds_resnet1d_(layers)
    """
    in_channel = [64, 64]
    inplanes = 64

    def __init__(self, model_name: str):
        super(DSResNet1DConfig, self).__init__(model_name)


class DSResNet1D(nn.Module):
    def __init__(self, branch_1: ResNet1D, branch_2: ResNet1D, config: DSResNet1DConfig):
        super(DSResNet1D, self).__init__()
        self.model_name = config.model_name

        self.branch_1 = branch_1
        self.branch_2 = branch_2

        self.output_size = self.branch_1.get_output_size() + self.branch_2.get_output_size()

    def forward(self, batch_data):
        feature_1, feature_2 = self.branch_1(batch_data[0]), self.branch_2(batch_data[1])

        feature_1, feature_2 = torch.mean(feature_1, dim=-1), torch.mean(feature_2, dim=-1)

        return torch.cat([feature_1, feature_2], dim=-1)

    def get_output_size(self):
        return self.output_size

    def get_model_name(self):
        return self.model_name


def ds_resnet1d(config: DSResNet1DConfig):
    _, __, layers = config.model_name.split('_')
    branch_1_config = ResNet1DConfig('resnet1d_%s' % layers)
    branch_2_config = ResNet1DConfig('resnet1d_%s' % layers)
    branch_1_config.in_channel = config.in_channel[0]
    branch_2_config.in_channel = config.in_channel[1]
    return DSResNet1D(
        branch_1=resnet1d(branch_1_config),
        branch_2=resnet1d(branch_2_config),
        config=config,
    )
