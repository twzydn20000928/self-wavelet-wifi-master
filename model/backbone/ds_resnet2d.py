import torch
import torch.nn as nn

from .resnet2d import ResNet2DConfig, resnet2d, ResNet2D
from ..model_config import ModelConfig


class DSResNet2DConfig(ModelConfig):
    """
        model_name format: ds_resnet2d_(layers)_(conv_dim)
    """

    def __init__(self, model_name: str):
        super(DSResNet2DConfig, self).__init__(model_name)


class DSResNet2D(nn.Module):
    def __init__(self, branch_1: ResNet2D, branch_2: ResNet2D, config: DSResNet2DConfig):
        super(DSResNet2D, self).__init__()
        self.model_name = config.model_name

        self.branch_1 = branch_1
        self.branch_2 = branch_2

        self.output_size = self.branch_1.get_output_size() + self.branch_2.get_output_size()

    def forward(self, batch_data):
        feature_1, feature_2 = self.branch_1(batch_data[0]), self.branch_2(batch_data[1])
        feature_1 = torch.mean(feature_1.view(feature_1.size(0), feature_1.size(1), -1), dim=-1)
        feature_2 = torch.mean(feature_2.view(feature_2.size(0), feature_2.size(1), -1), dim=-1)

        return torch.cat([feature_1, feature_2], dim=-1)

    def get_output_size(self):
        return self.output_size

    def get_model_name(self):
        return self.model_name


def ds_resnet2d(config: DSResNet2DConfig):
    _, __, layers, conv_dim = config.model_name.split('_')
    branch_1_config = ResNet2DConfig('resnet2d_%s_%s' % (layers, conv_dim))
    branch_2_config = ResNet2DConfig('resnet2d_%s_%s' % (layers, conv_dim))
    return DSResNet2D(
        branch_1=resnet2d(branch_1_config),
        branch_2=resnet2d(branch_2_config),
        config=config,
    )
