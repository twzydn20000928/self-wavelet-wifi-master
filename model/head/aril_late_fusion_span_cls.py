import torch.nn as nn

from .module import LateFusionSpanCLSHead
from ..model_config import ModelConfig


class ARILLateFusionSpanCLSConfig(ModelConfig):
    """
        model_name format: aril_late_fusion_span_cls
    """
    activity_n_class = 6
    location_n_class = 16
    pooling_method = 'max'
    downsample_factor = 2
    conv_in_channel = 6

    def __init__(self, model_name: str):
        super(ARILLateFusionSpanCLSConfig, self).__init__(model_name)


class ARILLateFusionSpanCLS(nn.Module):
    def __init__(self, hidden_dim, config: ARILLateFusionSpanCLSConfig):
        super(ARILLateFusionSpanCLS, self).__init__()
        self.model_name = config.model_name
        self.config = config

        self.activity_head = LateFusionSpanCLSHead(hidden_dim, config.activity_n_class,
                                                   pooling_method=config.pooling_method,
                                                   downsample_factor=config.downsample_factor,
                                                   conv_in_channel=config.conv_in_channel)
        self.location_head = LateFusionSpanCLSHead(hidden_dim, config.location_n_class,
                                                   pooling_method=config.pooling_method,
                                                   downsample_factor=config.downsample_factor,
                                                   conv_in_channel=config.conv_in_channel)

    def forward(self, features):
        return {
            'activity': self.activity_head(features),
            'location': self.location_head(features),
        }

    def get_model_name(self):
        return self.model_name
