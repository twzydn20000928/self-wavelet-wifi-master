import torch.nn as nn

from .module import LateFusionSpanCLSHead
from ..model_config import ModelConfig


class WiFiARLateFusionSpanCLSConfig(ModelConfig):
    """
        model_name format: wifi_ar_late_fusion_span_cls
    """
    label_n_class = 7
    pooling_method = 'max'
    downsample_factor = 2
    conv_in_channel = 6

    def __init__(self, model_name: str):
        super(WiFiARLateFusionSpanCLSConfig, self).__init__(model_name)


class WiFiARLateFusionSpanCLS(nn.Module):
    def __init__(self, hidden_dim, config: WiFiARLateFusionSpanCLSConfig):
        super(WiFiARLateFusionSpanCLS, self).__init__()
        self.model_name = config.model_name
        self.config = config

        self.label_head = LateFusionSpanCLSHead(hidden_dim, config.label_n_class,
                                                pooling_method=config.pooling_method,
                                                downsample_factor=config.downsample_factor,
                                                conv_in_channel=config.conv_in_channel)

    def forward(self, features):
        return {
            'label': self.label_head(features),
        }

    def get_model_name(self):
        return self.model_name
