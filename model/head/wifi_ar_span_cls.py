import torch.nn as nn

from .module import SimpleSpanCLSHead
from ..model_config import ModelConfig


class WiFiARSpanCLSConfig(ModelConfig):
    """
        model_name format: wifi_ar_span_cls
    """
    label_n_classes = 7

    def __init__(self, model_name: str):
        super(WiFiARSpanCLSConfig, self).__init__(model_name)


class WiFiARSpanCLS(nn.Module):
    def __init__(self, hidden_dim, config: WiFiARSpanCLSConfig):
        super(WiFiARSpanCLS, self).__init__()
        self.model_name = config.model_name

        self.label_head = SimpleSpanCLSHead(hidden_dim, config.label_n_classes)

    def forward(self, features):
        return {
            'label': self.label_head(features),
        }

    def get_model_name(self):
        return self.model_name
