import torch.nn as nn

from .module import SimpleSpanCLSHead
from ..model_config import ModelConfig


class HTHISpanCLSConfig(ModelConfig):
    """
        model_name format: hthi_span_cls
    """
    label_n_classes = 12

    def __init__(self, model_name: str):
        super(HTHISpanCLSConfig, self).__init__(model_name)


class HTHISpanCLS(nn.Module):
    def __init__(self, hidden_dim, config: HTHISpanCLSConfig):
        super(HTHISpanCLS, self).__init__()
        self.model_name = config.model_name

        self.label_head = SimpleSpanCLSHead(hidden_dim, config.label_n_classes)

    def forward(self, features):
        return {
            'label': self.label_head(features),
        }

    def get_model_name(self):
        return self.model_name
