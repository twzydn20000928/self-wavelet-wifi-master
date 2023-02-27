import torch.nn as nn

from .module import SimpleSpanCLSHead
from ..model_config import ModelConfig


class ARILSpanCLSConfig(ModelConfig):
    """
        model_name format: aril_span_cls
    """
    activity_n_classes = 6
    location_n_classes = 16

    def __init__(self, model_name: str):
        super(ARILSpanCLSConfig, self).__init__(model_name)


class ARILSpanCLS(nn.Module):
    def __init__(self, hidden_dim, config: ARILSpanCLSConfig):
        super(ARILSpanCLS, self).__init__()
        self.model_name = config.model_name

        self.activity_head = SimpleSpanCLSHead(hidden_dim, config.activity_n_classes)
        # self.location_head = SimpleSpanCLSHead(hidden_dim, config.location_n_classes)

    def forward(self, features):
        return {
            'activity': self.activity_head(features),
            # 'location': self.location_head(features),
        }

    def get_model_name(self):
        return self.model_name
