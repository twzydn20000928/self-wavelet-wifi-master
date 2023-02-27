import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelConfig


class DSResnet2DSpanCLSConfig(ModelConfig):
    """
        model_name format: ds_resnet2d_span_cls
    """

    def __init__(self, model_name):
        super(DSResnet2DSpanCLSConfig, self).__init__(model_name)
        # LOSS
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')


class DSResnet2DSpanCLS(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, config: DSResnet2DSpanCLSConfig):
        super(DSResnet2DSpanCLS, self).__init__()
        self.model_name = config.model_name

        self.backbone = backbone
        self.head = head
        self.config = config

    def to_backbone_to_head(self, input):
        batch_data = [
            input['data'],
            input['freq_data'],
        ]
        features = self.backbone(batch_data)
        logits = self.head(features)
        return logits

    def forward(self, input):
        logits = self.to_backbone_to_head(input)
        loss = None
        for key in logits.keys():
            if loss is None:
                loss = self.config.loss_fn(logits[key], input[key])
            else:
                loss = loss + self.config.loss_fn(logits[key], input[key])
        return loss

    def predict(self, input):
        logits = self.to_backbone_to_head(input)
        result = {}
        for key in logits.keys():
            result[key] = F.softmax(logits[key], dim=-1)
        return result

    def get_model_name(self):
        return self.model_name
