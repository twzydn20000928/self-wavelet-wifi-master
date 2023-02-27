import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelConfig


class DSViTSpanCLSConfig(ModelConfig):
    """
        model_name format: ds_vit_span_cls
    """

    def __init__(self, model_name):
        super(DSViTSpanCLSConfig, self).__init__(model_name)
        # patch_size
        self.patch_size = [8, 8]
        # LOSS
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')


class DSViTSpanCLS(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, config: DSViTSpanCLSConfig):
        super(DSViTSpanCLS, self).__init__()
        self.model_name = config.model_name

        self.backbone = backbone
        self.head = head
        self.config = config

    def to_backbone_to_head(self, input):
        batch_data = [
            input['data'],
            input['freq_data'],
        ]
        for i in range(2):
            if batch_data[i].size(2) % self.config.patch_size[i] != 0:
                batch_data[i] = F.pad(batch_data[i],
                                      (0, self.config.patch_size[i] - (batch_data[i].size(2) % self.config.patch_size[i])),
                                      'constant', 0)
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
