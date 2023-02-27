import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelConfig


class Resnet1DSpanCLSConfig(ModelConfig):
    """
        model_name format: resnet1d_span_cls_(conv_data)_(conv_dim)
    """

    def __init__(self, model_name):
        super(Resnet1DSpanCLSConfig, self).__init__(model_name)
        # 卷积的数据: raw|freq
        self.conv_data = 'raw'
        # 卷积的维度: time|channel
        self.conv_dim = 'time'
        # LOSS
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')


class Resnet1DSpanCLS(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, config: Resnet1DSpanCLSConfig):
        super(Resnet1DSpanCLS, self).__init__()
        self.model_name = config.model_name

        self.backbone = backbone
        self.head = head
        self.config = config

    def to_backbone_to_head(self, input):
        batch_data = None
        if self.config.conv_data == 'raw':
            batch_data = input['data']
        elif self.config.conv_data == 'freq':
            batch_data = input['freq_data']

        if self.config.conv_dim == 'channel':
            batch_data = batch_data.permute(0, 2, 1)
        features = self.backbone(batch_data)
        features = torch.mean(features, dim=-1)
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
