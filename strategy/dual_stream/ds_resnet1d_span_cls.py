import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelConfig


class DSResnet1DSpanCLSConfig(ModelConfig):
    """
        model_name format: ds_resnet1d_span_cls_((conv_data)(conv_dim)(conv_data)(conv_dim))
    """

    def __init__(self, model_name):
        super(DSResnet1DSpanCLSConfig, self).__init__(model_name)
        # 卷积的数据: raw|freq
        self.conv_data = [
            'raw', 'raw'
        ]
        # 卷积的维度: time|channel
        self.conv_dim = [
            'time', 'time'
        ]
        # LOSS
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')


class DSResnet1DSpanCLS(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, config: DSResnet1DSpanCLSConfig):
        super(DSResnet1DSpanCLS, self).__init__()
        self.model_name = config.model_name

        self.backbone = backbone
        self.head = head
        self.config = config

    def to_backbone_to_head(self, input):
        batch_data = [None for _ in range(2)]
        for i in range(2):
            if self.config.conv_data[i] == 'raw':
                batch_data[i] = input['data'].clone()
            elif self.config.conv_data[i] == 'freq':
                batch_data[i] = input['freq_data'].clone()

            if self.config.conv_dim[i] == 'channel':
                batch_data[i] = batch_data[i].permute(0, 2, 1)

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
