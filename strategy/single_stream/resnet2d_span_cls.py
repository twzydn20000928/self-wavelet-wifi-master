import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelConfig


class Resnet2DSpanCLSConfig(ModelConfig):
    """
        model_name format: resnet2d_span_cls_(calc_data)
    """

    def __init__(self, model_name):
        super(Resnet2DSpanCLSConfig, self).__init__(model_name)
        # 计算的数据: raw|freq
        self.calc_data = 'raw'
        # LOSS
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')


class Resnet2DSpanCLS(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, config: Resnet2DSpanCLSConfig):
        super(Resnet2DSpanCLS, self).__init__()
        self.model_name = config.model_name

        self.backbone = backbone
        self.head = head
        self.config = config

    def to_backbone_to_head(self, input):
        batch_data = None
        if self.config.calc_data == 'raw':
            batch_data = input['data']
        elif self.config.calc_data == 'freq':
            batch_data = input['freq_data']
        features = self.backbone(batch_data)
        features = torch.mean(features.view(features.size(0), features.size(1), -1), dim=-1)
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
