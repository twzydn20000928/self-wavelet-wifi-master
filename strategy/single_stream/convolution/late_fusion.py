import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelConfig


class LateFusionConfig(ModelConfig):
    """
        model_name format: late_fusion_(conv_data)_(conv_dim)_(pooling_method)_(downsample_factor)
    """

    def __init__(self, model_name):
        super(LateFusionConfig, self).__init__(model_name)
        # 卷积的数据: raw|freq
        self.conv_data = 'raw'
        # 卷积操作轴
        self.conv_dim = 'time'
        # 融合方法
        self.pooling_method = 'conv'
        # 下采样倍数
        self.downsample_factor = 2
        # LOSS
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')


class LateFusion(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, config: LateFusionConfig):
        super(LateFusion, self).__init__()
        self.model_name = config.model_name

        self.backbone = backbone
        self.head = head
        self.config = config

    def _transform(self, batch_data):
        if self.config.conv_dim == 'time':
            return batch_data
        elif self.config.conv_dim == 'channel':
            return batch_data.permute(0, 2, 1).contiguous()

    def to_backbone_to_head(self, input):
        batch_data = None
        if self.config.conv_data == 'raw':
            batch_data = input['data']
        elif self.config.conv_data == 'freq':
            batch_data = input['freq_data']
        batch_data = self._transform(batch_data)
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
