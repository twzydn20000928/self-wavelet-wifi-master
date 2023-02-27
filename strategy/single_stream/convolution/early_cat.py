import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelConfig


class EarlyCatConfig(ModelConfig):
    """
        model_name format: early_cat_(conv_data)_(stack_dim)_(stack_num)
    """

    def __init__(self, model_name):
        super(EarlyCatConfig, self).__init__(model_name)
        self.conv_data = 'raw'
        '''
        default: time
        time: (channel, seq_len) -> (channel // stack_num, seq_len * stack_num)，cat 相邻通道到时间维度，conv在时间维度卷积
        channel: (seq_len, channel) -> (seq_len // stack_num, channel // stack_num)，cat 相邻帧到通道维度，conv在通道维度卷积
        '''
        self.stack_dim = "time"
        self.stack_num = 2
        # LOSS
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')


class EarlyCat(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, config: EarlyCatConfig):
        super(EarlyCat, self).__init__()
        self.model_name = config.model_name

        self.backbone = backbone
        self.head = head
        self.config = config

    def _early_cat_transform(self, batch_data):
        if self.config.stack_dim == 'channel':
            batch_data = batch_data.permute(0, 2, 1).contiguous()
        batch_size, dim1, dim2 = batch_data.size()
        assert dim1 % self.config.stack_num == 0
        batch_data = batch_data.view((batch_size,
                                      dim1 // self.config.stack_num,
                                      self.config.stack_num,
                                      dim2))
        batch_data = batch_data.view((batch_size,
                                      dim1 // self.config.stack_num,
                                      self.config.stack_num * dim2))
        return batch_data

    def to_backbone_to_head(self, input):
        batch_data = None
        if self.config.conv_data == 'raw':
            batch_data = input['data']
        elif self.config.conv_data == 'freq':
            batch_data = input['freq_data']
        batch_data = self._early_cat_transform(batch_data)
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
