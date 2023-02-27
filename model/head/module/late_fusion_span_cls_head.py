import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionSpanCLSHead(nn.Module):
    """
    pooling_method:
        max: 在进行卷积的维度提取最大值特征
        late: 先将进行卷积的特征提取到logits，再取最大
        slow:
        local: 取卷积维度相邻最大特征，再通过head计算logits，最后平均
        conv: 在非卷积维度卷积融合，取完最大值特征后head计算logits
    """
    def __init__(self, hidden_dim, n_class, pooling_method, downsample_factor=2, conv_in_channel=0):
        super(LateFusionSpanCLSHead, self).__init__()

        self.pooling_method = pooling_method
        self.downsample_factor = downsample_factor
        self.conv_in_channel = conv_in_channel
        if self.conv_in_channel != 0:
            self.default_conv_channel = 128
            self.out_conv = nn.Conv1d(self.conv_in_channel, self.default_conv_channel,
                                      kernel_size=3, stride=1, padding=1)
            self.head_1 = nn.Linear(self.default_conv_channel, self.default_conv_channel, bias=False)
            self.head_2 = nn.Linear(self.default_conv_channel, n_class, bias=False)
        else:
            self.head_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.head_2 = nn.Linear(hidden_dim, n_class, bias=False)

        nn.init.xavier_uniform_(self.head_1.weight)
        nn.init.xavier_uniform_(self.head_2.weight)

    def forward(self, features):
        # batch_size, channel, time
        if self.pooling_method == 'max':
            features = F.adaptive_max_pool1d(features, 1).squeeze()
            return self.head_2(self.head_1(features))
        elif self.pooling_method == 'late':
            features = features.permute(0, 2, 1)
            logits = self.head_2(self.head_1(features))
            return F.adaptive_max_pool1d(logits.permute(0, 2, 1), 1).squeeze()
        elif self.pooling_method == 'slow':
            assert features.size(2) % self.downsample_factor == 0
            features = F.adaptive_max_pool1d(features, features.size(2) // self.downsample_factor)
            features = self.head_1(features.permute(0, 2, 1)).permute(0, 2, 1)
            features = F.adaptive_max_pool1d(features, 1).squeeze()
            return self.head_2(features)
        elif self.pooling_method == 'local':
            assert features.size(2) % self.downsample_factor == 0
            features = F.adaptive_max_pool1d(features, features.size(2) // self.downsample_factor)
            return torch.mean(self.head_2(self.head_1(features.permute(0, 2, 1))), dim=1)
        elif self.pooling_method == 'conv':
            features = self.out_conv(features.permute(0, 2, 1))
            features = F.adaptive_max_pool1d(features, 1).squeeze()
            return self.head_2(self.head_1(features))
