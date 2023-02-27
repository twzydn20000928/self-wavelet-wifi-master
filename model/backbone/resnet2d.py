import torch
import torch.nn as nn

from ..model_config import ModelConfig


class ResNet2DConfig(ModelConfig):
    """
        model_name format: resnet2d_(layers)_(conv_dim)
    """
    in_channel = 1
    inplane = 64
    """
    通过调整卷积核的大小，每个block卷积的维度相应的降维
    2: 全2d卷积，全stride=2
    1+2: 前2个block1d卷积；后3个block2d卷积
    2+1: 前2个block2d卷积，后3个block1d卷积
    1+1: 每个block都分别在两个维度1d卷积
    """
    conv_dim = '2'

    def __init__(self, model_name: str):
        super(ResNet2DConfig, self).__init__(model_name)
        _, layers, conv_dim = model_name.split('_')
        self.conv_dim = conv_dim


class BasicConv(nn.Module):
    def __init__(self, in_channel: int,
                 out_channel: int,
                 kernel_size: tuple,
                 stride: tuple,
                 padding: tuple,
                 bias=False,
                 activation=False, ):
        super(BasicConv, self).__init__()

        block = [
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channel),
        ]
        if activation:
            block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, is_2d=True, downsample=None):
        super(BasicBlock, self).__init__()
        if is_2d:
            # 2D
            self.conv1 = BasicConv(inplanes, planes,
                                   kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), activation=True)
            self.conv2 = BasicConv(planes, planes,
                                   kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=False)
        else:
            # 先最后一维卷积，再倒数第二维卷积
            self.conv1 = BasicConv(inplanes, planes,
                                   kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), activation=True)
            self.conv2 = BasicConv(planes, planes,
                                   kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), activation=False)
        self.downsample = downsample
        self.stride = stride

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, is_2d=True, downsample=None):
        super(Bottleneck, self).__init__()
        if is_2d:
            # 2D
            self.conv1 = BasicConv(inplanes, planes,
                                   kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation=True)
            self.conv2 = BasicConv(planes, planes,
                                   kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), activation=True)
            self.conv3 = BasicConv(planes, planes,
                                   kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=True)
            self.conv4 = BasicConv(planes, planes * self.expansion,
                                   kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation=False)
        else:
            # 先最后一维卷积，再倒数第二维卷积
            self.conv1 = BasicConv(inplanes, planes,
                                   kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation=True)
            self.conv2 = BasicConv(planes, planes,
                                   kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), activation=True)
            self.conv3 = BasicConv(planes, planes,
                                   kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), activation=True)
            self.conv4 = BasicConv(planes, planes * self.expansion,
                                   kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation=False)

        self.downsample = downsample
        self.stride = stride

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        out = self.conv4(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2D(nn.Module):
    def __init__(self, block, layers, config: ResNet2DConfig):
        super(ResNet2D, self).__init__()
        self.model_name = config.model_name
        self.config = config

        self.inplane = config.inplane

        if config.conv_dim == '2':
            self.in_conv = BasicConv(config.in_channel, config.inplane,
                                     kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), activation=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(block, 64, layers[0], stride=1, is_2d=True)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, is_2d=True)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, is_2d=True)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_2d=True)
        elif config.conv_dim == '1+1':
            self.in_conv = nn.Sequential(
                BasicConv(config.in_channel, config.inplane,
                          kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), activation=True),
                BasicConv(config.inplane, config.inplane,
                          kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), activation=True)
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(block, 64, layers[0], stride=1, is_2d=False)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, is_2d=False)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, is_2d=False)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_2d=False)
        elif config.conv_dim == '2+1':
            self.in_conv = BasicConv(config.in_channel, config.inplane,
                                     kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), activation=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(block, 64, layers[0], stride=1, is_2d=True)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, is_2d=False)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, is_2d=False)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_2d=False)
        elif config.conv_dim == '1+2':
            self.in_conv = nn.Sequential(
                BasicConv(config.in_channel, config.inplane,
                          kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), activation=True),
                BasicConv(config.inplane, config.inplane,
                          kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), activation=True)
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(block, 64, layers[0], stride=1, is_2d=False)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, is_2d=True)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, is_2d=True)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_2d=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.output_size = 512 * block.expansion

    def _make_layer(self, block, planes, blocks, stride=1, is_2d=True):
        downsample = None

        if stride != 1 or self.inplane != planes * block.expansion:
            downsample = BasicConv(self.inplane, planes * block.expansion,
                                   kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0), activation=False)
        layers = []
        layers.append(block(self.inplane, planes, stride, is_2d, downsample))
        self.inplane = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplane, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.in_conv(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def get_output_size(self):
        return self.output_size

    def get_model_name(self):
        return self.model_name


def resnet2d(config: ResNet2DConfig):
    if config.model_name.startswith('resnet2d_18'):
        return ResNet2D(BasicBlock, [2, 2, 2, 2], config)
    elif config.model_name.startswith('resnet2d_34'):
        return ResNet2D(BasicBlock, [3, 4, 6, 3], config)
    elif config.model_name.startswith('resnet2d_50'):
        return ResNet2D(Bottleneck, [3, 4, 6, 3], config)
    elif config.model_name.startswith('resnet2d_101'):
        return ResNet2D(Bottleneck, [3, 4, 23, 3], config)
