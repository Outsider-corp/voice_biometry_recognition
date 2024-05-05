import torch.nn as nn
import torch.nn.functional as F


class ClippedReLU(nn.Module):
    def __init__(self, max_value=20):
        super(ClippedReLU, self).__init__()
        self.max_value = max_value

    def forward(self, x):
        return x.clamp(min=0, max=self.max_value)


class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.clipped_relu = ClippedReLU(max_value=20)
        self.bn = nn.BatchNorm2d(filters)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.clipped_relu(self.bn(x))
        x = self.conv(x)
        x = self.bn(x)
        x = self.clipped_relu(x)
        x = self.identity(x)  # Add the residual (skip) connection
        out = self.clipped_relu(x)
        return F.relu(out)


class ConvResBlock(nn.Module):
    def __init__(self, filters: int):
        in_channel = filters // 2 if filters > 64 else 1
        super(ConvResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, filters, kernel_size=5, padding=2, stride=2)
        self.clipped_relu = ClippedReLU(max_value=20)
        self.res_block = ResBlock(filters)
        self.bn = nn.BatchNorm2d(filters)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.clipped_relu(self.bn(x))
        x = self.res_block(x)
        x = self.res_block(x)
        out = self.res_block(x)
        return out


class DeepSpeakerModel(nn.Module):
    def __init__(self, include_softmax=False):
        super(DeepSpeakerModel, self).__init__()
        self.include_softmax = include_softmax

        # Convolution and Residual blocks setup
        self.conv1 = ConvResBlock(64)
        self.conv2 = ConvResBlock(128)
        self.conv3 = ConvResBlock(256)
        self.conv4 = ConvResBlock(512)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense and output layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(512, 512)
        if include_softmax:
            self.dropout = nn.Dropout(0.5)
        else:
            self.norm = lambda x: F.normalize(x, p=2, dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_avg_pool(x)

        # x = x.view(-1, 2048)

        x = self.flatten(x)
        # x = torch.mean(x, dim=1)

        x = self.dense1(x)
        if self.include_softmax:
            x = self.dropout(x)
            x = self.output(x)
            x = F.softmax(x, dim=1)
        else:
            x = self.norm(x)
        return x
