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
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0)
        self.clipped_relu = ClippedReLU(max_value=20)
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)
        # self.identity = nn.Identity()

        # self.projection = nn.Sequential(
        #     nn.Conv2d(filters, filters, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(filters)
        # )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.clipped_relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.identity(out)  # Add the residual (skip) connection
        identity = self.projection(identity)
        out += identity
        out = self.clipped_relu(out)
        return out


class ConvResBlock(nn.Module):
    def __init__(self, filters: int):
        in_channel = filters // 2 if filters > 64 else 1
        super(ConvResBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, filters, kernel_size=5, padding=2, stride=2)
        self.clipped_relu = ClippedReLU(max_value=20)
        self.res_block1 = ResBlock(filters)
        self.res_block2 = ResBlock(filters)
        self.res_block3 = ResBlock(filters)
        self.bn = nn.BatchNorm2d(filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.clipped_relu(self.bn(x))
        x = self.res_block1(x)
        # x = self.clipped_relu(self.bn(x))
        x = self.res_block2(x)
        # x = self.clipped_relu(self.bn(x))
        out = self.res_block3(x)
        # out = self.clipped_relu(self.bn(out))
        return out


class DeepSpeakerModel(nn.Module):
    def __init__(self, include_softmax: bool = False, num_classes: int = 100):
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
            self.classifier = nn.Linear(512, num_classes)
        else:
            self.norm = lambda x: F.normalize(x, p=2, dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_avg_pool(x)

        x = self.flatten(x)
        x = self.dense1(x)

        if self.include_softmax:
            x = self.dropout(x)
            x = self.classifier(x)
            x = F.softmax(x, dim=1)
        else:
            x = self.norm(x)
        return x
