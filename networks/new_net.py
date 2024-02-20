from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter

import math
import numpy as np
import os
import cv2


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

Mobilefacenet_bottleneck_setting_mid = [
    # t, c , n ,s
    [2, 32, 5, 2],
    [4, 64, 1, 2],
    [2, 64, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 1, 1]
]

Mobilenetv2_bottleneck_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class MobileFaceNet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting_mid, channels=3):
        super(MobileFaceNet, self).__init__()


        self.conv1 = ConvBlock(channels, 32, 3, 2, 1)

        self.dw_conv1 = ConvBlock(32, 32, 3, 1, 1, dw=True)

        self.inplanes = 32
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)

        self.linear7 = ConvBlock(512, 512, (8, 8), 1, 0, dw=True, linear=True)
        # self.linear7 = ConvBlock(512, 512, (4, 4), 1, 0, dw=True, linear=True)
        # self.avgpool = nn.AvgPool2d((8, 8))
        # self.linear7 = ConvBlock(512, 512, (1, 1), 1, 0, dw=True, linear=True)

        # self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)
        self.cl1 = nn.Linear(512, 1)
        self.cl2 = nn.Linear(512, 2)
        self.cl3 = nn.Linear(512, 4)
        self.cl4 = nn.Linear(512, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x, train=False):
        x1 = self.conv1(x)
        x2 = self.dw_conv1(x1)
        x3 = self.blocks(x2)
        x4 = self.conv2(x3)
        x5 = self.linear7(x4)
        x5 = x5.view(x5.size(0), -1)
        cl1 = self.cl1(x5)
        cl2 = self.cl2(x5)
        cl3 = self.cl3(x5)
        cl4 = self.cl4(x5)
        return {
            'blurness': cl1,
            'left_eye': cl3,
            'right_eye': cl4,
            'mouth': cl2,
        }
        # return cl1, cl2, cl3, cl4


if __name__ == "__main__":
    # input = torch.rand((1, 3, 64, 64))
    input = torch.rand((1, 3, 128, 128))
    net = MobileFaceNet(channels=3)




    net.eval()

    output = net(input)
    print(output[3].shape)


