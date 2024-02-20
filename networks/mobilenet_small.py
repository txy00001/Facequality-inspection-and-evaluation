from torch import nn
import torch
import torch.nn.functional as F

import math
import numpy as np
import os

class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            #pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            #dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            #pw-linear
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



Mobilefacenet_bottleneck_setting_small = [
    # t, c , n ,s
    [2, 32, 2, 2],
    [2, 32, 1, 2],
    [2, 64, 1, 2],
    [1, 128, 2, 1],
]

Mobilefacenet_bottleneck_setting_small_new = [
    # t, c , n ,s
    [2, 32, 2, 2],
    [2, 32, 1, 2],
    [2, 64, 1, 2],
    [1, 64, 2, 1],


]



class MobileFaceNet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting_small_new, channels=3):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(channels, 16, 3, 2, 1)
        self.dw_conv1 = ConvBlock(16, 32, 3, 1, 1, dw=True)

        self.inplanes = 32
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(64, 128, 1, 1, 0)

        self.linear7 = ConvBlock(128, 128, (8, 8), 1, 0, dw=True, linear=True)
        self.cl = nn.Linear(128, 1)

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

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.dw_conv1(x1)
        x3 = self.blocks(x2)
        x4 = self.conv2(x3)
        x5 = self.linear7(x4)
        x5 = x5.view(x5.size(0), -1)
        cl = self.cl(x5)
        return {  ######训练
            'blurness': cl}

# def test(net):
#     import cv2
#     image1 = cv2.imread("./test/Alberto_Ruiz_Gallardon_0001.jpg")
#     image1 = cv2.resize(image1, (128, 128))
#     image1 = torch.from_numpy(image1[..., ::-1].transpose(2, 0, 1).copy()).unsqueeze(0).float()
#     image1 = (image1 / 127.5) - 1.0
#     x = net(input)
#     x = x.detach().numpy()
#     print(x)
#     exit()


if __name__ == "__main__":

    input = torch.rand((1, 3, 128, 128))
    net = MobileFaceNet(channels=3)

    # print(net)
    from thop import profile
    flops, params = profile(net, inputs=(input, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # exit()

    # ckpt = torch.load("./model/model_small/155.ckpt", map_location=torch.device('cpu'))
    # net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    # test(net)

    # model_name = "mobilenet_small_blurness_128x128"
    # torch.onnx.export(net, input, './onnx/'+model_name+".onnx", verbose=False,
    #     input_names=['input'],
    #     output_names=['output'],
    #     # dynamic_axes = {'input': {0: 'batch'},
    #     #                 'output': {0: 'batch'}
    #     #                 } if opt.dynamic else None,
    #     opset_version=11
    #     )
    

