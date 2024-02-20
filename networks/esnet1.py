from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from numbers import Integral

__all__ = ['ESNet']

acts = {"relu": nn.ReLU(inplace=True),
        "hard_swish": nn.Hardswish(inplace=True)}


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self._batch_norm = nn.BatchNorm2d(out_channels)

        self.act = nn.Identity() if act is None else acts[act]

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self.act(y)
        return y


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #####自适应平均池化
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs)
        return inputs * outputs


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    #x = x.view(batchsize, groups, channels_per_group, height, width)###训练
    x = x.view( groups, channels_per_group, height, width)###推理

    #x = torch.transpose(x, 1, 2).contiguous()###训练
    x = torch.transpose(x, 0, 1).contiguous()##推理

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 act="relu"):
        super(InvertedResidual, self).__init__()
        self._conv_pw = ConvBNLayer(
            in_channels=in_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels)

        self._conv_linear = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)

    def forward(self, inputs):
        x1, x2 = torch.chunk(
            inputs,
            chunks=2,
            dim=1)
        x2 = self._conv_pw(x2)
        x3 = self._conv_dw(x2)
        x3 = torch.cat([x2, x3], axis=1)
        x3 = self._se(x3)
        x3 = self._conv_linear(x3)
        out = torch.cat([x1, x3], axis=1)
        out = channel_shuffle(out, 2)
        return out


class InvertedResidualDS(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 act="relu"):
        super(InvertedResidualDS, self).__init__()

        # branch1
        self._conv_dw_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            act=None)
        self._conv_linear_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        # branch2
        self._conv_pw_2 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=mid_channels // 2,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=mid_channels // 2,
            act=None)
        self._se = SEModule(mid_channels // 2)
        self._conv_linear_2 = ConvBNLayer(
            in_channels=mid_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=act)
        self._conv_dw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            act="hard_swish")
        self._conv_pw_mv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act="hard_swish")

    def forward(self, inputs):
        x1 = self._conv_dw_1(inputs)
        x1 = self._conv_linear_1(x1)
        x2 = self._conv_pw_2(inputs)
        x2 = self._conv_dw_2(x2)
        x2 = self._se(x2)
        x2 = self._conv_linear_2(x2)
        out = torch.cat([x1, x2], axis=1)
        out = self._conv_dw_mv1(out)
        out = self._conv_pw_mv1(out)

        return out


# class FaceQuality(nn.Module):
#     def __init__(self, feature_dim):
#         super(FaceQuality, self).__init__()
#         self.qualtiy = nn.Sequential(
#             nn.Linear(feature_dim, 512, bias=False),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 2, bias=False),
#             nn.Softmax(dim=1)
#         )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x = self.qualtiy(x)
#         return x[:, 0:1]


class ESNet(nn.Module):
    def __init__(self,
                 scale=1.0,
                 act="hard_swish",
                 feature_maps=[4, 11, 14],
                 channel_ratio=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        super(ESNet, self).__init__()
        self.scale = scale
        if isinstance(feature_maps, Integral):
            feature_maps = [feature_maps]
        self.feature_maps = feature_maps
        stage_repeats = [3, 7, 3]

        stage_out_channels = [
            1, 16, make_divisible(32 * scale), make_divisible(64 * scale),
            make_divisible(128 * scale),256
        ]

        self._out_channels = []
        self._feature_idx = 0
        # 1. conv1
        self._conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            act=act)
        self._conv2 = ConvBNLayer(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            act=act)

        self._conv3 = ConvBNLayer(
            in_channels = 32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            act=act)



        # 为我们的输出创建单独的分类器
        self.blurness = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=32 * 14 *14 , out_features=2)###1568
        )
        self.left_eye = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=32* 14 * 14, out_features=6)
        )
        self.right_eye = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=32 * 14 * 14, out_features=6)

        )
        self.mouth = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=32 * 14 * 14, out_features=2)

        )

        self._max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(32 * 16 * 16, 4)
        self._feature_idx += 1
        # self.fq = FaceQuality(256 * 7 * 7)
        # 2. bottleneck sequences
        block_list = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(stage_repeats):
            for i in range(num_repeat):
                channels_scales = channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales),
                    divisor=8)
                if i == 0:
                    block = InvertedResidualDS(
                        in_channels=stage_out_channels[stage_id + 1],
                        mid_channels=mid_c,
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=2,
                        act=act)
                else:
                    block = InvertedResidual(
                        in_channels=stage_out_channels[stage_id + 2],
                        mid_channels=mid_c,
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=1,
                        act=act)
                block_list.append(block)
                arch_idx += 1
                self._feature_idx += 1
                self._update_out_channels(stage_out_channels[stage_id + 2],
                                          self._feature_idx, self.feature_maps)

        self._block_list = nn.ModuleList(block_list)

    def _update_out_channels(self, channel, feature_idx, feature_maps):
        if feature_idx in feature_maps:
            self._out_channels.append(channel)

    def forward(self, inputs):
        y = self._conv1(inputs)
        y = self._max_pool(y)
        outs = []
        for i, inv in enumerate(self._block_list):
            y = inv(y)
            if i + 2 in self.feature_maps:
                outs.append(y)
        # print(outs[0].shape,outs[1].shape,outs[2].shape)
        # 分类
        # print(outs[0].shape)
        cl = outs[0].view(outs[0].size(0), -1)

        # 质量检测
        # print(outs[1].size())
        """ fq=outs[1].view(outs[1].size(0), -1)
        #print(fq.shape)
        fq=self.fq(fq) """
        # print(fq.shape)
        # return fq
        """ return {
        'head0': self.left_eye(cl),
        'fq' : fq} """

        return {
            'blurness': self.blurness(cl),
            'left_eye': self.left_eye(cl),
            'right_eye': self.right_eye(cl),
            'mouth': self.mouth(cl),
        }
        #


import numpy as np


def main():
    dummpy_input = torch.randn(size=(1, 3, 112, 112))
    model = ESNet()
    output = model(dummpy_input)
    # print(output['left_eye'].shape)
    # torch.Size([1, 128, 32, 32]) h/8
    # torch.Size([1, 256, 20, 20]) h/16
    # torch.Size([1, 512, 10, 10])  h/32


if __name__ == "__main__":
    main()