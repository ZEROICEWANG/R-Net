import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


class C_attention(nn.Module):

    def __init__(self, channels=32):
        super(C_attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c).contiguous()
        y = self.fc(y).view(b, c, 1, 1).contiguous()
        x = torch.mul(y, x)
        x = self.out_conv(x)
        return x


class Self_Attention(nn.Module):
    def __init__(self, feature_size=88):
        super(Self_Attention, self).__init__()
        self.pooling = nn.AdaptiveAvgPool3d((1, feature_size, feature_size))
        self.linear = nn.Sequential(nn.Linear(feature_size ** 2, (feature_size ** 2) // 8),
                                    nn.ReLU(inplace=True),
                                    nn.Linear((feature_size ** 2) // 8, feature_size ** 2),
                                    nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, 1, c, h, w).contiguous()
        x = self.pooling(x).view(b, h * w).contiguous()
        x = self.linear(x)
        x = x.view(b, 1, h, w)
        return x


'''class Self_Attention(nn.Module):
    def __init__(self, feature_size=88):
        super(Self_Attention, self).__init__()
        self.pooling1 = nn.AdaptiveAvgPool3d((1, feature_size, feature_size))
        self.pooling2 = nn.AdaptiveMaxPool3d((1, feature_size, feature_size))
        self.in_conv = nn.Sequential(nn.Conv2d(2, 1, kernel_size=(3, 3), padding=1),
                                     nn.ReLU(inplace=True))
        self.linear = nn.Sequential(nn.Linear(feature_size ** 2, (feature_size ** 2) // 8),
                                    nn.ReLU(inplace=True),
                                    nn.Linear((feature_size ** 2) // 8, feature_size ** 2),
                                    nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1),
                                      nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, 1, c, h, w)
        x1 = self.pooling1(x).view(b, 1, h, w)
        x2 = self.pooling2(x).view(b, 1, h, w)
        x = torch.cat((x1, x2), dim=1)
        x = self.in_conv(x).view(b, h * w)
        x = self.linear(x)
        x = x.view(b, 1, h, w)
        x = self.out_conv(x)
        return x'''


class dense_aggregation(nn.Module):

    def __init__(self, inchannels=32, feature_num=4, up_num=2):
        super(dense_aggregation, self).__init__()
        self.conv_upsample = []
        for i in range(up_num):
            self.conv_upsample.append(
                nn.Sequential(nn.Conv2d(inchannels * (i + 1), inchannels * (i + 1), (3, 3), padding=1),
                              nn.ReLU(inplace=True)))
        self.conv_upsample1 = nn.ModuleList(self.conv_upsample)

        self.conv_poolsample = []
        for i in range(feature_num - up_num - 1):
            self.conv_poolsample.append(
                nn.Sequential(nn.Conv2d(inchannels * (i + 1), inchannels * (i + 1), (3, 3), padding=1, stride=(2, 2),
                                        groups=inchannels * (i + 1)),
                              nn.Conv2d(inchannels * (i + 1), inchannels * (i + 1), (3, 3), padding=1),
                              nn.ReLU(inplace=True)))
        self.conv_poolsample1 = nn.ModuleList(self.conv_poolsample)

        self.upconv = nn.Conv2d(inchannels, inchannels, 3, padding=1)
        self.c_attention = C_attention(inchannels * feature_num)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels * feature_num, inchannels * feature_num, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(inchannels * feature_num),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=inchannels * feature_num, out_channels=inchannels, kernel_size=(1, 1)),
            nn.BatchNorm2d(inchannels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))

        self.feature_num = feature_num

    def forward(self, pool_x, x, up_x, mask):
        counter = 0

        while len(up_x) >= 2:
            x1 = up_x.pop(-1)
            x1 = F.interpolate(x1, size=up_x[-1].size()[2:], mode='bilinear', align_corners=False)
            x1 = self.conv_upsample1[counter](x1)
            up_x[-1] = torch.cat((x1, up_x[-1]), dim=1)
            counter += 1
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear', align_corners=False)
        mask = self.upconv(mask)
        mask = torch.cat(tuple([mask for i in range(self.feature_num)]), dim=1)
        counter = 0
        while len(pool_x) >= 2:
            x1 = pool_x.pop(0)
            x1 = self.conv_poolsample1[counter](x1)
            pool_x[0] = torch.cat((x1, pool_x[0]), dim=1)
            counter += 1
        if len(pool_x) == 1 and len(up_x) == 1:
            up_x[0] = F.interpolate(up_x[0], x.size()[2:], mode='bilinear', align_corners=False)
            x_sum = torch.cat((self.conv_poolsample1[-1](pool_x[0]),
                               x,
                               self.conv_upsample1[-1](up_x[0])), dim=1)
        elif len(pool_x) == 0:
            up_x[0] = F.interpolate(up_x[0], x.size()[2:], mode='bilinear', align_corners=False)
            x_sum = torch.cat((x, self.conv_upsample1[-1](up_x[0])), dim=1)
        else:
            x_sum = torch.cat((self.conv_poolsample1[-1](pool_x[0]), x), dim=1)
        x_sum = x_sum * mask
        x_sum = self.c_attention(x_sum)
        x = self.conv(x_sum)
        return x
