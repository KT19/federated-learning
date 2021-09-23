#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReLU(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBnReLU, self).__init__()
        self.layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, 1, 1),
        nn.BatchNorm2d(out_c),
        nn.ReLU())

    def forward(self, z):
        return self.layer(z)

class Model(nn.Module):
    def __init__(self, input_c, cls_num):
        super(Model, self).__init__()

        self.features = nn.Sequential(ConvBnReLU(input_c, 64),
        ConvBnReLU(64, 64),
        nn.MaxPool2d(2, 2),
        ConvBnReLU(64, 128),
        ConvBnReLU(128, 128),
        nn.MaxPool2d(2, 2),
        ConvBnReLU(128, 256),
        ConvBnReLU(256, 256),
        ConvBnReLU(256, 256),
        nn.MaxPool2d(2, 2),
        ConvBnReLU(256, 512),
        ConvBnReLU(512, 512),
        ConvBnReLU(512, 512),
        nn.MaxPool2d(2, 2),
        ConvBnReLU(512, 512),
        ConvBnReLU(512, 512),
        ConvBnReLU(512, 512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, cls_num)

    def forward(self, x):
        h = self.features(x)
        h = self.gap(h)
        h = h.view(h.size(0), -1)
        o = self.linear(h)

        return o
