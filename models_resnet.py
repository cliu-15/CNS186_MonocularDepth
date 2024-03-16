import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import pandas as pd
import csv, itertools

from data_loader import KittiLoader
from loss_original import MonodepthLoss
from transforms_original import image_transforms


## Functions and classes for base functions (resnet, VGG)
def conv(in_layers, out_layers, kernel_size, stride, act_fn=nn.ELU(inplace=True)):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    padding = (p, p, p, p)
    layers = [nn.ZeroPad2d(padding),
              nn.Conv2d(in_layers, out_layers, kernel_size=kernel_size, stride=stride),
              nn.BatchNorm2d(out_layers)]
    if act_fn != None:  # changed to original
        layers.append(act_fn)
    return nn.Sequential(*layers)


def max_pool(kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    padding = (p, p, p, p)
    layers = [nn.ZeroPad2d(padding),
              nn.MaxPool2d(kernel_size, stride=2)]
    return nn.Sequential(*layers)


def res_block(in_layers, out_layers, n_blocks, stride):
    layers = [res_conv(in_layers, out_layers, stride)]
    for i in range(1, n_blocks - 1):
        layers.append(res_conv(out_layers * 4, out_layers, 1))
    layers.append(res_conv(out_layers * 4, out_layers, 1))  # changed to match original
    return nn.Sequential(*layers)


class res_conv(nn.Module):
    def __init__(self, in_layers, out_layers, stride):
        super(res_conv, self).__init__()
        self.in_layers = in_layers
        self.out_layers = out_layers
        self.stride = stride
        self.conv1 = conv(in_layers, out_layers, 1, 1)
        self.conv2 = conv(out_layers, out_layers, 3, stride)
        self.conv3 = nn.Conv2d(out_layers, out_layers * 4, kernel_size = 1, stride = 1)
        self.conv4 = nn.Conv2d(in_layers, out_layers * 4, kernel_size = 1, stride=stride)
        #self.conv3 = conv(out_layers, out_layers * 4, 1, 1, None)  # changed to match original
        #self.conv4 = conv(in_layers, out_layers * 4, 1, stride, None)  # changed to match original
        self.normalize = nn.BatchNorm2d(out_layers * 4)

    def forward(self, x):
        do_proj = x.size()[1] != self.out_layers or self.stride == 2
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return F.elu(self.normalize(x3 + shortcut), inplace=True)

def upconv(in_layers, out_layers, kernel_size, scale):
    layers = [nn.Upsample(scale_factor=scale),
              conv(in_layers, out_layers, kernel_size, 1)
              ]
    return nn.Sequential(*layers)


def deconv(in_layers, out_layers, kernel_size, scale):  # in original
    layers = [nn.ZeroPad2d((1, 1, 1, 1)),
              nn.ConvTranspose2d(in_layers, out_layers, kernel_size, stride, dilation=scale)
              ]
    return nn.Sequential(*layers)


def conv_block(in_layers, out_layers, kernel_size):  # for VGG
    layers = [conv(in_layers, out_layers, kernel_size, 1),
              conv(out_layers, out_layers, kernel_size, 2)
              ]
    return nn.Sequential(*layers)


class get_disp(nn.Module):
    def __init__(self, in_layers):
        super(get_disp, self).__init__()
        #self.conv1 = conv(in_layers, 2, 3, 1, act_fn=nn.Sigmoid())
        self.conv1 = nn.Conv2d(in_layers, 2, kernel_size = 3, stride = 1)
        self.normalize = nn.BatchNorm2d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(F.pad(x, (1,1,1,1)))
        x2 = self.normalize(x1)
        return 0.3 * self.sigmoid(x2)


class Resnet50_md(nn.Module):
    def __init__(self, in_layers):
        super(Resnet50_md, self).__init__()

        # Encoder
        self.conv1 = conv(in_layers, 64, 7, 2)
        self.pool1 = max_pool(3)
        self.conv2 = res_block(64, 64, 3, 2)
        self.conv3 = res_block(256, 128, 4, 2)
        self.conv4 = res_block(512, 256, 6, 2)
        self.conv5 = res_block(1024, 512, 3, 2)

        # Decoder
        self.upconv6 = upconv(2048, 512, 3, 2)
        self.iconv6 = conv(1024 + 512, 512, 3, 1)

        self.upconv5 = upconv(512, 256, 3, 2)
        self.iconv5 = conv(512 + 256, 256, 3, 1)

        self.upconv4 = upconv(256, 128, 3, 2)
        self.iconv4 = conv(256 + 128, 128, 3, 1)
        self.disp4 = get_disp(128)
        self.udisp4 = nn.Upsample(scale_factor=2)

        self.upconv3 = upconv(128, 64, 3, 2)
        self.iconv3 = conv(64 + 64 + 2, 64, 3, 1)
        self.disp3 = get_disp(64)
        self.udisp3 = nn.Upsample(scale_factor=2)

        self.upconv2 = upconv(64, 32, 3, 2)
        self.iconv2 = conv(32 + 64 + 2, 32, 3, 1)
        self.disp2 = get_disp(32)
        self.udisp2 = nn.Upsample(scale_factor=2)

        self.upconv1 = upconv(32, 16, 3, 2)
        self.iconv1 = conv(16 + 2, 16, 3, 1)
        self.disp1 = get_disp(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Skips
        skip1 = conv1
        skip2 = pool1
        skip3 = conv2
        skip4 = conv3
        skip5 = conv4

        # Decoder
        upconv6 = self.upconv6(conv5)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        disp4 = self.disp4(iconv4)
        udisp4 = self.udisp4(disp4)

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, udisp4), 1)
        iconv3 = self.iconv3(concat3)
        disp3 = self.disp3(iconv3)
        udisp3 = self.udisp3(disp3)

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, udisp3), 1)
        iconv2 = self.iconv2(concat2)
        disp2 = self.disp2(iconv2)
        udisp2 = self.udisp2(disp2)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, udisp2), 1)
        iconv1 = self.iconv1(concat1)
        disp1 = self.disp1(iconv1)

        return disp1, disp2, disp3, disp4


## Functions and classes for model training/testing
def post_process_disparity(disp):
    # Taken from paper's Github
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def create_model(model_name, input_channels):
    if model_name == 'resnet':
        model = ResNetModel(input_channels)
    # elif model_name == 'vgg':
    #     model = VGG(input_channels)
    return model