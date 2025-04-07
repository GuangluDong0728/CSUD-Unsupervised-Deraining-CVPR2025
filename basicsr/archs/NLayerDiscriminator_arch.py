#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-18 22:31:45

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from basicsr.utils.registry import ARCH_REGISTRY

import math
import numbers
import torch
from torch import nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import time
from skimage import io

import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
import torch.nn.utils as utils

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv1x1(in_chn,out_chn,bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=1, padding=1, bias = bias)
    return layer

class GaussionSmoothLayer(nn.Module):
    def __init__(self, channel, kernel_size, sigma, dim = 2):
        super(GaussionSmoothLayer, self).__init__()
        kernel_x = cv2.getGaussianKernel(kernel_size, sigma)
        kernel_y = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel_x * kernel_y.T
        self.kernel_data = kernel
        self.groups = channel
        if dim == 1:
            self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                    groups= channel, bias= False)
        elif dim == 2:
            self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                    groups= channel, bias= False)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                    groups= channel, bias= False)
            raise RuntimeError(
                'input dim is not supported !, please check it !'
            )
        self.conv.weight.requires_grad = False
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel))
        self.pad = int((kernel_size - 1) / 2)
    def forward(self, input):
        intdata = input
        intdata = F.pad(intdata, (self.pad, self.pad, self.pad, self.pad), mode='reflect')

        output = self.conv(intdata)
        return output

class LapLasGradient(nn.Module):
    def __init__(self, indim, outdim):
        super(LapLasGradient, self).__init__()
        # @ define the sobel filter for x and y axis
        kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]
            ]
        )
        kernel2 = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]
            ]
        )
        kernel3 = torch.tensor(
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]
            ]
        )
        kernel4 = torch.tensor(
            [[1, 1, 1],
             [1, -8, 1],
             [1, 1, 1]
            ]
        )
        self.conv = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)
        self.conv.weight.data.copy_(kernel4)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        grad = self.conv(x)
        return grad



class GradientLoss(nn.Module):
    def __init__(self, indim, outdim):
        super(GradientLoss, self).__init__()
        # @ define the sobel filter for x and y axis
        x_kernel = torch.tensor(
            [ [1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]
            ]
        )
        y_kernel = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]
            ]
        )
        self.conv_x = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)
        self.conv_y = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)

        self.conv_x.weight.data.copy_(x_kernel)
        self.conv_y.weight.data.copy_(y_kernel)
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        gradient = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
        return gradient
class GradientLoss_v1(nn.Module):
    def __init__(self, indim, outdim):
        super(GradientLoss_v1, self).__init__()
        # @ define the sobel filter for x and y axis
        x_kernel = torch.tensor(
            [[0, -1, 0],
             [0, 0, 0],
             [0, 1, 0]]
        )
        y_kernel = torch.tensor(
            [[0, 0, 0],
             [-1, 0, 1],
             [0, 0, 0]]
        )
        self.conv_x = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)
        self.conv_y = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)

        self.conv_x.weight.data.copy_(x_kernel)
        self.conv_y.weight.data.copy_(y_kernel)
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        gradient = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
        return gradient
def Norm(x):
    Max_item = torch.max(x)
    Min_item = torch.min(x)
    return (x-Min_item)/(Max_item-Min_item)

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x







def main(file1,file2):
    mat = cv2.imread(file1)
    nmat = cv2.imread(file2)
    tensor = torch.from_numpy(mat).float()
    tensor1 = torch.from_numpy(nmat).float()

    # 11, 17, 25, 50
    blurkernel = GaussionSmoothLayer(3, 11, 50)
    gradloss = GradientLoss(3, 3)


    tensor = tensor.permute(2, 0, 1)
    tensor = torch.unsqueeze(tensor, dim = 0)

    tensor1 = tensor1.permute(2, 0, 1)
    tensor1 = torch.unsqueeze(tensor1, dim = 0)

    out = blurkernel(tensor)
    out1 = blurkernel(tensor1)

    loss = gradloss(out)
    loss1 = gradloss(out1)

    out = out.permute(0, 2, 3, 1).int()
    out = out.numpy().squeeze().astype(np.uint8)

    out1 = out1.permute(0, 2, 3, 1).int()
    out1 = out1.numpy().squeeze().astype(np.uint8)

    cv2.imshow("1", out)
    cv2.imshow("2", out1)
    cv2.waitKey(0)

#   \
#
def testPIL(file1, file2):
    transform = transforms.Compose([
                                    transforms.ToTensor()
                                    ])
    image11 = transform(Image.open(file1).convert('RGB')).unsqueeze(0)
    image22 = transform(Image.open(file2).convert('RGB')).unsqueeze(0)
    # image1 = image11 - F.avg_pool2d(
    #     F.pad(image11, (2, 2, 2, 2), mode='reflect'), 5, 1, padding=0)

    blurkernel = GaussionSmoothLayer(3, 15, 25)
    # blurkernel = Get_gradient()
    # blurkerne2 = GradientLoss_v1(3,3)
    # image1 = image11-blurkernel(image11)
    image2 = blurkernel(image11)


    # # print('自定义花费时间为：{:.8f}s'.format(time.time() - t1))
    # image1 = Norm(image1)
    # image2 = Norm(image2)
    # print(F.mse_loss(image11, image22+image1))


    image2 = image2.numpy().squeeze()

    #
    #
    #
    image2 = np.transpose(image2, (1, 2, 0))

    io.imsave('NOI_SRGB_%d_%d.png' % (1, 1), np.uint8(np.round(image2*255)))


    # image2 = ((image22+image1).clamp_(0,1)).numpy().squeeze()

    # image2 = np.transpose(image2, (1,2,0))
    #
    # img3 = np.transpose(img3, (1, 2, 0))



    # plt.figure('1')
    # plt.imshow(img_as_ubyte(image11), interpolation='nearest')
    # plt.figure('2')
    # plt.imshow(img_as_ubyte(image2), interpolation='nearest')
    #
    # plt.figure('3')
    # # plt.imshow(img_as_ubyte(img3), interpolation='nearest')
    # plt.show()

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = 0.0001 * (0.5 ** ((epoch) //40))
    # lr = opt['lr']
    return str(lr)
if __name__ == "__main__":
    # for i in range(0,400):
    #     print("第{:0>3d}epoch的学习率是：".format(i)+adjust_learning_rate(i))
    file2 = './figs/NOISY_SRGB_0_0.png'
    file1 = './figs/NOISY_SRGB_0_0.png'
    # # main(file1, file2)
    testPIL(file1, file2)





class DiscriminatorLinear(nn.Module):
    def __init__(self, in_chn, ndf=64, slope=0.2):
        '''
        ndf: number of filters
        '''
        super(DiscriminatorLinear, self).__init__()
        self.ndf = ndf
        # input is N x C x 128 x 128
        main_module = [conv_down(in_chn, ndf, bias=False),
                       nn.LeakyReLU(slope, inplace=True)]
        # state size: N x ndf x 64 x 64
        main_module.append(conv_down(ndf, ndf * 2, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*2) x 32 x 32
        main_module.append(conv_down(ndf * 2, ndf * 4, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*4) x 16 x 16
        main_module.append(conv_down(ndf * 4, ndf * 8, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*8) x 8 x 8
        main_module.append(conv_down(ndf * 8, ndf * 16, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*16) x 4 x 4
        main_module.append(nn.Conv2d(ndf * 16, ndf * 32, 4, stride=1, padding=0, bias=False))
        main_module.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*32) x 1 x 1
        self.main = nn.Sequential(*main_module)
        self.output = nn.Linear(ndf * 32, 1)

        self._initialize()

    def forward(self, x):
        x = torch.cat([x, x - F.avg_pool2d(
            F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, 1, padding=0)], dim=1)
        feature = self.main(x)
        feature = feature.view(-1, self.ndf * 32)
        out = self.output(feature)
        return out.view(-1)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.normal_(m.weight.data, 0., 0.02)
                if not m.bias is None:
                    init.constant_(m.bias, 0)


class _NetD(nn.Module):
    def __init__(self, stride=1):
        super(_NetD, self).__init__()

        self.Gas = GaussionSmoothLayer(3, 15, 9)

        self.features = nn.Sequential(

            # input is (3) x 96 x 96
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=4, stride=stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 96 x 96
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 96 x 96
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 48 x 48
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 48 x 48
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, input):
        input = torch.cat([input, input - self.Gas(input)], dim=1)

        # input = torch.cat([input, input - F.avg_pool2d(
        #     F.pad(input, (1, 1, 1, 1), mode='reflect'), 3, 1, padding=0)], dim=1)

        out = self.features(input)
        return out  # self.sigmoid(out)#.view(-1, 1).squeeze(1)


class Discriminator1(nn.Module):
    def __init__(self, num_conv_block=4):
        super(Discriminator1, self).__init__()

        block = []

        in_channels = 6
        out_channels = 64

        for _ in range(num_conv_block):
            block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, out_channels, 3),
                      nn.LeakyReLU(),
                      nn.BatchNorm2d(out_channels)]
            in_channels = out_channels

            block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, out_channels, 3, 2),
                      nn.LeakyReLU()]
            out_channels *= 2

        out_channels //= 2
        in_channels = out_channels

        block += [nn.Conv2d(in_channels, out_channels, 3),
                  nn.LeakyReLU(0.2),
                  nn.Conv2d(out_channels, out_channels, 3)]

        self.feature_extraction = nn.Sequential(*block)

        self.avgpool = nn.AdaptiveAvgPool2d((512, 512))

        self.classification = nn.Sequential(
            nn.Linear(8192, 100),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = torch.cat([x, x - F.avg_pool2d(x, 3, 1, padding=1)], dim=1)

        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x


class VGGStyleDiscriminator128(nn.Module):
    """VGG style discriminator with input size 128 x 128.

    It is used to train SRGAN and ESRGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch=6, num_feat=64):
        super(VGGStyleDiscriminator128, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(
            num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(
            num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(
            num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(
            num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(
            num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == 128 and x.size(3) == 128, (
            f'Input spatial size must be 128x128, '
            f'but received {x.size()}.')
        x = torch.cat([x, x - F.avg_pool2d(x, 3, 1, padding=1)], dim=1)

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(
            self.conv0_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(
            self.conv1_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(
            self.conv2_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(
            self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(
            self.conv4_1(feat)))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out


import functools


@ARCH_REGISTRY.register()
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        # self.Gas = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.Gas = GaussionSmoothLayer(3, 15, 9)
        # kernel = 15
        # self.k = kernel // 2

    def forward(self, input):
        """Standard forward."""
        input = torch.cat([input, input - self.Gas(input)], dim=1)
        # input = input
        # input = torch.cat([input, input - F.avg_pool2d(
        #     F.pad(input, (self.k, self.k, self.k, self.k), mode='reflect'), 15, 1, padding=0)], dim=1)

        return self.model(input)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
# input = torch.rand(1,3,128,128).cuda()
# Net = NLayerDiscriminator(6).cuda()
# print_network(Net)
# print(Net(input).size())
