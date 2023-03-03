# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

import io

import torch.nn
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json


import matplotlib.pyplot as plt
import scipy.fft as fp
from scipy import fftpack

# hook the feature extractor
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx, cls_num):
    # class_idx 二维的，每个图的概率顺序】
    # cls_num 每个图的类别数 一维的
    # generate the class activation maps upsample to 256x256
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    print(f'bz:{bz}, nc={nc}, h={h}, w={w}')    #8*512*7*7
    print(f'weight_softmax.shape{weight_softmax.shape}')
    output_cam = []
    # 每张图
    for i in range(8):      # 8是batchsize
        for idx in cls_num:
            print(f'idx:{idx}')
            # cls_num是图像中所含有的类别总数  idx是按照概率由高到低的tensor
            # softmax 1*512               512*49       1*49
            feature_conv = torch.from_numpy(feature_conv.reshape((bz, nc, h * w)))      #特征图    8*512*7
            cam = torch.matmul(weight_softmax[idx], feature_conv)
            print(f'1cam_img.shape{cam_img.shape}')

            cam = cam.reshape(bz, 20, h, w)     # channel直接写成20
            print(f'1cam_img.shape{cam_img.shape}')
            # 加个for循环处理每张图的CAM
            # 由label获取每张图有几个类别目标
            # 根据概率生成前几
            cam = cam - np.min(cam)  # 减去最小，除以最大，目的是归一化
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            print(f'cam_imgshape{cam_img.shape}')
            output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

class Network(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()
        self.net = models.resnet18(pretrained=True)
        fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(fc_in_features, 20, bias=True)
        finalconv_name = 'layer4'
        self.net._modules.get(finalconv_name).register_forward_hook(hook_feature)

        # get the softmax weight
        params = list(self.net.parameters())
        # print(params[-2])
        self.weight_softmax = torch.squeeze(params[-2].data)
        # print(self.weight_softmax.shape)

    def forward(self, inputs, label):
        # 前向过程，也要label，确定生成几个CAM图
        # print(f'label={label}')
        cls_num = torch.sum(label, dim=1, keepdim=False)
        # print(f'a1={a1}')

        N, C, H, W = inputs.size()
        x = self.net(inputs)
        # print(f'x.shape:{x.shape}')

        h_x = F.softmax(x, dim=1).data.squeeze()

        # print(f'h_x.shape:{h_x.shape}')

        probs, idx = h_x.sort(1, True)
        # print(f'probs:{probs[0]} idx:{idx[0]}')
        # idx = idx.numpy()
        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], self.weight_softmax, idx, cls_num)
        cam = F.interpolate(CAMs, (H, W), mode='bilinear', align_corners=True)
        return cam, x
