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

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    print(f'bz:{bz}, nc={nc}, h={h}, w={w}')
    output_cam = []
    for idx in class_idx:
        # idx对应的那个概率最大的类，所以是1*512
        # softmax 1*512               512*49       1*49
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))  # feature conv 1*512*7*7  nc 512   h*w 49
        print(f'camshape{cam.shape}')
        cam = cam.reshape(h, w)
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
        finalconv_name = 'layer4'
        self.net._modules.get(finalconv_name).register_forward_hook(hook_feature)

        # get the softmax weight
        params = list(self.net.parameters())
        self.weight_softmax = np.squeeze(params[-2].data.numpy())

    def forward(self, inputs):
        N, C, H, W = inputs.size()
        x = self.net(inputs)
        h_x = F.softmax(x, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], self.weight_softmax, [idx[0]])
        cam = F.interpolate(CAMs, (H, W), mode='bilinear', align_corners=True)
        return cam
