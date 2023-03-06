# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

import io

import torch.nn
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy
import cv2
import json


import matplotlib.pyplot as plt
import scipy.fft as fp
from scipy import fftpack

import torch
from PIL import Image
import matplotlib.pyplot as plt
from tool import myutils

# hook the feature extractor
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM(feature_conv, weight_softmax, class_idx, cls_num, H, W, name):
    # print(f'feature_conv_i.shape{cls_num}')
    # print(f'cls_num):{cls_num}')
    transImg = transforms.ToPILImage()
    # print(f'1type(cls_num[0]):{type(int(cls_num[0]))}')
    # class_idx 二维的，每个图的概率顺序】
    # cls_num 每个图的类别数 一维的
    # generate the class activation maps upsample to 256x256
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    # print(f'bz:{bz}, nc={nc}, h={h}, w={w}')    # 8*512*7*7
    # print(f'weight_softmax.shape{weight_softmax.shape}')    # 20*512
    output_cam = []
    # 每张图
    for i in range(8):      # 8是batchsize
        output_cam_i = []
        for j in range(int(cls_num[i])):
            # print(f'idx:{idx}')
            # cls_num是图像中所含有的类别总数  idx是按照概率由高到低的tensor
            # softmax 1*512               512*49       1*49
            feature_conv_i = feature_conv[i]        # 512*7*7
            # print(f'feature_conv_i.shape{feature_conv_i.shape}')
            feature_conv_i = torch.from_numpy(feature_conv_i.reshape((nc, h * w)))      #特征图    512*49
            # print(f'feature_conv_i.shape{feature_conv_i.shape}')
            # print(f'weight_softmax.shape{weight_softmax.shape}')    # 20*512
            # print(f'class_idx{class_idx}')    # 20*512
            # print(f'class_idx{class_idx[i]}')    # 20*512
            # print(f'j.item(){type(int(j.item()))}')
            # print(f'class_idx{class_idx[i][j.item()]}')    # 20*512
            # print(f'weight_softmax[class_idx[i][int(j.item())]].shape{weight_softmax[class_idx[i][int(j.item())]].shape}')    # 20*512
            cam = torch.matmul(weight_softmax[class_idx[i][j]], feature_conv_i) # 49

            cam = cam.reshape(h, w)     # 7*7
            # print(f'type(cam){type(cam)}')
            # image11 = transImg(cam)
            # image11.save('./camImg/test11.jpg')
            # 加个for循环处理每张图的CAM
            # 由label获取每张图有几个类别目标
            # 根据概率生成前几
            cam = cam - torch.min(cam)  # 减去最小，除以最大，目的是归一化
            # print(f'cam.shape {cam.shape}')
            cam_img = cam / torch.max(cam)
            # print(f'cam_img.shape{cam_img.shape}')
            # image22 = transImg(cam_img)
            # image22.save('./camImg/test22.jpg')
            cam_img = torch.tensor(255 * cam_img, dtype=torch.uint8)
            # print(f'cam_img.shape{cam_img.shape}')
            # image33 = transImg(cam_img)
            # image33.save('./camImg/test33.jpg')
            cam_img = cam_img.numpy()
            cam_img = cv2.resize(cam_img, size_upsample)
            cam_img = torch.from_numpy(cam_img)
            # image44 = transImg(cam_img)
            # image44.save('./camImg/test44.jpg')
            cam_img = torch.tensor(cam_img, dtype=torch.float32)
            # image55 = transImg(cam_img)
            # image55.save('./camImg/test55.jpg')
            cam_img = cam_img.unsqueeze(0)
            cam_img = cam_img.unsqueeze(0)
            # print(f'cam_img.shape {cam_img.shape}')
            cam_tensorsize = F.interpolate(cam_img, size=(H, W), mode='bilinear', align_corners=True)
            cam_tensorsize = cam_tensorsize.squeeze()

            cam_te_u8 = torch.tensor(cam_tensorsize, dtype=torch.uint8)
            cam_nd = cam_te_u8.numpy()

            # 生成热力图 看效果
            strimg = '../datasets/VOC2012/JPEGImages/'+str(name[i])+'.jpg'
            img = cv2.imread(strimg)
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(cam_nd, (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5
            cv2.imwrite('./camImg/'+str(name[i])+'+'+str(j)+'_CAM.jpg', result)



            output_cam_i.append(cam_tensorsize)
            # print(type(cam_tensorsize))
            # print(cam_tensorsize.shape)
            # print(f'存入当前第{i}个图像的第{j}个CAM图:')
        output_cam.append(output_cam_i)
        # print(f'存入当前第{i}个图像的所有CAM图--------------------------:')
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

    def forward(self, inputs, label, name):
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
        # print(f'idx:{idx}')
        # idx = idx.numpy()
        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], self.weight_softmax, idx, cls_num, H, W, name)
        #CAMs为一个列表，表中每一个元素（也是列表）为每张图的所有CAM图
        return CAMs, x
