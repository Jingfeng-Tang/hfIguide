#  此文件包含工具函数
import torch
import PIL.Image
from torchvision import transforms
import numpy


# tensor转PIL + 水平翻转 image
def tensorToPILImage(x):
    trans = transforms.ToPILImage()
    image = trans(x)
    # image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    return image


# tensor转ndarray
def tensorToNdarray(x):
    x_nd = x.numpy()

    return x_nd
