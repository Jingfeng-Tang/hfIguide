# # -*- coding: utf-8 -*-
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# def gaussian_filter_high_f(fshift, D):
#     # 获取索引矩阵及中心点坐标
#     h, w = fshift.shape
#     x, y = np.mgrid[0:h, 0:w]
#     center = (int((h - 1) / 2), int((w - 1) / 2))
#
#     # 计算中心距离矩阵
#     dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
#
#     # 计算变换矩阵
#     template = np.exp(- dis_square / (2 * D ** 2))
#
#     return template * fshift
#
#
# def gaussian_filter_low_f(fshift, D):
#     # 获取索引矩阵及中心点坐标
#     h, w = fshift.shape
#     x, y = np.mgrid[0:h, 0:w]
#     center = (int((h - 1) / 2), int((w - 1) / 2))
#
#     # 计算中心距离矩阵
#     dis_square = (x - center[0]) ** 2 + (y - center[1]) ** 2
#
#     # 计算变换矩阵
#     template = 1 - np.exp(- dis_square / (2 * D ** 2))  # 高斯过滤器
#
#     return template * fshift
#
#
# def circle_filter_high_f(fshift, radius_ratio):
#     """
#     过滤掉除了中心区域外的高频信息
#     """
#     # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
#     template = np.zeros(fshift.shape, np.uint8)
#     crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
#     radius = int(radius_ratio * img.shape[0] / 2)
#     if len(img.shape) == 3:
#         cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
#     else:
#         cv2.circle(template, (crow, ccol), radius, 1, -1)
#     # 2, 过滤掉除了中心区域外的高频信息
#     return template * fshift
#
#
# def circle_filter_low_f(fshift, radius_ratio):
#     """
#     去除中心区域低频信息
#     """
#     # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
#     filter_img = np.ones(fshift.shape, np.uint8)
#     crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
#     radius = int(radius_ratio * img.shape[0] / 2)
#     if len(img.shape) == 3:
#         cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
#     else:
#         cv2.circle(filter_img, (crow, col), radius, 0, -1)
#     # 2 过滤中心低频部分的信息
#     return filter_img * fshift
#
#
# def ifft(fshift):
#     """
#     傅里叶逆变换
#     """
#     ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
#     iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
#     iimg = np.abs(iimg)  # 返回复数的模
#     return iimg
#
#
# def get_low_high_f(img, radius_ratio, D):
#     """
#     获取低频和高频部分图像
#     """
#     # 傅里叶变换
#     # np.fft.fftn
#     f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
#     fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频
#
#     # 获取低频和高频部分
#     hight_parts_fshift = circle_filter_low_f(fshift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
#     low_parts_fshift = circle_filter_high_f(fshift.copy(), radius_ratio=radius_ratio)
#     hight_parts_fshift = gaussian_filter_low_f(fshift.copy(), D=D)
#     low_parts_fshift = gaussian_filter_high_f(fshift.copy(), D=D)
#
#     low_parts_img = ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
#     high_parts_img = ifft(hight_parts_fshift)
#
#     # 显示原始图像和高通滤波处理图像
#     img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
#     img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
#             np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)
#
#     # uint8
#     img_new_low = np.array(img_new_low * 255, np.uint8)
#     img_new_high = np.array(img_new_high * 255, np.uint8)
#     return img_new_low, img_new_high
#
#
# # 频域中使用高斯滤波器能更好的减少振铃效应
# if __name__ == '__main__':
#     radius_ratio = 0.5  # 圆形过滤器的半径：ratio * w/2
#     D = 20  # 高斯过滤器的截止频率：2 5 10 20 50 ，越小越模糊信息越少
#     img = cv2.imread('2007_000783.jpg', cv2.IMREAD_GRAYSCALE)
#     low_freq_part_img, high_freq_part_img = get_low_high_f(img, radius_ratio=radius_ratio,
#                                                            D=D)  # multi channel or single
#
#     plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
#     plt.axis('off')
#     plt.subplot(132), plt.imshow(low_freq_part_img, 'gray'), plt.title('low_freq_img')
#     plt.axis('off')
#     plt.subplot(133), plt.imshow(high_freq_part_img, 'gray'), plt.title('high_freq_img')
#     plt.axis('off')
#     plt.show()








import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
import json


import matplotlib.pyplot as plt
import scipy.fft as fp
from scipy import fftpack

# input image
image_file = '../datasets/VOC2012/JPEGImages/'+'2010_002734'+'.jpg'

def fft():
    im = np.array(Image.open(image_file).convert('L'))  #
    im_rgb = np.array(Image.open(image_file).convert('RGB'))
    freq = fp.fft2(im)
    (w, h) = freq.shape
    half_w, half_h = int(w / 2), int(h / 2)  # 确定频谱图像的中心位置
    # 高通滤波器
    freq1 = np.copy(freq)
    freq2 = fftpack.fftshift(freq1)  # 将低频部分移至中心位置
    freq2[half_w - 20:half_w + 21, half_h - 20:half_h + 21] = 0  # select all but the first 20x20 (low) frequencies
    im1 = np.clip(fp.ifft2(fftpack.ifftshift(freq2)).real, 0, 255)  # 将两端的像素值缩至端点
    # 二值化
    ret, mask_all = cv2.threshold(src=im1,  # 要二值化的图片
                                  thresh=20,  # 全局阈值
                                  maxval=255,  # 大于全局阈值后设定的值
                                  type=cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # eroded = cv2.erode(mask_all, kernel)  # 腐蚀图像
    # dilated = cv2.dilate(mask_all, kernel)  # 膨胀图像
    #
    # plt.imshow(mask_all, cmap='gray')
    # plt.show()
    #
    # plt.imshow(eroded, cmap='gray')
    # plt.show()
    #
    # plt.imshow(dilated, cmap='gray')
    # plt.show()

    # 将im 按照维度切分
    numpydata=np.transpose(im_rgb, (2, 0, 1))
    print(f'ccim_rgb.shape:{im_rgb.shape}')
    # print(f'im_rgb:{im_rgb}')
    # print(numpydata.shape)
    # print(numpydata[0])
    print("----------------------------------------------------")
    # print(mask_all)
    mask_all[mask_all >= 255] = 1             # 生成掩模
    print("----------------------------------------------------")
    # print(mask_all)
    print(f'mask_all.shape:{mask_all.shape}')
    print(f'numpydata[0].shape:{numpydata[0].shape}')
    # im mask_all
    fin0 = np.einsum("ij, ij -> ij", numpydata[0], mask_all)
    fin1 = np.einsum("ij, ij -> ij", numpydata[1], mask_all)
    fin2 = np.einsum("ij, ij -> ij", numpydata[2], mask_all)
    print(f'fin0.shape:{fin0.shape}')
    # print(f'fin0:{fin2}')
    fin = np.array([fin0, fin1, fin2])
    np.set_printoptions(threshold=np.inf)
    print(im_rgb.dtype)
    fin = fin.astype('uint8')
    print(fin.dtype)
    finimg = np.transpose(fin, (1, 2, 0))
    print(finimg.dtype)
    img1 = Image.fromarray(finimg, 'RGB')
    img1.save('a.jpg')

    return mask_all

def edgeDetection(imgfile):
    image = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate_img = cv2.dilate(image, kernel)
    erode_img = cv2.erode(image, kernel)

    absdiff_img = cv2.absdiff(dilate_img, erode_img)
    retval, threshold_img = cv2.threshold(absdiff_img, 40, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_not(threshold_img)
    # # cv2.imshow('../datasets/VOC2012/JPEGImages/'+'2007_000068'+'.jpg', image)
    # plt.imshow(dilate_img)
    # plt.show()
    # plt.imshow(erode_img)
    # plt.show()
    # plt.imshow(absdiff_img)
    # plt.show()
    # plt.imshow(threshold_img)
    # plt.show()

    # plt.imshow(result)
    # plt.show()

    return result


fftimg = fft()
edge_img = edgeDetection(image_file)

plt.subplot(121), plt.title("fft_img"), plt.axis('off')
plt.imshow(fftimg)
plt.subplot(122), plt.title("edge_img"), plt.axis('off')
plt.imshow(edge_img)
plt.show()