import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
from tool import imutils
from torchvision import transforms

import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
from torchvision import transforms

import matplotlib.pyplot as plt
import scipy.fft as fp
from scipy import fftpack


IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST, range(len(CAT_LIST))))


def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME, img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def load_image_label_list_from_xml(img_name_list, voc12_root):
    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


def load_image_label_list_from_npy(img_name_list):
    cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    # img_name_list = img_gt_name_list
    return img_name_list


def fft(img):
    im = np.array(img)
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

    mask_all[mask_all >= 255] = 1  # 生成掩模，边缘高频部分为1，其余为0
    mask_all = torch.from_numpy(mask_all)   # tensor类型

    return mask_all     # numpy.ndarray类型


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)  # 原图随机水平翻转，颜色抖动，resize
        # print(type(img))

        image_transforms_gray = transforms.Compose([
                     transforms.Grayscale(1)
        ])

        img_gray = image_transforms_gray(img)

        # 获取图像高频部分
        # img_gray = PIL.Image.open(get_img_path(name, self.voc12_root)).convert('L')
        # 通过FFT获取到高频图像，作为mask，
        mask = fft(img_gray)    # tensor类型
        # 去除mask部分，保留高频部分
        # img转tensor
        transf = transforms.ToTensor()
        img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)
        # hadamard积，提取高频边缘部分
        # print(f'img_tensor[0].shape:{img_tensor[0].shape}')
        # print(f'mask.shape:{mask.shape}')
        img_hf_tensor0 = torch.einsum("ij, ij -> ij", img_tensor[0], mask)
        img_hf_tensor1 = torch.einsum("ij, ij -> ij", img_tensor[1], mask)
        img_hf_tensor2 = torch.einsum("ij, ij -> ij", img_tensor[2], mask)
        # 合成高频图
        img_hf_tensor = torch.stack([img_hf_tensor0, img_hf_tensor1, img_hf_tensor2], dim=0)  # tensor数据格式是torch(C,H,W)

        # print(f'name:{name}')
        # print(f'img_tensor.shape:{img_tensor.shape}')
        # print(f'img_hf_tensor.shape:{img_hf_tensor.shape}')


        return name, img_tensor, img_hf_tensor


class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        # self.label_list = load_image_label_list_from_xml(self.img_name_list, self.voc12_root)

    def __getitem__(self, idx):
        name, img, img_hf = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, img_hf, label


class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (
            int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label


class VOC12ClsDatasetMS(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (
            int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        return name, ms_img_list, label


class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius + 1, radius):
                if x * x + y * y < radius * radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius - 1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy + self.crop_height, self.radius_floor + dx:self.radius_floor + dx + self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)),
                                               concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(
            neg_affinity_label)


class VOC12AffDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize // 8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_la_path = os.path.join(self.label_la_dir, name + '.npy')

        label_ha_path = os.path.join(self.label_ha_dir, name + '.npy')

        label_la = np.load(label_la_path, allow_pickle=True).item()
        label_ha = np.load(label_ha_path, allow_pickle=True).item()

        label = np.array(list(label_la.values()) + list(label_ha.values()))
        label = np.transpose(label, (1, 2, 0))

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        no_score_region = np.max(label, -1) < 1e-5
        label_la, label_ha = np.array_split(label, 2, axis=-1)
        label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
        label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
        label = label_la.copy()
        label[label_la == 0] = 255
        label[label_ha == 0] = 0
        label[no_score_region] = 255  # mostly outer of cropped region
        label = self.extract_aff_lab_func(label)

        return img, label


class VOC12AffGtDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_dir = label_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize // 8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_path = os.path.join(self.label_dir, name + '.png')

        label = scipy.misc.imread(label_path)

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        label = self.extract_aff_lab_func(label)

        return img, label


if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    print("start")
    #
    # train_dataset = VOC12ImageDataset("train_aug.txt", voc12_root="../../datasets/VOC2012",
    #                                                transform=transforms.Compose([
    #                                                    transforms.RandomHorizontalFlip(),
    #                                                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
    #                                                                           hue=0.1),
    #                                                    transforms.Resize((224, 224))
    #                                                ]))
    #
    # train_data_loader = DataLoader(train_dataset, batch_size=8,
    #                                    shuffle=True, num_workers=8, pin_memory=True, drop_last=True,
    #                                    )
    #
    # for iter, pack in enumerate(train_data_loader):
    #     print(f'iter:{iter}')
    #     print(type(pack[0]))
    #     print(type(pack[1]))
    #     print(type(pack[2]))
    #
    # print("end")