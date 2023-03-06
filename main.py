import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization, myutils
import argparse
import importlib
# from tensorboardX import SummaryWriter
import torch.nn.functional as F
# loss = F.multilabel_soft_margin_loss(outputs, targets)
from model import Network
import PIL.Image




def loss2_camNeighbourhood(cam, hf_mask):
    """
    loss2 CAM领域损失 计算CAM中像素是否在高频图的边缘内部
    （1）无     如果cam内部像素，且在边缘内；
    （2）惩罚1   如果cam边缘像素，且在边缘内；
    （3）惩罚2   如果cam内部像素，且在边缘外；
    （4）惩罚3   如果cam边缘像素，且在边缘外；
    :param cam:CAM图像
    :param hf_mask:高频图像mask
    """





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=1, type=int)
    parser.add_argument("--network", default="resnet18", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet18_hfig", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=False, type=str)
    parser.add_argument("--voc12_root", default='../datasets/VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--device", default='cuda', type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    # print(device)
    # print(vars(args))

    model = Network()

    # print(model)

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                                                   # transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                          hue=0.1),
                                                   transforms.Resize((224, 224))
                                               ]))

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True,
                                   worker_init_fn=worker_init_fn)

    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params=params, lr=args.lr, weight_decay=args.wt_dec, momentum=0.9, nesterov=False)


    # print(next(model.parameters()).device)
    model.train()
    # print(torch.cuda.device_count())
    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr')

    timer = pyutils.Timer("Session started: ")
    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):
            if iter > 0:        # 测试用，只训练一个迭代
                break
            print(pack[0])
            img = pack[1]       # 图像
            N, C, H, W = img.size()
            img_mask = pack[2]  # mask
            img_mask = img_mask.cuda()
            # # generate high frequency Image for test
            # img_mask_i = img_mask[0][0]
            # # print(f'img_mask_i.shape:{img_mask_i.shape} type(img_mask_i):{type(img_mask_i)}')
            # img_hf = myutils.tensorToPILImage(img_mask_i)  # numpy.darray 转换成 PIL.Image
            # str_imghf = "./hfImg/" + pack[0][0] + '.jpg'
            # img_hf.save(str_imghf)

            # generate high frequency 3 dimension Image for test
            img_mask_i = img_mask[0]
            # print(f'img_mask_i.shape:{img_mask_i.shape} type(img_mask_i):{type(img_mask_i)}')
            img_hf = myutils.tensorToPILImage(img_mask_i)  # numpy.darray 转换成 PIL.Image
            str_imghf = "./hfImg/" + pack[0][0] + '.jpg'
            img_hf.save(str_imghf)

            label = pack[3]
            label = label.cuda()
            cam, output = model(img.cuda(), label, pack[0])
            # print(cam[0][0].shape)

            # # 计算mask与cam的哈达玛乘积
            # img_mask_i = img_mask_i.cuda()
            # cam_syn = cam[0][0]
            # cam_syn = cam_syn.cuda()
            # img_syn_tensor = torch.einsum("ij, ij -> ij", cam_syn, img_mask_i)
            # img_syn = myutils.tensorToPILImage(img_syn_tensor)  # numpy.darray 转换成 PIL.Image
            # str_img_syn = "./hfImg/" + pack[0][0] + 'syn' + '.jpg'
            # img_syn.save(str_img_syn)

            # loss1 分类损失
            loss1 = F.multilabel_soft_margin_loss(output, label)
            # loss2 CAM领域损失 计算CAM中像素是否在高频图的边缘内部
            # （1）无     如果cam内部像素，且在边缘内；
            # （2）惩罚1   如果cam边缘像素，且在边缘内；
            # （3）惩罚2   如果cam内部像素，且在边缘外；
            # （4）惩罚3   如果cam边缘像素，且在边缘外；

            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'current epoch:{ep}, loss:{loss}')

    torch.save(model.state_dict(), args.session_name + '.pth')
    print("end")
