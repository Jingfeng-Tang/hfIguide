import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
# from tensorboardX import SummaryWriter
import torch.nn.functional as F
# loss = F.multilabel_soft_margin_loss(outputs, targets)
from model import Network

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
                                                   transforms.RandomHorizontalFlip(),
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
            if iter > 0:
                break
            print(pack[0])
            img = pack[1]
            N, C, H, W = img.size()
            img_mask = pack[2]
            img_mask = img_mask.cuda()
            label = pack[3]
            label = label.cuda()
            cam, output = model(img.cuda(), label, pack[0])
            # print(type(cam))
            # print(cam[0][0])
            # print('-----------------------一次前向完成-----------------')
            # loss1 分类损失
            loss1 = F.multilabel_soft_margin_loss(output, label)
            # print(f'loss1 OK-----------------------------------------')
            # loss2 hfiguide loss

            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'current epoch:{ep}, loss:{loss}')

    torch.save(model.state_dict(), args.session_name + '.pth')
    print("end")
