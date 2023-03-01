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

from model import Network

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="resnet18", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_SEAM", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=False, type=str)
    parser.add_argument("--voc12_root", default='../../datasets/VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--device", default='cuda', type=str)
    args = parser.parse_args()

    print(vars(args))

    model = Network()

    print(model)

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
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)

    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    optimizer = torch.optim.SGD(lr=args.lr, decay=args.wt_dec, momentum=0.9, nesterov=False)
    model = model.to(device=args.device)
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr')

    timer = pyutils.Timer("Session started: ")
    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):


            img1 = pack[1]
            N, C, H, W = img1.size()
            label = pack[2]
            bg_score = torch.ones((N, 1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            cam1, cam_rv1 = model(img1)  # t cam_rv1由PCM模型生成
            label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label)[:, 1:, :, :])
            cam1 = F.interpolate(visualization.max_norm(cam1), scale_factor=scale_factor, mode='bilinear',
                                 align_corners=True) * label
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1), scale_factor=scale_factor, mode='bilinear',
                                    align_corners=True) * label

            cam2, cam_rv2 = model(img2)
            label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label)[:, 1:, :, :])
            cam2 = visualization.max_norm(cam2) * label
            cam_rv2 = visualization.max_norm(cam_rv2) * label

            loss_cls1 = F.multilabel_soft_margin_loss(label1[:, 1:, :, :], label[:, 1:, :, :])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:, 1:, :, :], label[:, 1:, :, :])

            ns, cs, hs, ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1[:, 1:, :, :] - cam2[:, 1:, :, :]))
            # loss_er = torch.mean(torch.pow(cam1[:,1:,:,:]-cam2[:,1:,:,:], 2))
            cam1[:, 0, :, :] = 1 - torch.max(cam1[:, 1:, :, :], dim=1)[0]
            cam2[:, 0, :, :] = 1 - torch.max(cam2[:, 1:, :, :], dim=1)[0]
            #            with torch.no_grad():
            #                eq_mask = (torch.max(torch.abs(cam1-cam2),dim=1,keepdim=True)[0]<0.7).float()
            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=(int)(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=(int)(21 * hs * ws * 0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2
            loss = loss_cls + loss_er + loss_ecr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item(), 'loss_cls': loss_cls.item(), 'loss_er': loss_er.item(),
                           'loss_ecr': loss_ecr.item()})

            if (optimizer.global_step - 1) % 50 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr'),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()))

                avg_meter.pop()

                # Visualization for training process
                img_8 = img1[0].numpy().transpose((1, 2, 0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:, :, 0] = (img_8[:, :, 0] * std[0] + mean[0]) * 255
                img_8[:, :, 1] = (img_8[:, :, 1] * std[1] + mean[1]) * 255
                img_8[:, :, 2] = (img_8[:, :, 2] * std[2] + mean[2]) * 255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)

                input_img = img_8.transpose((2, 0, 1))
                h = H // 4;
                w = W // 4
                p1 = F.interpolate(cam1, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p2 = F.interpolate(cam2, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p_rv1 = F.interpolate(cam_rv1, (h, w), mode='bilinear')[0].detach().cpu().numpy()
                p_rv2 = F.interpolate(cam_rv2, (h, w), mode='bilinear')[0].detach().cpu().numpy()

                image = cv2.resize(img_8, (w, h), interpolation=cv2.INTER_CUBIC).transpose((2, 0, 1))
                CLS1, CAM1, _, _ = visualization.generate_vis(p1, None, image,
                                                              func_label2color=visualization.VOClabel2colormap,
                                                              threshold=None, norm=False)
                CLS2, CAM2, _, _ = visualization.generate_vis(p2, None, image,
                                                              func_label2color=visualization.VOClabel2colormap,
                                                              threshold=None, norm=False)
                CLS_RV1, CAM_RV1, _, _ = visualization.generate_vis(p_rv1, None, image,
                                                                    func_label2color=visualization.VOClabel2colormap,
                                                                    threshold=None, norm=False)
                CLS_RV2, CAM_RV2, _, _ = visualization.generate_vis(p_rv2, None, image,
                                                                    func_label2color=visualization.VOClabel2colormap,
                                                                    threshold=None, norm=False)
                # MASK = eq_mask[0].detach().cpu().numpy().astype(np.uint8)*255
                loss_dict = {'loss': loss.item(),
                             'loss_cls': loss_cls.item(),
                             'loss_er': loss_er.item(),
                             'loss_ecr': loss_ecr.item()}
                itr = optimizer.global_step - 1


        else:
            print('')
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + '.pth')
