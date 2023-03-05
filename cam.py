# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

import io
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

# input image
LABELS_file = 'imagenet-simple-labels.json'
image_file = '../datasets/VOC2012/JPEGImages/2008_001034.jpg'
# image_file = 'a.jpg'

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# print(net)
# print(net._modules.get('backbone'))


net._modules.get(finalconv_name).register_forward_hook(hook_feature)


# get the softmax weight
params = list(net.parameters())
# print(params[-2].data.shape)
weight_softmax = np.squeeze(params[-2].data.numpy())
print(weight_softmax.shape)
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    print(f'bz:{bz}, nc={nc}, h={h}, w={w}')
    output_cam = []
    for idx in class_idx:
        # idx对应的那个概率最大的类，所以是1*512
        # softmax 1*512               512*49       1*49
        print(f'feature_conv.shape:{feature_conv.shape}')
        print(f'weight_softmax[idx].shape:{weight_softmax[idx].shape}')
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))  #feature conv 1*512*7*7  nc 512   h*w 49
        print(f'camshape{cam.shape}')
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)     # 减去最小，除以最大，目的是归一化
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        print(f'cam_imgshape{cam_img.shape}')
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

# load test image
img_pil = Image.open(image_file)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)
print(f'logit:{logit.shape}')
# load the imagenet category list
with open(LABELS_file) as f:
    classes = json.load(f)


h_x = F.softmax(logit, dim=1).data.squeeze()
print(f'h_x:{h_x.shape}')
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
# print(f'probs:{probs.shape}')
idx = idx.numpy()
# print(f'probs:{probs} idx:{idx}')
# output the prediction
for i in range(0, 5):
    # print(f'idx[{i}], class {idx[i]}')
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction

# 最后一层（layer4）输出的特征图 1*512*7*7
# print(f'type(features_blobs[0])={features_blobs[0].shape}')
# print(f'features_blobs[0]={features_blobs[0]}')

# 最后一层（layer4）输出的weight_softmax 1000*512       1000个类 对应的512通道
# print(f'type(weight_softmax)={weight_softmax.shape}')
# print(f'weight_softmax[0]={weight_softmax}')



CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

print(f'cams999: {CAMs[0].shape}')



# render the CAM and output
# print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
img = cv2.imread(image_file)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)
