from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import torch.nn as nn

LABELS_file = 'imagenet-simple-labels.json'
image_file = '../datasets/VOC2012/JPEGImages/2007_004481.jpg'

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


net = models.resnet50(pretrained=True)
fc_in_features = net.fc.in_features
net.fc = nn.Linear(fc_in_features, 20, bias=True)
finalconv_name = 'layer4'
net.modules.get(finalconv_name).register_forward_hook(hook_feature)

net.eval()

# get the softmax weight
params = list(net.parameters())
# print(params[-2].data.shape)
weight_softmax = np.squeeze(params[-2].data.numpy())

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

# load the imagenet category list
with open(LABELS_file) as f:
    classes = json.load(f)

h_x = F.softmax(logit, dim=1).data.squeeze()
print(f'h_x:{h_x.shape}')
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()
# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
img = cv2.imread(image_file)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)
