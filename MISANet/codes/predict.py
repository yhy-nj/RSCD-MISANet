import importlib
import ISDANetxiaorong
importlib.reload(ISDANetxiaorong)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from PIL import Image
import cv2

from MISA import MY_NET

import warnings
warnings.filterwarnings('ignore')
from torchvision.models import vgg16
n_class = 2

checkpoint = torch.load(r'C:\PycharmProjects\pytorch-from book\codes-paper2\results\xiaorong_v2_SY\xiaorong_attcat\F1_0.8297_iou_0.7090_epoch_13.pth',map_location=torch.device('cuda:0'))


net = MY_NET().cuda()
net.load_state_dict(checkpoint)
net.eval()  # 将网络设置为评估模式


# 定义颜色
COLORS = {
    0: (0, 0, 0),  # 黑色背景
    1: (255, 255, 255),  # 白色前景
    2: (0, 0, 255),  # 红色误报
    3: (0, 255, 0)  # 绿色
}



# 定义create_visual_anno函数
def create_visual_anno(label):
    label = label.squeeze()
    H, W = label.shape[:2]
    image = np.zeros((H, W, 3), dtype=np.uint8)
    for key in COLORS:
        if key == 0:
            continue
        color = COLORS[key]
        indices = np.where(label == int(key))
        image[indices[0], indices[1], :] = color
    return image

# 预测文件夹中的所有图片
img_folder = r'C:\PycharmProjects\pytorch-from book\codes-paper2\datasets\SYSU-CD\test'
# img_list = os.listdir(os.path.join(img_folder, 't1'))
img_list = os.listdir(os.path.join(img_folder, 'A'))
# print(len(img_list))

for i in range(len(img_list)):
    before = tf.to_tensor(Image.open(os.path.join(img_folder, 'A', img_list[i]))).unsqueeze(dim=0).cuda()
    # before = tf.to_tensor(Image.open(os.path.join(img_folder, 't1', img_list[i]))).unsqueeze(dim=0).cuda()
    after = tf.to_tensor(Image.open(os.path.join(img_folder, 'B', img_list[i]))).unsqueeze(dim=0).cuda()
    # after = tf.to_tensor(Image.open(os.path.join(img_folder, 't2', img_list[i]))).unsqueeze(dim=0).cuda()
    # change = tf.to_tensor(Image.open(os.path.join(img_folder, 'OUT', img_list[i]))).cuda()
    change_filename = img_list[i].replace('.jpg', '.png')
    # change_filename = img_list[i].replace('.png', '.jpg')
    # change_filename = img_list[i]
    change = tf.to_tensor(Image.open(os.path.join(img_folder, 'OUT', change_filename))).cuda()
    # change = tf.to_tensor(Image.open(os.path.join(img_folder, 'label', change_filename))).cuda()

    pred,aux1,aux2,aux3 = net(before, after)
    # pred= net(before, after)
    # label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy().astype(np.int)
    label_pred = torch.argmax(pred, dim=1).cpu().numpy().astype(np.int32)
    label_true = change.data.cpu().numpy()
    label_true_ = label_true.astype(np.int32)

    # 计算true positive、false positive、false negative
    true_positive = np.logical_and(label_pred == 1, label_true_ == 1)
    false_positive = np.logical_and(label_pred == 1, label_true_ == 0)
    false_negative = np.logical_and(label_pred == 0, label_true_ == 1)

    # 创建可视化标注图
    visual_anno = np.zeros_like(label_true_)
    visual_anno[true_positive] = 1
    visual_anno[false_positive] = 2
    visual_anno[false_negative] = 3
    visual_anno_img = create_visual_anno(visual_anno)

    # 保存预测结果

    cv2.imwrite(os.path.join(r"E:\yucetu\ABMFNet\GZ", img_list[i]), visual_anno_img)
