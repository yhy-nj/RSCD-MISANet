from PIL import Image
import os
import glob
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data
import random
import torchvision.transforms.functional as tf

# root_path = r'C:\PycharmProjects\pytorch-from book\codes-paper2\datasets\LEVIR-CD'
root_path = r"D:\桌面\科研\遥感图像论文代码\LEVIR\LEVIR"

def read_images(root):
    imgs_At = glob.glob(root + '/train/A/*.png')
    imgs_Bt = glob.glob(root + '/train/B/*.png')
    labelst = glob.glob(root + '/train/label/*.png')
    imgs_Att = glob.glob(root + '/test/A/*.png')
    imgs_Btt = glob.glob(root + '/test/B/*.png')
    labelstt = glob.glob(root + '/test/label/*.png')

    # imgs_At = glob.glob(root + '/train/t1/*.png')
    # imgs_Bt = glob.glob(root + '/train/t2/*.png')
    # labelst = glob.glob(root + '/train/label/*.png')
    # imgs_Att = glob.glob(root + '/test/t1/*.png')
    # imgs_Btt = glob.glob(root + '/test/t2/*.png')
    # labelstt = glob.glob(root + '/test/label/*.png')
    return imgs_At, imgs_Bt, labelst, imgs_Att, imgs_Btt, labelstt

# """
# 划分训练集和测试集
# """
# def split_dataset(before, after, change):
#     train_size = int(0.8 * len(before))
#     random.seed(train_size)
#     random.shuffle(before)
#     random.seed(train_size)
#     random.shuffle(after)
#     random.seed(train_size)
#     random.shuffle(change)
#
#     random.seed(len(before)-train_size)
#     train_dataset_before = random.sample(before, train_size)
#     random.seed(len(before)-train_size)
#     train_dataset_after = random.sample(after, train_size)
#     random.seed(len(before)-train_size)
#     train_dataset_change = random.sample(change, train_size)
#
#     test_dataset_before = [i for i in before if i not in train_dataset_before]
#     test_dataset_after = [i for i in after if i not in train_dataset_after]
#     test_dataset_change = [i for i in change if i not in train_dataset_change]
#     return train_dataset_before, train_dataset_after, train_dataset_change, test_dataset_before, test_dataset_after, test_dataset_change

def imagestransforms(before, after, change):
    if random.random() > 0.7:
        before = tf.hflip(before)
        after = tf.hflip(after)
        change = tf.hflip(change)
    if random.random() < 0.3:
        before = tf.vflip(before)
        after = tf.vflip(after)
        change = tf.vflip(change)
    angle = transforms.RandomRotation.get_params([-180, 180])
    before = before.rotate(angle)  #按指定的角度angle进行旋转，并返回新旋转的对象。
    after = after.rotate(angle)
    change = change.rotate(angle)

    before = tf.to_tensor(before) #转换为tensor对象
    after = tf.to_tensor(after)
    change = tf.to_tensor(change)
    return before, after, change

def images_transforms(before, after, change):
    before, after, change = imagestransforms(before, after, change)
    return before, after, change

def images_transforms_(before, after, change):
    before = tf.to_tensor(before)
    after = tf.to_tensor(after)
    change = tf.to_tensor(change)
    return before, after, change

class LEVIRDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        # self.before_list, self.after_list, self.change_list =
        self.train_dataset_before, self.train_dataset_after, self.train_dataset_change, self.test_dataset_before, self.test_dataset_after, self.test_dataset_change = read_images(root_path)
        if mode == 'train':

            print('训练集加载了' + str(len(self.train_dataset_before)) + '张图片')
        elif mode == 'test':
            print('测试集加载了' + str(len(self.test_dataset_before)) + '张图片')

    def __getitem__(self, item):
        if self.mode == 'train':

            before = Image.open(self.train_dataset_before[item]).convert('RGB') #打开并转换成rgb图像
            after = Image.open(self.train_dataset_after[item]).convert('RGB')
            change = Image.open(self.train_dataset_change[item]).convert('L') #打开并转换为灰度图像
            # print("原始标签值范围:", np.unique(np.array(change)))  # 应为 [0, 255] 或 [0, 1]
            before, after, change = images_transforms(before, after, change)
            # print("代码执行到这里了")

            # print('before shape:', before.shape)
            # print('after shape:', after.shape)
            # print('change shape:', change.shape)

            return before, after, change
        elif self.mode == 'test':
            before = Image.open(self.test_dataset_before[item]).convert('RGB')
            after = Image.open(self.test_dataset_after[item]).convert('RGB')
            change = Image.open(self.test_dataset_change[item]).convert('L')
            before, after, change = images_transforms_(before, after, change)
            # print('before shape:', before.shape)
            # print('after shape:', after.shape)
            # print('change shape:', change.shape)

            return before, after, change


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset_before)
        elif self.mode == 'test':
            return len(self.test_dataset_before)

##测试用，只有运行当前模块时才会运行此模块。
if __name__ == '__main__':
    train_data = LEVIRDataset(mode='train')
    for i in range(1):
        before, after, change = train_data[i]

    test_data = LEVIRDataset(mode='test')
    for i in range(1):
        before, after, change = test_data[i]


"""
这段代码实现了一个用于变化检测任务的自定义数据集类，主要功能包括：

读取 LEVIR-CD 数据集的图像路径
实现训练集的数据增强（水平翻转、垂直翻转、随机旋转）
将图像转换为 PyTorch 模型可处理的 Tensor 格式
分别处理训练集和测试集（测试集不进行数据增强）

通过这个数据集类，可以方便地使用 PyTorch 的 DataLoader 来加载和批处理数据，为后续的模型训练和评估做准备。
###

"""

