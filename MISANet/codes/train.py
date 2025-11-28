import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from assess import hist_sum, compute_metrics  # 假设这是你的评估函数
from MISA import MY_NET  # 替换为修改后的极简 MY_NET
# from GZCDDataset import GZCDDataset  # 数据集根据需求调整
from LEVIRdataset import LEVIRDataset  # 数据集根据需求调整
from poly import adjust_learning_rate_poly
import warnings
warnings.filterwarnings('ignore')

# 训练参数
Epoch = 200
lr = 0.0001
n_class = 2
F1_max = 0.5
root = r'D:\桌面\MISANet-main\misanet\result'  # 结果保存路径

# 数据集加载（以 LEVIR 为例，可替换）
train_data = LEVIRDataset(mode='train')
data_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_data = LEVIRDataset(mode='test')
test_data_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# 初始化模型（极简版 MY_NET，仅用 backbone）
net = MY_NET(num_classes=2).cuda()  # 注意：修改 MY_NET 构造函数，确保 num_classes 入参正确

# 损失函数、优化器
criterion = nn.BCEWithLogitsLoss().cuda()  # 二分类任务用 BCE
optimizer = optim.Adam(net.parameters(), lr=lr)

# 训练日志文件
with open(root + '/train.txt', 'a') as f_train, open(root + '/test.txt', 'a') as f_test:
    for epoch in range(Epoch):
        # 学习率调整（可选，这里保留原 poly 策略）
        new_lr = adjust_learning_rate_poly(optimizer, epoch, Epoch, lr, 0.9)
        print(f'Epoch {epoch} | lr: {new_lr}')

        # ===================== 训练阶段 =====================
        net.train()
        _train_loss = 0.0
        _hist = np.zeros((n_class, n_class))

        for before, after, change in tqdm(data_loader, desc=f'train epoch{epoch}', ncols=100):
            before, after, change = before.cuda(), after.cuda(), change.cuda()

            # 标签处理：转为独热编码（适配 BCEWithLogitsLoss）
            change = change.squeeze(dim=1).long()  # 去除通道维度，转为长整型
            # change = change.squeeze(dim=1).float()  # 去除通道维度，转为长整型
            # change = change.unsqueeze(1)


            one_hot = F.one_hot(change, num_classes=n_class)  # 生成独热编码
            change_one_hot = one_hot.permute(0, 3, 1, 2).contiguous().float().cuda()  # 调整维度并转 float

            optimizer.zero_grad()

            output, output1, output2, output3 = net(before, after)
            pred = output  # 只用主输出

            change_one_hot = change_one_hot.float().cuda()

            output = output.cuda()
            output1 = output1.cuda()
            output2 = output2.cuda()
            output3 = output3.cuda()
            change_one_hot = change_one_hot.cuda()

            loss_output = criterion(output, change_one_hot)
            loss_output1 = criterion(output1, change_one_hot)
            loss_output2 = criterion(output2, change_one_hot)
            loss_output3 = criterion(output3, change_one_hot)
            loss = loss_output + loss_output1 + loss_output2 + loss_output3

            # 前向传播（极简模型仅输出主预测）
            # pred = net(before, after)  # 修改后 MY_NET 的 forward 直接返回差异预测

            # 计算损失（主输出 + 辅助输出，若有的话；此处极简版可能只有主输出）
            # loss = criterion(pred, change_one_hot)
            # loss = criterion(pred,change)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 累计损失和混淆矩阵
            _train_loss += loss.item()
            label_pred = F.softmax(pred, dim=1).max(dim=1)[1].cpu().numpy()  # 预测类别
            label_true = change.cpu().numpy()  # 真实类别
            _hist += hist_sum(label_true, label_pred, n_class)

        # 训练指标计算
        train_loss = _train_loss / len(data_loader)
        miou, oa, kappa, precision, recall, iou, F1 = compute_metrics(_hist)
        print(f'Train | Epoch {epoch} | Loss: {train_loss:.4f} | F1: {F1:.4f} | mIoU: {miou:.4f}')
        f_train.write(f'Epoch:{epoch}|train loss:{train_loss:.4f}|train miou:{miou:.4f}|train F1:{F1:.4f}\n')

        # ===================== 测试阶段 =====================
        net.eval()
        _test_loss = 0.0
        _hist_test = np.zeros((n_class, n_class))

        with torch.no_grad():
            for before, after, change in tqdm(test_data_loader, desc=f'test epoch{epoch}', ncols=100):
                before, after, change = before.cuda(), after.cuda(), change.cuda()

                # 标签处理（同训练阶段）
                change = change.squeeze(dim=1).long()

                one_hot = F.one_hot(change, num_classes=n_class)
                change_one_hot = one_hot.permute(0, 3, 1, 2).contiguous().float().cuda()

                output, output1, output2, output3 = net(before, after)
                pred = output  # 只用主输出

                change_one_hot = change_one_hot.float().cuda()
                output = output.cuda()
                output1 = output1.cuda()
                output2 = output2.cuda()
                output3 = output3.cuda()
                change_one_hot = change_one_hot.cuda()

                loss_output = criterion(output, change_one_hot)
                loss_output1 = criterion(output1, change_one_hot)
                loss_output2 = criterion(output2, change_one_hot)
                loss_output3 = criterion(output3, change_one_hot)
                loss = loss_output + loss_output1 + loss_output2 + loss_output3

                # # 前向传播
                # pred = net(before, after)
                # loss = criterion(pred, change_one_hot)

                # 累计损失和混淆矩阵
                _test_loss += loss.item()
                label_pred = F.softmax(pred, dim=1).max(dim=1)[1].cpu().numpy()
                label_true = change.cpu().numpy()
                _hist_test += hist_sum(label_true, label_pred, n_class)

        # 测试指标计算
        test_loss = _test_loss / len(test_data_loader)
        miou_test, oa_test, kappa_test, precision_test, recall_test, iou_test, F1_test = compute_metrics(_hist_test)
        print(f'Test  | Epoch {epoch} | Loss: {test_loss:.4f} | F1: {F1_test:.4f} | mIoU: {miou_test:.4f}')
        f_test.write(f'Epoch:{epoch}|test loss:{test_loss:.4f}|test miou:{miou_test:.4f}|test F1:{F1_test:.4f}\n')

        # 模型保存（保留 F1 最高的模型）
        if F1_test > F1_max:
            save_path = root + f'/F1_{F1_test:.4f}_epoch_{epoch}.pth'
            torch.save(net.state_dict(), save_path)  # 建议保存 state_dict 而非整个模型
            F1_max = F1_test