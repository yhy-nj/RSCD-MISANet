import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from MISA import MY_NET


# 加载模型
model = torch.load(r"C:\Users\YHY\xwechat_files\wxid_vt3t3ulua77p11_c449\msg\file\2025-10\F1_0.9119_epoch_184.pth")
net = MY_NET(2).cuda()
net.load_state_dict(model)
net.eval()  # 将网络设置为评估模式


# 加载两张图像
image1 = Image.open(r"D:\桌面\科研\遥感图像论文代码\LEVIR\LEVIR\test\A\292.png").convert("RGB")
image2 = Image.open(r"D:\桌面\科研\遥感图像论文代码\LEVIR\LEVIR\test\B\292.png").convert("RGB")

# image1=image1[:, :, ::-1]
# image2=image2[:, :, ::-1]
# 转换图像为 PyTorch 张量，并归一化
tensor1 = torch.tensor(np.array(image1)).float() / 255
tensor2 = torch.tensor(np.array(image2)).float() / 255


# tensor1 = torch.tensor(np.array(image1[:, :, ::-1]).copy()).float() / 255
# tensor2 = torch.tensor(np.array(image2[:, :, ::-1]).copy()).float() / 255

# 将张量调整为模型所需的形状
tensor1 = tensor1.permute(2, 0, 1).unsqueeze(0).to('cuda')
tensor2 = tensor2.permute(2, 0, 1).unsqueeze(0).to('cuda')



# 使用模型进行预测
with torch.no_grad():
    # output = net(tensor1, tensor2)
    output, aux1, aux2, aux3 = net(tensor1, tensor2)
    # output = net(tensor1, tensor2)
    # output,out1,out2,out3 = model(tensor1, tensor2)

# 将输出转换为热力图，并可视化
heatmap = output[0, 0].cpu().numpy()
plt.imshow(heatmap, cmap="jet_r")
plt.axis("off")
plt.savefig("E:\\yucetu\\relitu", bbox_inches='tight')
plt.show()

