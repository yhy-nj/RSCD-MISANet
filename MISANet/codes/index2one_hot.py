import torch
# def get_one_hot(label, N):
#     size = list(label.size())
#     label = label.view(-1)   # reshape 为向量
#     ones = torch.sparse.torch.eye(N)
#     ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
#     size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
#     return ones.view(*size)

# def get_one_hot(label, N):
#     size = list(label.size())        # 获取标签张量维度
#     label = label.view(-1)           # 展平为 1D 向量
#     ones = torch.eye(N).to(label.device)  # 生成单位矩阵（与标签同设备）
#     ones = ones.index_select(dim=0, index=label)  # 按标签索引生成独热编码
#     size.append(N)                   # 追加类别维度
#     return ones.view(*size)          # 恢复原始维度 + 独热编码维度
def get_one_hot(label, num_classes):
    """将标签转换为独热编码（支持多维度输入）"""
    shape = list(label.shape)
    label = label.view(-1).long()  # 展平并确保为整数类型
    ones = torch.eye(num_classes, device=label.device)
    ones = ones.index_select(dim=0, index=label)
    shape.append(num_classes)  # 追加独热编码维度
    return ones.view(*shape).permute(0, 3, 1, 2).float()  # 调整为 [B, C, H, W]