import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from copy import deepcopy

INT = torch.IntTensor
LONG = torch.LongTensor
BYTE = torch.ByteTensor
FLOAT = torch.FloatTensor


def init_weight(m):
    """
    偏重函数：用于初始化神经网络模型中的权重和偏置
    :param m:
    :return:
    """
    if type(m) == nn.Linear:  # 判断当前模型m是否为线性层
        size = m.weight.size()
        fan_out = size[0]  # 输出特征的数量
        fan_in = size[1]  # 输入特征的数量s
        variance = np.sqrt(2.0 / (fan_in + fan_out))  # 根据输入特征数量和输出特征数量计算权重的标准差
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)
