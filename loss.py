# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
class NLLLoss(nn.Module):
    """Self-Defined NLLLoss Function

    Args:
        weight: Tensor (num_class, )
    """
    def __init__(self, weight, device='cpu'):
        super(NLLLoss, self).__init__()
        self.weight = weight.to(device)  # 将 weight 转移到指定设备
        self.device = device  # 设备参数


    def forward(self, prob, target):
        """
        Args:
            prob: (N, C) 
            target : (N, )
        """
        N = target.size(0)
        C = prob.size(1)
        weight = Variable(self.weight).view((1, -1)).to(self.device)
        weight = weight.expand(N, C)  # (N, C)
        
        prob = weight * prob  # 计算加权的概率值

        # 创建 one_hot 编码的 target，并将其放到指定设备上
        one_hot = torch.zeros((N, C)).to(self.device)
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.bool()  # 这里用 .bool() 代替原先的 .ByteTensor
        one_hot = Variable(one_hot)

        # 使用 mask 获取目标对应的概率值
        loss = torch.masked_select(prob, one_hot)
        return -torch.sum(loss)



