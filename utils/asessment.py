# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : asessment.py
@ Time    ：2022/12/17 12:30
"""
# NSE Implementation in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)


class mape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:`
        """
        # res = torch.abs((input - target) / target)
        res = torch.where(target==0.0, 0.0, torch.abs((input - target) / target))
        return torch.mean(res, dim=0, keepdim=True)


class nse(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        :param input:  default shape is [batch, channel, height, width]
        :param target: same shape as input
        :return:`
        """
        ym = torch.mean(target, dim=0, keepdim=True)
        # Duplicate ym to match the shape of target
        ym = ym.expand(target.size(0), 1, 2)
        up = torch.sum((input - target) ** 2, dim=0, keepdim=True) # .to(torch.float32)
        down = torch.sum((target - ym) ** 2, dim=0, keepdim=True) # .to(torch.float32)
        res = torch.where(down==0.0, 0.0, up/down)
        # print(res.shape)
        return 1 - res


if __name__ == '__main__':
    loss_mape = mape()
    loss_nse = nse()
    a = torch.from_numpy(np.zeros(shape=(2, 1, 6)))
    b = torch.from_numpy(np.ones(shape=(2, 1, 6)))
    c = b * 2

    print(loss_mape(b, c))
    print(loss_nse(b, c))

