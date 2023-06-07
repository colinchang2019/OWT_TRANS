# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : dataset.py
@ Time    ：2021/9/8 19:39
"""
import torch.utils.data as data
import numpy as np
import gc
import random
import pandas as pd
import os


class CSVSingleDataset(data.Dataset):  # 需要继承data.Dataset
    def __init__(self, trains):
        """
        :param pre: "./dataTransform" as default
        :param n: 15076 for train, 2542 for test
        :param dt: 1000 or 3000
        :param train: "train" or "test" in type of str
        """
        # TODO
        # 1. Initialize file path or list of file names.
        self.df = pd.read_excel(trains)
        self.lenTotal = len(self.df)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        input_columns = [8, 9, 10, 6, 7, 0, 1]  # [L, D, soil, u_i, f_i, H, M,],
        output_columns = [4, 5]  # [H_i, M_i]
        x = self.df.iloc[index: index+1, [8, 9, 10, 6, 7, 0, 1]].values
        y = self.df.iloc[index:index + 1, [4, 5]].values
        x, y = x.astype(np.float), y.astype(np.float)
        # print(x.shape, ymask.shape, y.shape)
        return x, y

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.lenTotal

if __name__ == '__main__':
    print()