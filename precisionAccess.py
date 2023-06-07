# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : precisionAccess.py
@ Time    ：2021/6/30 10:56
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from dataset import CSVDataset, CSVSingleDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
from modelSimple import OWTTransformer as KTransformer
from config import cfg
from utils.asessment import mape, nse


def validate(model, test="test"):
    """
    :param model:
    :param config:
    :param path: for dataset and pth file
    :param case: for two kind of dataset
    :param los: for loss in training
    :return: [(loss, precision, recall, accuracy)]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # create dataloader
    print("Laoding dataset to torch.")
    pathtrainx = "./dataTrain/train_x.npz"
    pathtrainy = "./dataTrain/train_y.npz"
    pathtestx = "./dataTrain/test_x.npz"
    pathtesty = "./dataTrain/test_y.npz"

    train_x = np.load(pathtrainx)["sequence"]
    train_y = np.load(pathtrainy)["sequence"]
    test_x = np.load(pathtestx)["sequence"]
    test_y = np.load(pathtesty)["sequence"]

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()
    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).float()
    print(train_x.shape, test_x.shape)

    if test == "test":
        testDataset = TensorDataset(test_x, test_y)
        print("test: ")
    else:
        testDataset = TensorDataset(train_x, train_y)
        print("train: ")

    testDataloader = DataLoader(
        dataset=testDataset,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    torch.cuda.empty_cache()
    model.eval()
    loss_fn = nn.MSELoss()
    loss_mape = mape()
    loss_nse = nse()

    with torch.no_grad():
        loss_t, loss_m, loss_n = 0, np.array([]), np.array([])
        n = len(testDataloader)
        for j, (x, y) in enumerate(testDataloader):
            x,  y = x.to(device), y.to(device)
            x = x.double()  # Convert input to Float
            y = y.double()
            # print(x.shape, ymask.shape)
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss_t += loss.item()

            loss_test = np.array(loss_mape(outputs, y).cpu())
            if j == 0:
                loss_m = loss_test
            else:
                loss_m = np.concatenate((loss_m, loss_test), axis=0)

            loss_test = np.array(loss_nse(outputs, y).cpu())
            # print("nse: ", loss_test)
            if j == 0:
                loss_n = loss_test
            else:
                loss_n = np.concatenate((loss_n, loss_test), axis=0)
    print("Loss in Test dataset: {}".format(loss_t / (j + 1)))
    n = j + 1
    loss_m = np.mean(loss_m, axis=0)
    loss_n = np.mean(loss_n, axis=0)
    print("mse loss", loss_t / (j + 1))
    print("mape loss", loss_m)
    print("nse loss", loss_n)
    return loss_t / n


def validatesingle(model, tests):
    """
    :param model:
    :param config:
    :param path: for dataset and pth file
    :param case: for two kind of dataset
    :param los: for loss in training
    :return: [(loss, precision, recall, accuracy)]
    """
    path = cfg.preTransform + tests + cfg.last
    pathout = cfg.csvpred + tests + cfg.last
    print(path, pathout)
    if not os.path.exists(path):
        return

    testDataset = CSVSingleDataset(path)
    testDataloader = DataLoader(
        dataset=testDataset,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    torch.cuda.empty_cache()
    model.eval()
    loss_fn = nn.MSELoss()  # nn.MSELoss()

    with torch.no_grad():
        loss_t, loss_m, loss_n = 0, np.array([]), np.array([])
        n = len(testDataloader)
        for j, (x, y) in enumerate(testDataloader):
            x, y = x.to(device),  y.to(device)
            # print(x.shape, ymask.shape)
            outputs = model(x)
            # print(outputs.shape, y.shape)
            print(outputs.shape)

            if j==0:
                dfx = x.cpu().numpy().tolist()
                dfreal = y.cpu().numpy().tolist()
                dfpred = outputs.cpu().numpy().tolist()
            else:
                dfx += x.cpu().numpy().tolist()
                dfreal += y.cpu().numpy().tolist()
                dfpred += outputs.cpu().numpy().tolist()
    dfpred, dfreal, dfx = np.array(dfpred, dtype=np.float), np.array(dfreal, dtype=np.float), np.array(dfx, dtype=np.float)
    sz = dfx.shape
    df = pd.DataFrame(dfx.reshape(sz[0], sz[-1]))
    df.columns = ["L", "D", "soil", "U", "Fi", "H1", "M1"]  # [L, D, soil, u_i, f_i, H, M,]

    sz0 = dfreal.shape
    df0 = pd.DataFrame(dfreal.reshape(sz0[0], sz0[-1]))
    df0.columns = ["H", "M"]  # ["H", "M", "U", "Fi", "LD", "soil"]
    # df["H"] = df.stress * cfg.stress_sca
    df["H"] = df0.H
    df["M"] = df0.M

    sz1 = dfpred.shape
    df1 = pd.DataFrame(dfpred.reshape(sz1[0], sz1[-1]))
    df1.columns = ["Hp", "Mp"]  # ["H", "M", "U", "Fi", "LD", "soil"]
    df["Hp"] = df1.Hp
    df["Mp"] = df1.Mp

    # scaler:
    hmin, hmax = cfg.hs
    mmin, mmax = cfg.ms
    df["H"] = df.H * (hmax - hmin) + hmin
    df["M"] = df.M * (mmax - mmin) + mmin
    df["Hp"] = df.Hp * (hmax - hmin) + hmin
    df["Mp"] = df.Mp * (mmax - mmin) + mmin

    umin, umax = cfg.us
    fmin, fmax = cfg.fs
    df["U"] = df.U * (umax - umin) + umin
    df["Fi"] = df.Fi * (fmax - fmin) + fmin

    df.to_excel(pathout, index=False)


def genPred(model):
    for _ in cfg.tests:
        for i in range(1, 27):
            _1 = _ + str(i)
            validatesingle(model, _1)


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 200
    path_m = "./modelResult/transform_" + str(batch) + "_" + "pile.pth"
    print(path_m)
    model = KTransformer().to(device)
    model.load_state_dict(torch.load(path_m)["state_dict"])
    # res = validate(model, test="Test")  # 805
    genPred(model)

    # view_pic_result(path, case=0, config=configC)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Finish")