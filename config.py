# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : config.py
@ Time    ：2022/12/14 11:29
"""
import torch

class config:
    def __init__(self):

        # For new model
        self.input_dim = 7
        self.output_dim = 2

        # for data
        self.pre = "./data"
        self.csvpred = "./csvpred"
        self.preTransform = "./dataTransform"  # "D:/PycharmProject/Idea1Transformer/dataTransform"  #
        self.last = ".xlsx"
        self.inter_a = ["/D6L30", "/D6L36", "/D6L42", "/D6L48", "/D6L54", "/D6L60", "/D6L66", "/D6L72", "/D8L36",
                        "/D8L42", "/D8L54", "/D10L36", "/D10L42", "/D10L54"]
        self.inter_b = ["-dense-", "-mdense-", "-loose-"]

        self.trains = ['/D6L30/D6L30-dense-', '/D6L30/D6L30-mdense-',
                       '/D6L36/D6L36-mdense-', '/D6L36/D6L36-loose-',
                       '/D6L42/D6L42-dense-', '/D6L42/D6L42-mdense-', '/D6L42/D6L42-loose-',
                       '/D6L48/D6L48-dense-', '/D6L48/D6L48-loose-',
                       '/D6L54/D6L54-dense-', '/D6L54/D6L54-mdense-', '/D6L54/D6L54-loose-',
                       '/D6L60/D6L60-dense-', '/D6L60/D6L60-mdense-',
                       '/D6L66/D6L66-mdense-', '/D6L66/D6L66-loose-',
                       '/D6L72/D6L72-dense-', '/D6L72/D6L72-mdense-', '/D6L72/D6L72-loose-',
                       '/D8L36/D8L36-dense-', '/D8L36/D8L36-mdense-', '/D8L36/D8L36-loose-',
                       '/D8L42/D8L42-dense-', '/D8L42/D8L42-loose-',
                       '/D8L54/D8L54-mdense-', '/D8L54/D8L54-loose-',
                       '/D10L36/D10L36-dense-', '/D10L36/D10L36-mdense-', '/D10L36/D10L36-loose-',
                       '/D10L42/D10L42-dense-', '/D10L42/D10L42-loose-',
                       '/D10L54/D10L54-dense-', '/D10L54/D10L54-mdense-', '/D10L54/D10L54-loose-']
        self.tests = ['/D6L36/D6L36-dense-', '/D6L48/D6L48-mdense-', '/D6L60/D6L60-loose-',
                      '/D6L66/D6L66-dense-', '/D8L42/D8L42-mdense-', '/D8L54/D8L54-dense-',
                      '/D10L42/D10L42-mdense-']
        self.max_n = 18

        # for min-max-scaler
        self.hs = (-794444000.0, 721258000.0)
        self.ms = (-18437000000.0, 20486200000.0)
        self.us = (-1.017, 1.235)
        self.fs = (-3.73582e-09, 0.00436002)
        self.LDs = (3.0, 12.0)
        self.Ls = (30, 72)
        self.Ds = (6, 10)


        # for training
        self.batch = 1000 # 200 # 1000
        self.num_epochs = 14 # 200
        self.num_workers = 0  # 多线程/ windows必须设置为0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters = (0.0001, 5, 0.5)
        self.num_folds = 10

        # saveing path for model
        self.pathm = "./modelResult/transform_"

        # prepare for earlystopping
        self.patience = 10


cfg = config()