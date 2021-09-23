#-*-coding:utf-8-*-
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import device, args

class Client(object):
    def __init__(self, dataloader, idx):
        """
        args:
        dataset: dataset
        idx: client number
        """
        self.idx = idx
        self.dataloader = dataloader
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, model, epochs):
        """
        args:
        model: recieved model
        epochs: epoch in each client

        return:
        model state dict
        """

        print("This client number :{}".format(self.idx))
        model = copy.deepcopy(model)
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        for epoch in range(1, epochs+1, 1):
            print("Epoch:[{}] / [{}]".format(epoch, epochs))
            for img,label in tqdm(self.dataloader):
                img = img.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                out = model(img)
                loss = self.loss_func(out, label)
                loss.backward()

                optimizer.step()

        model.eval()
        return model.state_dict()
